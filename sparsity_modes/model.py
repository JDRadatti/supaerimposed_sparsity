"""
``SparsityModel`` — a wrapper around an ``nn.Module`` that manages multiple
sparsity, freeze, and training profiles.

Usage::

    sparse = SparsityModel(
        vgg.build_model(),
        mode_profiles=[...],
        freeze_profiles=[...],
        training_profiles=[...],
        output_options=OutputOptions(...),
    )

    # Training
    sparse.train(mode="standard_1_32", freeze="freeze_kept", training="default")

    # Evaluation (all registered modes)
    results = sparse.evaluate()

    # Simultaneous multi-mode training (Slimmable Neural Networks)
    sparse.train_all_modes(freeze="freeze_kept", training="default")

    # Inference in a specific mode
    with sparse.mode("mixed"):
        logits = sparse(images)

    # Persistence
    sparse.save("./runs/checkpoint.pt")
    loaded = SparsityModel.load("./runs/checkpoint.pt", vgg.build_model())
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._sparse import _nm_mask_conv2d_axis, _nm_mask_linear
from .profiles import (
    _validate_mode_profile,
    _validate_freeze_profile,
    _validate_training_profile,
)
from .output import OutputLogger


class SparsityModel:
    """Wraps an ``nn.Module`` with profile‑managed sparsity, freezing, and training.

    All profiles are validated at construction time against the inner model.
    The wrapper is transparent — attribute access falls through to the inner
    model via ``__getattr__``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: nn.Module,
        *,
        mode_profiles: list[Any] | None = None,
        freeze_profiles: list[Any] | None = None,
        training_profiles: list[Any] | None = None,
        output_options: Any | None = None,
    ) -> None:
        """Wrap *model* with the given profiles.

        Raises:
            ValueError: If any profile fails validation against *model*.
        """
        # Store inner model
        self._model: nn.Module = model

        # Profile storage (name → dataclass)
        self._mode_profiles: dict[str, Any] = {}
        self._freeze_profiles: dict[str, Any] = {}
        self._training_profiles: dict[str, Any] = {}

        # Active profile tracking (None / [] = off)
        self._active_mode: str | None = None
        self._active_freeze: list[str] = []
        self._active_training: str | None = None

        # Derived state
        # {mode_name: {module_fqn: "dense" | (N, M)}}
        self._mode_layer_map: dict[str, dict[str, Any]] = {}

        # Per-mode BN buffers: {mode_name: {module_fqn: {running_mean, ...}}}
        self._bn_buffers: dict[str, dict[str, dict[str, torch.Tensor]]] = {}

        # Per-instance freeze state (not global).
        self._frozen_params: set[nn.Parameter] = set()
        self._element_freeze_masks: dict[nn.Parameter, torch.Tensor] = {}

        # Output
        self._output_options: Any = output_options
        self._logger: OutputLogger | None = None
        if output_options is not None:
            self._logger = OutputLogger(output_options)

        # Validation & mask injection
        mode_profiles = mode_profiles or []
        freeze_profiles = freeze_profiles or []
        training_profiles = training_profiles or []

        # Validate and register mode profiles.
        for mp in mode_profiles:
            if mp.name in self._mode_profiles:
                raise ValueError(f"Duplicate ModeProfile name: {mp.name!r}")
            layer_map = _validate_mode_profile(model, mp)
            self._mode_layer_map[mp.name] = layer_map
            self._mode_profiles[mp.name] = mp
            _inject_masks(model, layer_map, mp.conv_axis, mp.linear_orientation)

        # Validate and register freeze profiles.
        mode_names = set(self._mode_profiles.keys())
        for fp in freeze_profiles:
            if fp.name in self._freeze_profiles:
                raise ValueError(f"Duplicate FreezeProfile name: {fp.name!r}")
            _validate_freeze_profile(model, fp, mode_names)
            self._freeze_profiles[fp.name] = fp

        # Validate and register training profiles.
        for tp in training_profiles:
            if tp.name in self._training_profiles:
                raise ValueError(f"Duplicate TrainingProfile name: {tp.name!r}")
            _validate_training_profile(tp)
            self._training_profiles[tp.name] = tp

    # ------------------------------------------------------------------
    # Delegation to inner model
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the inner ``nn.Module``.

        This makes ``sparse.to(device)``, ``sparse.parameters()``,
        ``sparse.train()`` etc. work without explicit forwarding.
        """
        inner = self.__dict__.get("_model", None)
        if inner is not None:
            return getattr(inner, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass — delegates to the inner model's ``forward``."""
        return self._model(*args, **kwargs)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def mode(self, name: str):
        """Temporarily activate *name* as the active mode profile.

        Usage::

            with sparse.mode("mixed"):
                outputs = sparse(images)

        Saves the current BN buffers for the previous mode, restores BN
        buffers for *name*, and on exit restores the previous mode and
        its BN buffers.
        """
        prev = self._active_mode
        if prev is not None:
            self._save_bn_buffers()
        self._set_mode_profile(name)
        try:
            yield
        finally:
            if name is not None:
                self._save_bn_buffers()
            if prev is not None:
                self._set_mode_profile(prev)
            else:
                self._set_mode_profile(None)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def train(
        self,
        mode: str,
        freeze: str | list[str] | None,
        training: str,
        *,
        epochs: int | None = None,
        train_loader: Any = None,
        val_loader: Any = None,
    ) -> dict[str, Any]:
        """Train the model using the named profiles.

        Parameters
        ----------
        mode:       Name of a registered ``ModeProfile``.
        freeze:     Name(s) of registered ``FreezeProfile``(s), or ``None``.
                    If a list is given, each profile is applied cumulatively.
        training:   Name of a registered ``TrainingProfile``.
        epochs:     Override the epoch count from the training profile.
        train_loader: Optional DataLoader (random data used if ``None``).
        val_loader:   Optional DataLoader for validation.

        Returns:
            Dict with keys ``"final_train_loss"``, ``"final_val_loss"``,
            ``"final_val_acc"``, ``"epochs"``.
        """
        if mode not in self._mode_profiles:
            raise ValueError(f"Unknown ModeProfile: {mode!r}")
        freeze_names = [freeze] if isinstance(freeze, str) else (freeze or [])
        for fn in freeze_names:
            if fn not in self._freeze_profiles:
                raise ValueError(f"Unknown FreezeProfile: {fn!r}")
        if training not in self._training_profiles:
            raise ValueError(f"Unknown TrainingProfile: {training!r}")

        train_profile = self._training_profiles[training]
        n_epochs = epochs if epochs is not None else train_profile.epochs

        # 1. Activate profiles.
        self._set_mode_profile(mode)
        self._clear_freeze_state()
        self._active_freeze = []
        for fn in freeze_names:
            self._set_freeze_profile(fn)

        # 2. Build optimizer, scheduler, loss.
        optimizer = train_profile.optimizer_fn(self._model.parameters())
        scheduler = train_profile.scheduler_fn(optimizer) if train_profile.scheduler_fn is not None else None
        loss_fn = train_profile.loss_fn

        # 2b. Ensure logger run is started.
        if self._logger is not None and self._logger.run_dir is None:
            self._logger.start_run(
                f"train_{mode}",
                profiles_info=_build_profiles_info(self),
                model_info=_build_model_info(self),
            )

        # 3. Synthetic dataloader fallback.
        if train_loader is None:
            import torch.utils.data as tud
            class _RandDataset(tud.Dataset):
                def __len__(self): return 100
                def __getitem__(self, i): return torch.randn(3, 32, 32), torch.randint(0, 10, ())

            train_loader = tud.DataLoader(
                _RandDataset(), batch_size=train_profile.batch_size,
                num_workers=train_profile.num_workers,
            )

        # 4. Training loop.
        final_train_loss = 0.0
        final_val_loss = 0.0
        final_val_acc = 0.0
        self._model.train()

        import time as _time

        for epoch in range(1, n_epochs + 1):
            total, correct, loss_sum = 0, 0, 0.0
            device = next(self._model.parameters()).device

            for batch_idx, batch in enumerate(train_loader):
                if isinstance(batch, (list, tuple)):
                    images, targets = batch[0], batch[1]
                else:
                    images, targets = batch, torch.zeros(batch.size(0), dtype=torch.long)

                images = images.to(device)
                targets = targets.to(device)

                if targets.dim() > 1:
                    targets = targets.view(-1)

                _t0 = _time.time()
                optimizer.zero_grad()
                _t1 = _time.time()
                outputs = self._model(images)
                loss = loss_fn(outputs, targets)
                loss.backward()
                _t2 = _time.time()
                self._zero_frozen_gradients(optimizer)
                frozen_snap = self._snapshot_frozen_positions()
                optimizer.step()
                self._restore_frozen_positions(frozen_snap)
                _t3 = _time.time()

                if epoch <= 2 and batch_idx < 5:
                    print(f"  [TIMING] ep={epoch} batch={batch_idx}: "
                          f"zero_grad={_t1-_t0:.4f}s  fwd+bwd={_t2-_t1:.4f}s  "
                          f"freeze_step={_t3-_t2:.4f}s  total={_t3-_t0:.4f}s")

                loss_sum += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += images.size(0)

            if scheduler is not None:
                scheduler.step()

            train_acc = 100.0 * correct / total if total > 0 else 0.0
            train_loss = loss_sum / total if total > 0 else 0.0
            final_train_loss = train_loss

            # Validation.
            # if val_loader is not None:
            #     vl, va = self._eval_on_loader(val_loader)
            #     final_val_loss, final_val_acc = vl, va
            # else:
            #     vl, va = 0.0, 0.0

            # Log.
            lr = optimizer.param_groups[0]["lr"]
            if self._logger is not None:
                self._logger.log_epoch(epoch, train_loss, train_acc, vl, va, lr)

            print(f"Epoch {epoch:3d}:  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
                  f"val_loss={vl:.4f}  val_acc={va:.2f}%  lr={lr:.6f}")

        # 5. Save BN buffers for the trained mode.
        self._save_bn_buffers()

        # 6. Write overview.
        self._write_overview()

        return {
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc,
            "epochs": n_epochs,
        }

    def _eval_on_loader(self, loader: Any) -> tuple[float, float]:
        """Run evaluation on *loader* and return ``(loss, accuracy)``."""
        self._model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        loss_fn = torch.nn.CrossEntropyLoss()
        device = next(self._model.parameters()).device
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    images, targets = batch[0], batch[1]
                else:
                    images, targets = batch, torch.zeros(batch.size(0), dtype=torch.long)
                images, targets = images.to(device), targets.to(device)
                if targets.dim() > 1:
                    targets = targets.view(-1)
                outputs = self._model(images)
                loss_sum += loss_fn(outputs, targets).item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += images.size(0)
        self._model.train()
        avg_loss = loss_sum / total if total > 0 else 0.0
        acc = 100.0 * correct / total if total > 0 else 0.0
        return avg_loss, acc

    def evaluate(self, val_loader: Any = None) -> dict[str, tuple[float, float]]:
        """Evaluate the model in **all registered mode profiles**.

        For each mode the corresponding BN buffers are restored.  The
        previously active mode is restored after evaluation.

        Returns:
            ``{mode_name: (avg_loss, accuracy)}`` for every registered
            ``ModeProfile`` (plus ``"dense"`` if not registered).
        """
        prev_mode = self._active_mode
        if prev_mode is not None:
            self._save_bn_buffers()

        # Ensure logger run is started.
        if self._logger is not None and self._logger.run_dir is None:
            self._logger.start_run(
                "evaluate",
                profiles_info=_build_profiles_info(self),
                model_info=_build_model_info(self),
            )

        # Synthetic val_loader fallback.
        if val_loader is None:
            import torch.utils.data as tud
            class _RandDataset(tud.Dataset):
                def __len__(self): return 100
                def __getitem__(self, i): return torch.randn(3, 32, 32), torch.randint(0, 10, ())

            val_loader = tud.DataLoader(
                _RandDataset(), batch_size=64, num_workers=0,
            )

        results: dict[str, tuple[float, float]] = {}
        mode_names = sorted(self._mode_profiles.keys())

        for mn in mode_names:
            self._set_mode_profile(mn)
            vl, va = self._eval_on_loader(val_loader)
            results[mn] = (vl, va)
            self._save_bn_buffers()

        # "dense" evaluation.
        self._set_mode_profile(None)
        vl, va = self._eval_on_loader(val_loader)
        results["dense"] = (vl, va)

        # Restore previous mode.
        if prev_mode is not None:
            self._set_mode_profile(prev_mode)
        else:
            self._set_mode_profile(None)

        if self._logger is not None:
            self._logger.log_eval_summary(results)

        for mn, (vl, va) in results.items():
            print(f"[EVAL] {mn:<16}  loss={vl:.4f}  acc={va:.2f}%")

        self._write_overview()

        return results

    def train_all_modes(
        self,
        freeze: str | list[str] | None,
        training: str,
        *,
        mode_names: list[str] | None = None,
        epochs: int | None = None,
        train_loader: Any = None,
        val_loader: Any = None,
    ) -> dict[str, Any]:
        """Train the model at **all sparsity levels simultaneously**.

        Implements the training scheme of *Slimmable Neural Networks*
        (Yu et al., ICLR 2019): for each mini‑batch every registered mode
        profile is activated in turn, the loss is computed and gradients are
        accumulated via ``loss.backward()``.  A single ``optimizer.step()``
        is issued after all modes have been visited, so that weights are
        jointly optimised for every sparsity level at once.

        Per‑mode BatchNorm statistics are maintained via ``_bn_buffers``,
        mimicking the paper's *Switchable Batch Normalisation*.  After
        training, each mode's BN running statistics are saved.

        Parameters
        ----------
        freeze:     Name(s) of registered ``FreezeProfile``(s), or ``None``.
                    If a list is given, each profile is applied cumulatively.
        training:   Name of a registered ``TrainingProfile``.
        mode_names: Subset of mode profiles to include.  ``None`` = all
                    registered profiles.
        epochs:     Override the epoch count from the training profile.
        train_loader: Optional DataLoader (random data used if ``None``).
        val_loader:   Optional DataLoader for validation.

        Returns:
            ``{mode_name: {"val_loss": float, "val_acc": float,
            "train_loss": float, "train_acc": float}}`` per mode.
        """
        # 1. Lookup profiles.
        freeze_names = [freeze] if isinstance(freeze, str) else (freeze or [])
        for fn in freeze_names:
            if fn not in self._freeze_profiles:
                raise ValueError(f"Unknown FreezeProfile: {fn!r}")
        if training not in self._training_profiles:
            raise ValueError(f"Unknown TrainingProfile: {training!r}")

        train_profile = self._training_profiles[training]
        n_epochs = epochs if epochs is not None else train_profile.epochs

        # Determine which mode profiles to train.
        names = mode_names if mode_names is not None else sorted(self._mode_profiles.keys())
        for name in names:
            if name not in self._mode_profiles:
                raise ValueError(f"Unknown ModeProfile: {name!r}")
        if not names:
            raise ValueError("No mode profiles to train.")

        # Save the currently active mode for later restoration.
        prev_mode = self._active_mode
        if prev_mode is not None:
            self._save_bn_buffers()

        # 2. Activate freeze profile (shared across all modes).
        self._clear_freeze_state()
        self._active_freeze = []
        for fn in freeze_names:
            self._set_freeze_profile(fn)

        # 3. Build optimizer, scheduler from train_profile.
        optimizer = train_profile.optimizer_fn(self._model.parameters())
        scheduler = train_profile.scheduler_fn(optimizer) if train_profile.scheduler_fn is not None else None
        loss_fn = train_profile.loss_fn

        # Ensure logger run is started.
        if self._logger is not None and self._logger.run_dir is None:
            self._logger.start_run(
                "train_all_modes",
                profiles_info=_build_profiles_info(self),
                model_info=_build_model_info(self),
            )

        # 4. Synthetic dataloader fallback.
        if train_loader is None:
            import torch.utils.data as tud
            class _RandDataset(tud.Dataset):
                def __len__(self): return 100
                def __getitem__(self, i): return torch.randn(3, 32, 32), torch.randint(0, 10, ())

            train_loader = tud.DataLoader(
                _RandDataset(), batch_size=train_profile.batch_size,
                num_workers=train_profile.num_workers,
            )

        device = next(self._model.parameters()).device

        # Per-mode accumulators for metrics.
        mode_metrics: dict[str, dict[str, float]] = {
            mn: {"train_loss_sum": 0.0, "train_correct": 0, "train_total": 0}
            for mn in names
        }

        # 5. Training loop — gradient accumulation across modes.
        for epoch in range(1, n_epochs + 1):
            self._model.train()
            # Zero gradients once per batch.
            for mn in names:
                mode_metrics[mn]["train_loss_sum"] = 0.0
                mode_metrics[mn]["train_correct"] = 0
                mode_metrics[mn]["train_total"] = 0

            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    images, targets = batch[0], batch[1]
                else:
                    images, targets = batch, torch.zeros(batch.size(0), dtype=torch.long)

                images = images.to(device)
                targets = targets.to(device)
                if targets.dim() > 1:
                    targets = targets.view(-1)

                optimizer.zero_grad()

                for mn in names:
                    # Switch mode profile + restore its BN buffers.
                    self._set_mode_profile(mn)

                    # Forward + backward (gradients accumulate across modes).
                    outputs = self._model(images)
                    loss = loss_fn(outputs, targets)
                    loss.backward()

                    # Track per-mode metrics.
                    bs = images.size(0)
                    mode_metrics[mn]["train_loss_sum"] += loss.item() * bs
                    _, predicted = outputs.max(1)
                    mode_metrics[mn]["train_correct"] += predicted.eq(targets).sum().item()
                    mode_metrics[mn]["train_total"] += bs

                # One optimizer step for all modes.
                self._zero_frozen_gradients(optimizer)
                frozen_snap = self._snapshot_frozen_positions()
                optimizer.step()
                self._restore_frozen_positions(frozen_snap)

            if scheduler is not None:
                scheduler.step()

            # After each epoch: validation sweep across all modes.
            val_results: dict[str, tuple[float, float]] = {}
            if val_loader is not None:
                for mn in names:
                    self._set_mode_profile(mn)
                    self._restore_bn_buffers(mn)
                    vl, va = self._eval_on_loader(val_loader)
                    val_results[mn] = (vl, va)

            # Print per-mode epoch summary.
            for mn in names:
                mm = mode_metrics[mn]
                tl = mm["train_loss_sum"] / mm["train_total"] if mm["train_total"] > 0 else 0.0
                ta = 100.0 * mm["train_correct"] / mm["train_total"] if mm["train_total"] > 0 else 0.0
                vl, va = val_results.get(mn, (0.0, 0.0))
                lr = optimizer.param_groups[0]["lr"]
                if self._logger is not None:
                    self._logger.log_epoch(epoch, tl, ta, vl, va, lr)
                print(f"  [{mn}] Epoch {epoch:3d}:  "
                      f"train_loss={tl:.4f}  train_acc={ta:.2f}%  "
                      f"val_loss={vl:.4f}  val_acc={va:.2f}%  lr={lr:.6f}")

        # 6. Save BN buffers for every trained mode.
        for mn in names:
            self._set_mode_profile(mn)
            self._save_bn_buffers()

        # 7. Restore previous mode.
        if prev_mode is not None:
            self._set_mode_profile(prev_mode)
        else:
            self._set_mode_profile(None)

        # 8. Build return dict with per-mode final metrics.
        result: dict[str, Any] = {}
        for mn in names:
            mm = mode_metrics[mn]
            tl = mm["train_loss_sum"] / mm["train_total"] if mm["train_total"] > 0 else 0.0
            ta = 100.0 * mm["train_correct"] / mm["train_total"] if mm["train_total"] > 0 else 0.0
            vl, va = val_results.get(mn, (0.0, 0.0))
            result[mn] = {
                "train_loss": tl,
                "train_acc": ta,
                "val_loss": vl,
                "val_acc": va,
            }

        self._write_overview()

        return result

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, first_n: int = 50) -> None:
        """Write per‑mode CSV weight visualizations."""
        from .visualize import visualize as _vis
        _vis(self, first_n=first_n)

    def visualize_diff(
        self,
        other: SparsityModel,
        mode_name: str | None = None,
        first_n: int = 50,
    ) -> None:
        """Write a CSV comparing weights against *other* model."""
        from .visualize import visualize_diff as _vdiff
        _vdiff(self, other, mode_name=mode_name, first_n=first_n)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model state, masks, BN buffers, and profiles to *path*.

        All active profiles are deactivated before saving so the checkpoint
        contains no transient mode‑specific state.  Profiles are persisted
        as metadata and re‑registered on load.
        """
        # 1. Deactivate all profiles.
        self._set_mode_profile(None)
        self._set_freeze_profile(None)
        self._active_training = None

        # 2. Collect masks from every module.
        masks: dict[str, dict[str, torch.Tensor]] = {}
        for mname, module in self._model.named_modules():
            if not hasattr(module, "nm_masks"):
                continue
            saved = {k: v.cpu() for k, v in module.nm_masks.items() if v is not None}
            if saved:
                masks[mname] = saved

        # 3. BN buffers — move to CPU.
        bn_bufs: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        for mode_name, mdict in self._bn_buffers.items():
            bn_bufs[mode_name] = {
                mname: {bk: bv.cpu() for bk, bv in stats.items()}
                for mname, stats in mdict.items()
            }

        # 4. Serialize profiles.
        checkpoint: dict[str, Any] = {
            "model_state_dict": {k: v.cpu() for k, v in self._model.state_dict().items()},
            "masks": masks,
            "bn_buffers": bn_bufs,
            "mode_profiles": [
                _serialize_mode_profile(mp) for mp in self._mode_profiles.values()
            ],
            "freeze_profiles": [
                _serialize_freeze_profile(fp) for fp in self._freeze_profiles.values()
            ],
            "training_profiles": [
                _serialize_training_profile(tp) for tp in self._training_profiles.values()
            ],
            "output_options": _serialize_output_options(self._output_options),
            "v3_version": 1,
        }

        torch.save(checkpoint, path)
        print(f"[SAVE] Checkpoint saved to {path}")

    @classmethod
    def load(cls, path: str, model: nn.Module) -> SparsityModel:
        """Load a ``SparsityModel`` from a checkpoint created by ``save()``.

        The *model* argument supplies a fresh architecture instance with the
        same structure as the saved model.  Profiles and BN buffers are
        restored, but no profile is active — call ``_set_mode_profile()`` or
        ``train()`` to activate one.

        Returns:
            A new ``SparsityModel`` with all profiles loaded and deactivated.
        """
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")

        # 1. Reconstruct profiles.
        mode_profiles = [
            _deserialize_mode_profile(d) for d in checkpoint.get("mode_profiles", [])
        ]
        freeze_profiles = [
            _deserialize_freeze_profile(d) for d in checkpoint.get("freeze_profiles", [])
        ]
        training_profiles = [
            _deserialize_training_profile(d) for d in checkpoint.get("training_profiles", [])
        ]
        output_options = _deserialize_output_options(
            checkpoint.get("output_options", {})
        )

        # 2. Build SparsityModel — validation + mask injection runs.
        sparse = cls(
            model,
            mode_profiles=mode_profiles,
            freeze_profiles=freeze_profiles,
            training_profiles=training_profiles,
            output_options=output_options,
        )

        # 3. Load state dict.
        sparse._model.load_state_dict(checkpoint["model_state_dict"])

        # 4. Overwrite masks with saved copies (those computed at init
        #    time used the random initial weights, not the loaded ones).
        name_to_module = dict(sparse._model.named_modules())
        for mname, saved_masks in checkpoint.get("masks", {}).items():
            module = name_to_module[mname]
            for key, mask in saved_masks.items():
                module.nm_masks[key] = mask
            _patch_forward_for_masks(module)

        # 5. Restore BN buffers.
        for mode_name, mdict in checkpoint.get("bn_buffers", {}).items():
            sparse._bn_buffers[mode_name] = {
                mname: {bk: bv for bk, bv in stats.items()}
                for mname, stats in mdict.items()
            }

        print(f"[LOAD] Checkpoint loaded from {path}")

        return sparse

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines: list[str] = []
        lines.append(f"SparsityModel(inner={type(self._model).__name__})")

        # Mode profiles
        if self._mode_profiles:
            lines.append("  Mode profiles:")
            for name, mp in self._mode_profiles.items():
                active = " (active)" if name == self._active_mode else ""
                desc = _describe_mode_profile(mp, self._mode_layer_map.get(name, {}))
                lines.append(f"    {name}{active}: {desc}")
        else:
            lines.append("  Mode profiles: (none)")

        # Freeze profiles
        if self._freeze_profiles:
            lines.append("  Freeze profiles:")
            for name, fp in self._freeze_profiles.items():
                active = " (active)" if name in self._active_freeze else ""
                dep = f" → mode={fp.mode_profile}" if fp.mode_profile else ""
                lines.append(f"    {name}{active}: {fp.freeze_type}{dep}")
        else:
            lines.append("  Freeze profiles: (none)")

        # Training profiles
        if self._training_profiles:
            lines.append("  Training profiles:")
            for name, tp in self._training_profiles.items():
                active = " (active)" if name == self._active_training else ""
                opt_desc = _describe_optimizer(tp)
                lines.append(f"    {name}{active}: {tp.epochs} epochs, {opt_desc}")
        else:
            lines.append("  Training profiles: (none)")

        # Active state
        lines.append(
            f"  Active: mode={self._active_mode or 'none'}, "
            f"freeze={', '.join(self._active_freeze) if self._active_freeze else 'none'}, "
            f"training={self._active_training or 'none'}"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private: mode profile activation
    # ------------------------------------------------------------------

    def _set_mode_profile(self, name: str | None) -> None:
        """Activate (or deactivate) a mode profile.

        ``None`` restores every module to dense behaviour.

        Raises:
            ValueError: If *name* is not a registered mode profile.
        """
        if name is not None and name not in self._mode_profiles:
            raise ValueError(
                f"Unknown ModeProfile: {name!r}. "
                f"Available: {sorted(self._mode_profiles.keys())}."
            )

        # Save BN buffers for the currently active mode.
        if self._active_mode is not None:
            self._save_bn_buffers()

        # Build a name→module lookup once.
        name_to_module = dict(self._model.named_modules())

        if name is None:
            # Set all patched modules to dense.
            self._active_mode = None
            for module in self._model.modules():
                if hasattr(module, "active_mask_key"):
                    module.active_mask_key = "dense"
        else:
            # Restore BN buffers for the new mode.
            self._restore_bn_buffers(name)
            self._active_mode = name
            # Set each layer's active mask according to the layer map.
            layer_map = self._mode_layer_map[name]
            for module_fqn, sparsity in layer_map.items():
                module = name_to_module[module_fqn]
                if not hasattr(module, "active_mask_key"):
                    module.active_mask_key = "dense"
                if sparsity == "dense":
                    module.active_mask_key = "dense"
                else:
                    key = f"{sparsity[0]}:{sparsity[1]}"
                    module.active_mask_key = key

    # ------------------------------------------------------------------
    # Private: freeze profile activation
    # ------------------------------------------------------------------

    def _set_freeze_profile(self, name: str | None) -> None:
        """Activate a freeze profile (cumulative) or clear all.

        Each call adds to the active set — element‑freeze masks OR‑together
        and fully‑frozen params union.  Pass ``None`` to clear everything.

        Raises:
            ValueError: If *name* is not a registered freeze profile.
        """
        if name is None:
            self._clear_freeze_state()
            self._active_freeze = []
            return

        if name not in self._freeze_profiles:
            raise ValueError(
                f"Unknown FreezeProfile: {name!r}. "
                f"Available: {sorted(self._freeze_profiles.keys())}."
            )

        self._active_freeze.append(name)
        fp = self._freeze_profiles[name]
        ft = fp.freeze_type

        if ft == "all":
            _freeze_all(self)
        elif ft == "biases":
            _freeze_biases(self)
        elif ft == "batchnorm":
            _freeze_batchnorm(self)
        elif ft == "mask":
            if fp.mode_profile is None:
                raise ValueError(
                    f"FreezeProfile '{name}': freeze_type='mask' "
                    f"requires mode_profile."
                )
            _freeze_mask(self, fp.mode_profile)
        elif ft == "inverse_mask":
            if fp.mode_profile is None:
                raise ValueError(
                    f"FreezeProfile '{name}': freeze_type='inverse_mask' "
                    f"requires mode_profile."
                )
            _freeze_inverse_mask(self, fp.mode_profile)
        elif ft == "non_masked":
            if fp.mode_profile is None:
                raise ValueError(
                    f"FreezeProfile '{name}': freeze_type='non_masked' "
                    f"requires mode_profile."
                )
            _freeze_non_masked(self, fp.mode_profile)

    # ------------------------------------------------------------------
    # Private: BN buffer management
    # ------------------------------------------------------------------

    def _save_bn_buffers(self) -> None:
        """Save the current BN running statistics under ``self._active_mode``.

        If no mode is active this is a no‑op.
        """
        if self._active_mode is None:
            return

        stats: dict[str, dict[str, torch.Tensor]] = {}
        for name, module in self._model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                stats[name] = {
                    "running_mean": module.running_mean.detach().clone(),
                    "running_var": module.running_var.detach().clone(),
                    "num_batches_tracked": module.num_batches_tracked.detach().clone(),
                }
        self._bn_buffers[self._active_mode] = stats

    def _restore_bn_buffers(self, mode_name: str) -> None:
        """Restore previously saved BN buffers for *mode_name*.

        No‑op if no buffers have been saved for this mode.
        """
        if mode_name not in self._bn_buffers:
            return

        saved = self._bn_buffers[mode_name]
        for name, module in self._model.named_modules():
            if name in saved:
                module.running_mean.copy_(saved[name]["running_mean"])
                module.running_var.copy_(saved[name]["running_var"])
                module.num_batches_tracked.copy_(saved[name]["num_batches_tracked"])

    # ------------------------------------------------------------------
    # Private: freeze state cleanup
    # ------------------------------------------------------------------

    def _clear_freeze_state(self) -> None:
        """Clear all per‑parameter and element‑wise freeze registrations."""
        for p in list(self._frozen_params):
            if hasattr(p, "_frozen"):
                delattr(p, "_frozen")
        self._frozen_params.clear()
        self._element_freeze_masks.clear()

    def _zero_frozen_gradients(self, optimizer: Any) -> None:
        """Zero gradients for fully‑frozen params and element‑frozen positions."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None and p in self._frozen_params:
                    p.grad.zero_()
        for p, freeze_mask in self._element_freeze_masks.items():
            if p.grad is not None:
                p.grad[freeze_mask] = 0.0

    def _snapshot_frozen_positions(self) -> dict:
        """Cache current values of all frozen positions before ``optimizer.step()``.

        Weight decay modifies values despite zero gradients, so we save
        before the step and restore afterwards.
        """
        snap: dict[int, torch.Tensor] = {}
        # Fully frozen params first — full clone.
        for p in self._frozen_params:
            snap[id(p)] = p.data.clone()
        # Element freeze only for params not already fully frozen.
        for p, freeze_mask in self._element_freeze_masks.items():
            if id(p) not in snap and freeze_mask.any():
                snap[id(p)] = p.data[freeze_mask].clone()
        return snap

    def _restore_frozen_positions(self, snap: dict) -> None:
        """Restore frozen positions from a snapshot."""
        # Element freeze — skip params that are fully frozen.
        for p, freeze_mask in self._element_freeze_masks.items():
            if p in self._frozen_params:
                continue
            pid = id(p)
            if pid in snap:
                p.data[freeze_mask] = snap[pid]
        # Fully frozen params.
        for p in self._frozen_params:
            pid = id(p)
            if pid in snap:
                p.data[:] = snap[pid]

    # ------------------------------------------------------------------
    # Private: overview generation
    # ------------------------------------------------------------------

    def _write_overview(self) -> None:
        """Write ``overview.txt`` into the current run directory."""
        if self._logger is None or self._logger.run_dir is None:
            return

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"  OVERVIEW")
        lines.append(f"  Run: {self._logger.run_dir}")
        lines.append(f"  Timestamp: {self._logger._timestamp}")
        lines.append(f"  Git: {self._logger.commit} on branch {self._logger.branch}")
        lines.append(f"  Model: {type(self._model).__name__}")
        total_params = sum(p.numel() for p in self._model.parameters())
        lines.append(f"  Total parameters: {total_params:,}")
        lines.append("=" * 60)

        # Mode profiles
        lines.append("")
        lines.append("  Mode Profiles:")
        if self._mode_profiles:
            for name, mp in self._mode_profiles.items():
                layer_map = self._mode_layer_map.get(name, {})
                desc = _describe_mode_profile(mp, layer_map)
                lines.append(f"    {name}: {desc}")
        else:
            lines.append("    (none)")

        # Freeze profiles
        lines.append("")
        lines.append("  Freeze Profiles:")
        if self._freeze_profiles:
            for name, fp in self._freeze_profiles.items():
                dep = f" -> mode={fp.mode_profile}" if fp.mode_profile else ""
                lines.append(f"    {name}: {fp.freeze_type}{dep}")
        else:
            lines.append("    (none)")

        # Training profiles
        lines.append("")
        lines.append("  Training Profiles:")
        if self._training_profiles:
            for name, tp in self._training_profiles.items():
                opt_desc = _describe_optimizer(tp)
                lines.append(f"    {name}: {tp.epochs} epochs, {opt_desc}")
        else:
            lines.append("    (none)")

        # Active state
        lines.append("")
        lines.append("  Active State:")
        lines.append(f"    mode: {self._active_mode or 'none'}")
        lines.append(f"    freeze: {', '.join(self._active_freeze) if self._active_freeze else 'none'}")
        lines.append(f"    training: {self._active_training or 'none'}")

        lines.append("")
        lines.append("=" * 60)

        self._logger.write_overview("\n".join(lines))


# ===================================================================
# Freeze helpers
# ===================================================================

def _freeze_all(sparse: SparsityModel) -> None:
    """Freeze every parameter in the inner model."""
    count = 0
    for p in sparse._model.parameters():
        sparse._frozen_params.add(p)
        setattr(p, "_frozen", True)
        count += 1
    print(f"[FREEZE] All {count} parameters frozen.")


def _freeze_biases(sparse: SparsityModel) -> None:
    """Freeze all bias parameters."""
    count = 0
    for module in sparse._model.modules():
        if hasattr(module, "bias") and module.bias is not None:
            sparse._frozen_params.add(module.bias)
            setattr(module.bias, "_frozen", True)
            count += 1
    print(f"[FREEZE] {count} bias parameters frozen.")


def _freeze_batchnorm(sparse: SparsityModel) -> None:
    """Freeze all BatchNorm weight and bias parameters."""
    count = 0
    for module in sparse._model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if module.weight is not None:
                sparse._frozen_params.add(module.weight)
                setattr(module.weight, "_frozen", True)
                count += 1
            if module.bias is not None:
                sparse._frozen_params.add(module.bias)
                setattr(module.bias, "_frozen", True)
                count += 1
    print(f"[FREEZE] {count} BatchNorm parameters frozen.")


def _freeze_mask(sparse: SparsityModel, mode_name: str) -> None:
    """Element‑freeze weight positions **kept** by the mask (mask == 1)."""
    layer_map = sparse._mode_layer_map[mode_name]
    name_to_module = dict(sparse._model.named_modules())
    count = 0

    for module_fqn, sparsity in layer_map.items():
        if sparsity == "dense":
            continue
        module = name_to_module[module_fqn]
        key = f"{sparsity[0]}:{sparsity[1]}"
        mask = module.nm_masks.get(key)
        if mask is None:
            continue
        fm = mask.bool().to(module.weight.device)
        if module.weight in sparse._element_freeze_masks:
            sparse._element_freeze_masks[module.weight] |= fm
        else:
            sparse._element_freeze_masks[module.weight] = fm
        count += 1
    print(f"[FREEZE] {count} modules mask '{mode_name}' frozen (kept positions).")


def _freeze_inverse_mask(sparse: SparsityModel, mode_name: str) -> None:
    """Element‑freeze weight positions **pruned** by the mask (mask == 0)."""
    layer_map = sparse._mode_layer_map[mode_name]
    name_to_module = dict(sparse._model.named_modules())
    count = 0

    for module_fqn, sparsity in layer_map.items():
        if sparsity == "dense":
            continue
        module = name_to_module[module_fqn]
        key = f"{sparsity[0]}:{sparsity[1]}"
        mask = module.nm_masks.get(key)
        if mask is None:
            continue
        fm = mask.eq(0).to(module.weight.device)
        if module.weight in sparse._element_freeze_masks:
            sparse._element_freeze_masks[module.weight] |= fm
        else:
            sparse._element_freeze_masks[module.weight] = fm
        count += 1
    print(f"[FREEZE] {count} modules inverse mask '{mode_name}' frozen (pruned positions).")


def _freeze_non_masked(sparse: SparsityModel, mode_name: str) -> None:
    """Full‑parameter freeze of Conv/Linear weights not in *mode_name*'s layer_map."""
    layer_map = sparse._mode_layer_map[mode_name]
    name_to_module = dict(sparse._model.named_modules())
    count = 0

    for module_fqn, module in name_to_module.items():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if module_fqn in layer_map:
            continue
        if module.weight is not None:
            sparse._frozen_params.add(module.weight)
            setattr(module.weight, "_frozen", True)
            count += 1
    print(f"[FREEZE] {count} non‑masked layers frozen.")


# ===================================================================
# Description helpers
# ===================================================================

def _describe_mode_profile(
    profile: Any,
    layer_map: dict[str, Any],
) -> str:
    """Return a compact string describing a mode profile's layer assignments."""
    parts: list[str] = []
    for name, sparsity in layer_map.items():
        s = sparsity if isinstance(sparsity, str) else f"{sparsity[0]}:{sparsity[1]}"
        parts.append(f"{name}={s}")
    return ", ".join(parts)


def _describe_optimizer(tp: Any) -> str:
    """Return a compact string describing a training profile's optimizer config."""
    try:
        # Try to inspect the optimizer_fn default to get lr/momentum.
        import inspect
        sig = inspect.signature(tp.optimizer_fn)
        # The default is a lambda: optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4)
        default_opt = tp.optimizer_fn([])
        opt_type = type(default_opt).__name__
        pg = default_opt.param_groups[0] if default_opt.param_groups else {}
        lr = pg.get("lr", "?")
        parts = [f"{opt_type}(lr={lr})"]
        if tp.scheduler_fn is not None:
            try:
                default_sched = tp.scheduler_fn(default_opt)
                sched_type = type(default_sched).__name__
                parts.append(sched_type)
            except Exception:
                parts.append("scheduler")
        if tp.label_smoothing > 0:
            parts.append(f"label_smoothing={tp.label_smoothing}")
        if tp.cutout > 0:
            parts.append(f"cutout={tp.cutout}")
        return ", ".join(parts)
    except Exception:
        return f"lr=?, epochs={tp.epochs}"


# ===================================================================
# Mask injection helpers
# ===================================================================

def _patch_forward_for_masks(module: nn.Module) -> None:
    """Patch ``module.forward`` so the active N:M mask is applied at runtime.

    Must be called after ``module.nm_masks`` has been populated.
    Only patches once — safe to call repeatedly.
    """
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return
    if not hasattr(module, "nm_masks"):
        return
    if hasattr(module, "_original_forward"):
        return  # already patched

    module._original_forward = module.forward
    module.active_mask_key = "dense"

    def _make_forward(m):
        def _forward(input):
            key = getattr(m, "active_mask_key", "dense")
            if key == "dense":
                return m._original_forward(input)

            mask = m.nm_masks.get(key)
            if mask is None:
                return m._original_forward(input)

            masked_weight = m.weight * mask.to(m.weight.device)
            if isinstance(m, nn.Linear):
                return F.linear(input, masked_weight, m.bias)
            return F.conv2d(
                input, masked_weight, m.bias,
                m.stride, m.padding, m.dilation, m.groups,
            )

        return _forward

    module.__dict__["forward"] = _make_forward(module)


def _inject_masks(
    model: nn.Module,
    layer_map: dict[str, tuple[int, int] | str],
    conv_axis: str,
    linear_orientation: str,
) -> None:
    """Pre‑compute and register N:M masks on every layer in *layer_map*.

    ``"dense"`` layers get ``nm_masks = {"dense": None}`` but no actual mask
    tensor.  Layers specified as ``(N, M)`` get a mask computed from their
    current weight and stored under ``nm_masks[f"{N}:{M}"]``.
    """
    name_to_module = dict(model.named_modules())

    for module_fqn, sparsity in layer_map.items():
        module = name_to_module[module_fqn]

        # Ensure the masks dict exists.
        if not hasattr(module, "nm_masks"):
            module.nm_masks = {}
        module.nm_masks.setdefault("dense", None)

        if sparsity == "dense":
            pass
        else:
            N, M = sparsity
            key = f"{N}:{M}"
            if key not in module.nm_masks:
                w = module.weight.detach()
                if isinstance(module, nn.Conv2d):
                    mask = _nm_mask_conv2d_axis(w, N, M, axis=conv_axis)
                else:
                    mask = _nm_mask_linear(w, N, M, orientation=linear_orientation)
                module.nm_masks[key] = mask.type_as(w)

        _patch_forward_for_masks(module)


# ===================================================================
# Serialization helpers
# ===================================================================

def _serialize_layer_spec(spec: Any) -> dict[str, Any]:
    mode = list(spec.mode) if isinstance(spec.mode, tuple) else spec.mode
    return {"start": spec.start, "end": spec.end, "mode": mode}


def _deserialize_layer_spec(d: dict[str, Any]) -> Any:
    from .profiles import LayerSpec
    mode = tuple(d["mode"]) if isinstance(d["mode"], list) else d["mode"]
    return LayerSpec(start=d["start"], end=d["end"], mode=mode)


def _serialize_mode_profile(mp: Any) -> dict[str, Any]:
    return {
        "name": mp.name,
        "layer_specs": [_serialize_layer_spec(ls) for ls in mp.layer_specs],
        "conv_axis": mp.conv_axis,
        "linear_orientation": mp.linear_orientation,
    }


def _deserialize_mode_profile(d: dict[str, Any]) -> Any:
    from .profiles import ModeProfile
    return ModeProfile(
        name=d["name"],
        layer_specs=[_deserialize_layer_spec(ls) for ls in d["layer_specs"]],
        conv_axis=d.get("conv_axis", "out"),
        linear_orientation=d.get("linear_orientation", "row"),
    )


def _serialize_freeze_profile(fp: Any) -> dict[str, Any]:
    return {
        "name": fp.name,
        "freeze_type": fp.freeze_type,
        "mode_profile": fp.mode_profile,
        "layer_specs": (
            [_serialize_layer_spec(ls) for ls in fp.layer_specs]
            if fp.layer_specs is not None
            else None
        ),
        "param_names": fp.param_names,
    }


def _deserialize_freeze_profile(d: dict[str, Any]) -> Any:
    from .profiles import FreezeProfile
    return FreezeProfile(
        name=d["name"],
        freeze_type=d["freeze_type"],
        mode_profile=d.get("mode_profile"),
        layer_specs=(
            [_deserialize_layer_spec(ls) for ls in d["layer_specs"]]
            if d.get("layer_specs") is not None
            else None
        ),
        param_names=d.get("param_names"),
    )


def _serialize_training_profile(tp: Any) -> dict[str, Any]:
    return {
        "name": tp.name,
        "epochs": tp.epochs,
        "label_smoothing": tp.label_smoothing,
        "cutout": tp.cutout,
        "verify_frequency": tp.verify_frequency,
        "batch_size": tp.batch_size,
        "num_workers": tp.num_workers,
    }


def _deserialize_training_profile(d: dict[str, Any]) -> Any:
    from .profiles import TrainingProfile
    return TrainingProfile(
        name=d["name"],
        epochs=d.get("epochs", 10),
        label_smoothing=d.get("label_smoothing", 0.0),
        cutout=d.get("cutout", 0),
        verify_frequency=d.get("verify_frequency", 5),
        batch_size=d.get("batch_size", 128),
        num_workers=d.get("num_workers", 4),
    )


def _serialize_output_options(oo: Any) -> dict[str, Any]:
    if oo is None:
        return {}
    return {"output_dir": oo.output_dir, "filename_prefix": oo.filename_prefix}


def _deserialize_output_options(d: dict[str, Any]) -> Any:
    if not d:
        return None
    from .profiles import OutputOptions
    return OutputOptions(
        output_dir=d.get("output_dir", "./runs"),
        filename_prefix=d.get("filename_prefix", "run"),
    )


# ===================================================================
# Info builders for logger
# ===================================================================

def _build_profiles_info(sparse: SparsityModel) -> dict[str, list[str]]:
    return {
        "mode_profiles": sorted(sparse._mode_profiles.keys()),
        "freeze_profiles": sorted(sparse._freeze_profiles.keys()),
        "training_profiles": sorted(sparse._training_profiles.keys()),
    }


def _build_model_info(sparse: SparsityModel) -> dict[str, Any]:
    total_params = sum(p.numel() for p in sparse._model.parameters())
    return {
        "model_class": type(sparse._model).__name__,
        "total_params": total_params,
    }
