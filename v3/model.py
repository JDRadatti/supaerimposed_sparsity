"""
`SparsityModel` — a wrapper around an `nn.Module` that manages multiple
sparsity, freeze, and training profiles.

Usage:

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

    # Inference in a specific mode
    with sparse.mode("mixed"):
        logits = sparse(images)

    # Persistence
    sparse.save()
    loaded = SparsityModel.load("./runs/checkpoint.pt", vgg.build_model())

"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import sparse as _sparse

from .output import OutputLogger


class SparsityModel:
    """Wraps an `nn.Module` with profile‑managed sparsity, freezing, and training.

    All profiles are validated at construction time against the inner model.
    The wrapper is transparent — attribute access falls through to the inner
    model via `__getattr__`.
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
        # ── Store inner model ──────────────────────────────────────────
        self._model: nn.Module = model

        # ── Profile storage (name → dataclass) ─────────────────────────
        self._mode_profiles: dict[str, Any] = {}
        self._freeze_profiles: dict[str, Any] = {}
        self._training_profiles: dict[str, Any] = {}

        # ── Active profile tracking (None = off) ──────────────────────
        self._active_mode: str | None = None
        self._active_freeze: str | None = None
        self._active_training: str | None = None

        # ── Derived state ─────────────────────────────────────────────
        # {mode_name: {module_fqn: "dense" | (N, M)}}
        self._mode_layer_map: dict[str, dict[str, Any]] = {}

        # Per‑mode BN buffers: {mode_name: {module_fqn: {running_mean, ...}}}
        self._bn_buffers: dict[str, dict[str, dict[str, torch.Tensor]]] = {}

        # Per‑instance freeze state (not global).
        self._frozen_params: set[nn.Parameter] = set()
        self._element_freeze_masks: dict[nn.Parameter, torch.Tensor] = {}

        # ── Output ─────────────────────────────────────────────────────
        self._output_options: Any = output_options
        self._logger: OutputLogger | None = None
        if output_options is not None:
            self._logger = OutputLogger(output_options)

        # ── Validation & mask injection ────────────────────────────────
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
        """Delegate unknown attributes to the inner `nn.Module`.

        This makes `sparse.to(device)`, `sparse.parameters()`,
        `sparse.train()`, etc. work without explicit forwarding.
        """
        # Avoid infinite recursion: Python only calls __getattr__ when
        # normal attribute lookup fails, so _model is already stored.
        inner = self.__dict__.get("_model", None)
        if inner is not None:
            return getattr(inner, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def mode(self, name: str):
        """Temporarily activate *name* as the active mode profile.

        Usage::

            with sparse.mode("mixed"):
                outputs = sparse(images)   # forward pass in "mixed" mode

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
        freeze: str | None,
        training: str,
        *,
        epochs: int | None = None,
    ) -> dict[str, Any]:
        """Train the model using the named profiles.

        Temporarily activates *mode* and *freeze*, runs the training loop
        configured by *training*, then saves BN buffers for the mode and
        leaves the profiles active.

        Training metrics are logged via the `OutputLogger` and also
        returned as a dict.

        Parameters
        ----------
        mode:       Name of a registered `ModeProfile`.
        freeze:     Name of a registered `FreezeProfile`, or `None`.
        training:   Name of a registered `TrainingProfile`.
        epochs:     Override the epoch count from the training profile (useful
                    for quick tests).

        Returns:
            Dict with keys `"final_train_loss"`, `"final_val_loss"`,
            `"final_val_acc"`, `"epochs"`, etc.

        Raises:
            ValueError: If any named profile does not exist.
        """
        # 1. Lookup profiles (raise ValueError if missing).
        if mode not in self._mode_profiles:
            raise ValueError(f"Unknown ModeProfile: {mode!r}")
        if freeze is not None and freeze not in self._freeze_profiles:
            raise ValueError(f"Unknown FreezeProfile: {freeze!r}")
        if training not in self._training_profiles:
            raise ValueError(f"Unknown TrainingProfile: {training!r}")

        train_profile = self._training_profiles[training]
        n_epochs = epochs if epochs is not None else train_profile.epochs

        # 2. Activate profiles and restore BN buffers for the mode.
        self._set_mode_profile(mode)
        self._set_freeze_profile(freeze)

        # 3. Build optimizer, scheduler, dataloaders from train_profile.
        #    optimizer = train_profile.optimizer_fn(self._model.parameters())
        #    scheduler = train_profile.scheduler_fn(optimizer) if train_profile.scheduler_fn else None
        #    train_loader, val_loader = create_dataloaders(...)

        # 4. Run the training loop.
        #    - Take pre‑training snapshot.
        #    - For each epoch: train on train_loader, evaluate on val_loader.
        #    - Log each epoch via self._logger.log_epoch(...).
        #    - Every verify_frequency epochs: verify_after_training(...).
        #    - Restore frozen positions after each optimizer step.

        # 5. Save BN buffers for the trained mode.
        #    self._save_bn_buffers()

        # 6. Return metrics dict.
        ...

    def evaluate(self) -> dict[str, tuple[float, float]]:
        """Evaluate the model in **all registered mode profiles**.

        For each mode, the corresponding BN buffers are restored so that
        evaluation uses the correct running statistics.  The previously
        active mode is restored after evaluation.

        Results are logged via the `OutputLogger`.

        Returns:
            `{mode_name: (avg_loss, accuracy)}` for every registered
            `ModeProfile` (plus `"dense"` if not registered).
        """
        # 1. Save current mode.
        # 2. For each mode profile:
        #    a. _set_mode_profile(mode_name) → restores BN.
        #    b. Run evaluation on val_loader.
        #    c. Store (loss, accuracy).
        # 3. Restore original mode.
        # 4. Log eval summary.
        # 5. Return results.
        ...

    def train_all_modes(
        self,
        freeze: str | None,
        training: str,
        *,
        mode_names: list[str] | None = None,
        epochs: int | None = None,
    ) -> dict[str, Any]:
        """Train the model at **all sparsity levels simultaneously**.

        Implements the training scheme of *Slimmable Neural Networks*
        (Yu et al., ICLR 2019): for each mini‑batch every registered mode
        profile is activated in turn, the loss is computed and gradients are
        accumulated via `loss.backward()`.  A single `optimizer.step()`
        is issued after all modes have been visited, so that weights are
        jointly optimised for every sparsity level at once.

        Per‑mode BatchNorm statistics are maintained via `_bn_buffers`,
        mimicking the paper's *Switchable Batch Normalization*.  After
        training, each mode's BN running statistics are saved.

        Parameters
        ----------
        freeze:     Name of a registered `FreezeProfile`, or `None`.
        training:   Name of a registered `TrainingProfile`.
        mode_names: Subset of mode profiles to include.  `None` = all
                    registered profiles.
        epochs:     Override the epoch count from the training profile.

        Returns:
            Dict with per‑mode final validation metrics.

        Notes
        -----
        Unlike `train()` which optimises for a single mode, this method
        accumulates gradients across *all* modes per batch.  A weight element
        only receives gradient from modes where its mask bit is 1 (i.e. the
        element is active in that mode).  This naturally regularises the
        weights toward configurations that perform well at every sparsity
        level.
        """
        # 1. Lookup profiles.
        if freeze is not None and freeze not in self._freeze_profiles:
            raise ValueError(f"Unknown FreezeProfile: {freeze!r}")
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
        self._set_freeze_profile(freeze)

        # 3. Build optimizer, scheduler, dataloaders from train_profile.
        #    optimizer = train_profile.optimizer_fn(self._model.parameters())
        #    scheduler = train_profile.scheduler_fn(optimizer) if train_profile.scheduler_fn else None
        #    train_loader, val_loader = create_dataloaders(...)

        # 4. Training loop — gradient accumulation across modes.
        #    for epoch in range(n_epochs):
        #        for batch in train_loader:
        #            optimizer.zero_grad()
        #            for name in names:
        #                # Switch mode profile + restore its BN buffers.
        #                self._restore_bn_buffers(name)
        #                self._set_active_masks(name)   # set module.active_mask_key per layer map
        #
        #                # Forward + backward (gradients accumulate).
        #                outputs = self._model(batch_inputs)
        #                loss = train_profile.loss_fn(outputs, targets)
        #                loss.backward()
        #
        #            # One step for all modes.
        #            zero_frozen_gradients(optimizer)
        #            frozen_snap = _snapshot_frozen_positions()
        #            optimizer.step()
        #            _restore_frozen_positions(frozen_snap)
        #
        #        # After each epoch:
        #        #   - Save BN buffers for every trained mode.
        #        #   - Run validation sweep: evaluate each mode with its own BN.
        #        #   - Log via self._logger.log_epoch(...).

        # 5. Save BN buffers for every trained mode.
        #    for name in names:
        #        self._restore_bn_buffers(name)
        #        self._save_bn_buffers()

        # 6. Restore previous mode.
        #    if prev_mode is not None:
        #        self._set_mode_profile(prev_mode)

        # 7. Return per‑mode metrics.
        ...

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, first_n: int = 50) -> None:
        """Write per‑mode CSV weight visualizations.

        Delegates to `visualize.visualize(sparse_model=self, first_n=first_n)`.
        """
        from .visualize import visualize as _vis

        _vis(self, first_n=first_n)

    def visualize_diff(
        self,
        other: SparsityModel,
        mode_name: str | None = None,
        first_n: int = 50,
    ) -> None:
        """Write a CSV comparing weights against *other* model.

        Delegates to `visualize.visualize_diff(model_a=self, model_b=other, ...)`.
        """
        from .visualize import visualize_diff as _vdiff

        _vdiff(self, other, mode_name=mode_name, first_n=first_n)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model state, masks, BN buffers, and profiles to *path*.

        Before saving, all active profiles are deactivated (set to `None`)
        so the checkpoint contains no transient mode‑specific state.
        Profiles are persisted as metadata so they can be re‑registered
        after loading.
        """
        # 1. Deactivate all profiles.
        #    self._set_mode_profile(None)
        #    self._set_freeze_profile(None)
        #    self._active_training = None
        # 2. Collect masks from every module.
        # 3. Collect BN buffers (already in self._bn_buffers).
        # 4. Serialize profile dataclasses (name + fields).
        # 5. torch.save(checkpoint_dict, path) with:
        #    - model_state_dict
        #    - masks: {module_name: {key: cpu_tensor}}
        #    - bn_buffers: {mode: {module: {mean, var, n}}}
        #    - mode_profiles: [{serialized fields}, ...]
        #    - freeze_profiles: [...]
        #    - training_profiles: [...] (callables won't serialize; store names + params)
        ...

    @classmethod
    def load(cls, path: str, model: nn.Module) -> SparsityModel:
        """Load a `SparsityModel` from a checkpoint created by `save()`.

        The *model* argument supplies a fresh architecture instance.
        Profiles and BN buffers are restored from the checkpoint, but
        no profile is active — the caller must call `set_mode_profile()`
        or `train()` to activate one.

        Parameters
        ----------
        path:   Path to the checkpoint file.
        model:  Fresh `nn.Module` instance with the same architecture.

        Returns:
            A new `SparsityModel` with all profiles loaded and deactivated.
        """
        # 1. torch.load the checkpoint.
        # 2. Reconstruct profile dataclasses from serialized metadata.
        # 3. Create SparsityModel with reconstructed profiles (validation runs).
        # 4. Load model_state_dict.
        # 5. Restore masks on modules.
        # 6. Restore BN buffers.
        # 7. Return the SparsityModel (nothing active).
        ...

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
                active = " (active)" if name == self._active_freeze else ""
                dep = f" → mode={fp.mode_profile}" if fp.mode_profile else ""
                lines.append(f"    {name}{active}: {fp.freeze_type}{dep}")
        else:
            lines.append("  Freeze profiles: (none)")

        # Training profiles
        if self._training_profiles:
            lines.append("  Training profiles:")
            for name, tp in self._training_profiles.items():
                active = " (active)" if name == self._active_training else ""
                lines.append(f"    {name}{active}: {tp.epochs} epochs")
        else:
            lines.append("  Training profiles: (none)")

        # Active state
        lines.append(
            f"  Active: mode={self._active_mode or 'none'}, "
            f"freeze={self._active_freeze or 'none'}, "
            f"training={self._active_training or 'none'}"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private: mode profile activation
    # ------------------------------------------------------------------

    def _set_mode_profile(self, name: str | None) -> None:
        """Activate (or deactivate) a mode profile.

        `None` restores every module to dense behaviour.

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
        """Activate (or deactivate) a freeze profile.

        `None` clears all freeze state.

        Raises:
            ValueError: If *name* is not a registered freeze profile.
        """
        if name is not None and name not in self._freeze_profiles:
            raise ValueError(
                f"Unknown FreezeProfile: {name!r}. "
                f"Available: {sorted(self._freeze_profiles.keys())}."
            )

        # Clear current freeze state.
        self._clear_freeze_state()

        if name is not None:
            self._active_freeze = name
            fp = self._freeze_profiles[name]
            # Apply fp.freeze_type to the model.
            # e.g., for "biases": freeze every bias param.
            # for "mask": use fp.mode_profile's masks to element‑freeze.
            # for "batchnorm": freeze BN weight/bias.
            # for "all": freeze every param.
            # for "non_masked": freeze weights of layers without a mask.
            # Respect fp.layer_specs and fp.param_names.
            ...
        else:
            self._active_freeze = None

    # ------------------------------------------------------------------
    # Private: BN buffer management
    # ------------------------------------------------------------------

    def _save_bn_buffers(self) -> None:
        """Save the current BN running statistics under `self._active_mode`.

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
        self._frozen_params.clear()
        self._element_freeze_masks.clear()
        # Also clean up any attribute flags on parameters.
        ...


# ===================================================================
# Internal helpers (imported from profiles module at module level
# to avoid circular imports; called inside SparsityModel.__init__).
# ===================================================================

from .profiles import (                    # noqa: E402
    _validate_mode_profile,
    _validate_freeze_profile,
    _validate_training_profile,
)


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


def _patch_forward_for_masks(module: nn.Module) -> None:
    """Patch `module.forward` so the active N:M mask is applied at runtime.

    Must be called after `module.nm_masks` has been populated.
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

    `"dense"` layers get `nm_masks = {"dense": None}` but no actual mask
    tensor.  Layers specified as `(N, M)` get a mask computed from their
    current weight and stored under `nm_masks[f"{N}:{M}"]`.

    Every layer in *layer_map* has its forward pass patched via
    `_patch_forward_for_masks`.
    """
    name_to_module = dict(model.named_modules())

    for module_fqn, sparsity in layer_map.items():
        module = name_to_module[module_fqn]

        # Ensure the masks dict exists.
        if not hasattr(module, "nm_masks"):
            module.nm_masks = {}
        module.nm_masks.setdefault("dense", None)

        if sparsity == "dense":
            # No actual mask needed — just ensure patching.
            pass
        else:
            N, M = sparsity
            key = f"{N}:{M}"
            if key not in module.nm_masks:
                w = module.weight.detach()
                if isinstance(module, nn.Conv2d):
                    mask = _sparse._nm_mask_conv2d_axis(w, N, M, axis=conv_axis)
                else:
                    mask = _sparse._nm_mask_linear(w, N, M, orientation=linear_orientation)
                module.nm_masks[key] = mask.type_as(w)

        _patch_forward_for_masks(module)


def _snapshot_frozen_positions(self) -> dict:
    """Cache current values of all frozen positions before `optimizer.step()`.

    (Used to undo weight‑decay corruption despite zero gradients.)
    """
    ...


def _restore_frozen_positions(self, snap: dict) -> None:
    """Restore frozen positions from a snapshot."""
    ...
