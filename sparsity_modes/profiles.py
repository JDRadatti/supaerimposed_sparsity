"""
Sparsity, freeze, and training profile data objects.

All profiles are defined as dataclasses and validated against the inner model
at ``SparsityModel`` construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LayerSpec
# ---------------------------------------------------------------------------

@dataclass
class LayerSpec:
    """Targets a contiguous range of Conv2d / Linear layers by index.

    Indices are 0‑based positions in the ordered list of all Conv2d/Linear
    layers in the model.  Negative indices count from the end.

    Attributes:
        start:  Inclusive index (negative = from end).
        end:    Exclusive index (negative = from end, ``None`` = to end).
        mode:   ``"dense"`` (no mask) or ``(N, M)`` for N:M sparsity.
    """
    start: int
    end: int | None
    mode: tuple[int, int] | str


# ---------------------------------------------------------------------------
# ModeProfile
# ---------------------------------------------------------------------------

@dataclass
class ModeProfile:
    """A named sparsity configuration that maps layer ranges to N:M masks.

    Example — first & last dense, middle layers 1:32::

        ModeProfile(
            name="standard_1_32",
            layer_specs=[
                LayerSpec(0, 1, "dense"),
                LayerSpec(1, -1, (1, 32)),
                LayerSpec(-1, None, "dense"),
            ],
        )

    Example — mixed granularity::

        ModeProfile(
            name="mixed",
            layer_specs=[
                LayerSpec(0, 4, "dense"),      # first 4 layers: dense
                LayerSpec(4, 8, (1, 4)),       # next 4 layers: 1:4
                LayerSpec(8, None, (1, 32)),   # remaining layers: 1:32
            ],
        )
    """
    name: str
    layer_specs: list[LayerSpec]
    conv_axis: str = "out"
    linear_orientation: str = "row"


# ---------------------------------------------------------------------------
# FreezeProfile
# ---------------------------------------------------------------------------

@dataclass
class FreezeProfile:
    """A named freeze configuration.

    ``freeze_type`` controls what is frozen:

        ``"mask"``           — freeze weight elements kept by a sparsity mask.
        ``"inverse_mask"``   — freeze weight elements pruned by a sparsity mask.
        ``"biases"``         — freeze all bias parameters.
        ``"batchnorm"``      — freeze all BatchNorm weight + bias.
        ``"non_masked"``     — freeze entire weight of layers with no mask.
        ``"all"``            — freeze every parameter.

    ``mode_profile`` is required when ``freeze_type`` is ``"mask"`` or
    ``"inverse_mask"`` — it names the ``ModeProfile`` whose masks to use.

    ``layer_specs`` restricts which layers are affected.  ``None`` = all
    eligible layers.

    ``param_names`` targets specific parameter FQNs (e.g.
    ``"features.3.bias"``), overriding ``layer_specs``.
    """
    name: str
    freeze_type: str
    mode_profile: str | None = None
    layer_specs: list[LayerSpec] | None = None
    param_names: list[str] | None = None

    # The set of legal freeze_type values.
    _VALID_FREEZE_TYPES: set[str] = field(
        default=frozenset(
            {"mask", "inverse_mask", "biases", "batchnorm", "non_masked", "all"}
        ),
        init=False,
        repr=False,
    )


# ---------------------------------------------------------------------------
# TrainingProfile
# ---------------------------------------------------------------------------

@dataclass
class TrainingProfile:
    """A named training configuration.

    ``loss_fn``, ``optimizer_fn``, and ``scheduler_fn`` accept real callables
    so that new losses / optimisers / schedulers can be used without modifying
    this module.  ``optimizer_fn`` receives the model parameters and must
    return an ``Optimizer``.  ``scheduler_fn`` receives the optimizer and must
    return a ``LRScheduler`` (or ``None`` for no scheduling).

    Example — custom training::

        TrainingProfile(
            name="adam_fine_tune",
            epochs=5,
            loss_fn=lambda out, tgt: F.cross_entropy(out, tgt, label_smoothing=0.1),
            optimizer_fn=lambda params: optim.Adam(params, lr=1e-4),
            scheduler_fn=None,
        )
    """
    name: str
    epochs: int = 10

    # Callables for full extensibility.
    loss_fn: Callable = field(default=F.cross_entropy)
    optimizer_fn: Callable = field(
        default_factory=lambda: lambda params: optim.SGD(
            params, lr=0.1, momentum=0.9, weight_decay=5e-4
        )
    )
    scheduler_fn: Callable | None = field(
        default_factory=lambda: lambda opt: lr_scheduler.StepLR(
            opt, step_size=6, gamma=0.5
        )
    )

    label_smoothing: float = 0.0
    cutout: int = 0
    verify_frequency: int = 5
    batch_size: int = 128
    num_workers: int = 4


# ---------------------------------------------------------------------------
# OutputOptions
# ---------------------------------------------------------------------------

@dataclass
class OutputOptions:
    """Controls where run artefacts are written.

    Each run creates a sub‑directory under ``output_dir/filename_prefix/``
    named ``run_<timestamp>_<commit>`` containing all logs, visualizations,
    and checkpoints.
    """
    output_dir: str = "./runs"
    filename_prefix: str = "run"


# ===================================================================
# Validation
# ===================================================================

_VALID_LAYER_TYPES = (nn.Conv2d, nn.Linear)


def _collect_layer_names(model: nn.Module) -> list[str]:
    """Return the ordered list of Conv2d / Linear module FQNs.

    Modules are listed in the order they appear during a forward pass
    (``model.named_modules()`` guarantees this for Sequential models).
    """
    names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, _VALID_LAYER_TYPES) and hasattr(module, "weight"):
            names.append(name)
    return names


def _resolve_layer_spec(
    spec: LayerSpec,
    n_layers: int,
) -> tuple[int, int]:
    """Resolve ``spec.start`` and ``spec.end`` to concrete ``[start, end)``.

    Negative indices are offset by ``n_layers``.  ``spec.end is None``
    becomes ``n_layers``.

    Raises:
        ValueError: If indices are out of range or ``start >= end``.
    """
    start = spec.start
    end = spec.end

    if start < 0:
        start += n_layers
    if end is not None and end < 0:
        end += n_layers
    if end is None:
        end = n_layers

    if not (0 <= start < n_layers):
        raise ValueError(
            f"LayerSpec start={spec.start} resolves to {start}, "
            f"must be in [0, {n_layers}) for this model."
        )
    if not (0 <= end <= n_layers):
        raise ValueError(
            f"LayerSpec end={spec.end} resolves to {end}, "
            f"must be in [0, {n_layers}] for this model."
        )
    if start >= end:
        raise ValueError(
            f"LayerSpec start={spec.start} (resolved {start}) must be "
            f"less than end={spec.end} (resolved {end})."
        )

    return start, end


def _validate_mode_profile(
    model: nn.Module,
    profile: ModeProfile,
) -> dict[str, tuple[int, int] | str]:
    """Validate a ``ModeProfile`` against *model* and return a layer→mode map.

    The returned map has form ``{module_fqn: "dense" | (N, M)}`` for every
    layer covered by the profile's ``layer_specs``.

    Raises:
        ValueError: If any spec is invalid or specs overlap.
    """
    all_names = _collect_layer_names(model)
    n_layers = len(all_names)
    assigned: dict[int, str | tuple[int, int]] = {}

    for spec in profile.layer_specs:
        istart, iend = _resolve_layer_spec(spec, n_layers)

        # Validate mode value.
        if isinstance(spec.mode, tuple):
            N, M = spec.mode
            if not (isinstance(N, int) and isinstance(M, int)):
                raise ValueError(
                    f"ModeProfile '{profile.name}': mode must be 'dense' or "
                    f"(N:int, M:int), got {spec.mode}."
                )
            if not (0 < N < M):
                raise ValueError(
                    f"ModeProfile '{profile.name}': invalid N={N}, M={M}. "
                    f"Must satisfy 0 < N < M."
                )
        elif spec.mode != "dense":
            raise ValueError(
                f"ModeProfile '{profile.name}': mode must be 'dense' or "
                f"(N, M) tuple, got {spec.mode!r}."
            )

        # Check for overlaps.
        for i in range(istart, iend):
            if i in assigned:
                raise ValueError(
                    f"ModeProfile '{profile.name}': layer {i} "
                    f"({all_names[i]}) is covered by multiple LayerSpecs."
                )
            assigned[i] = spec.mode

    if not assigned:
        raise ValueError(
            f"ModeProfile '{profile.name}': no layers were covered by the "
            f"layer_specs.  Add at least one LayerSpec."
        )

    # Build the name→mode map for covered indices.
    return {all_names[i]: assigned[i] for i in sorted(assigned)}


def _validate_freeze_profile(
    model: nn.Module,
    profile: FreezeProfile,
    mode_profile_names: set[str],
) -> None:
    """Validate a ``FreezeProfile`` against *model* and available mode profiles.

    Raises:
        ValueError: If configuration is invalid.
    """
    if profile.freeze_type not in FreezeProfile._VALID_FREEZE_TYPES:
        raise ValueError(
            f"FreezeProfile '{profile.name}': freeze_type must be one of "
            f"{sorted(FreezeProfile._VALID_FREEZE_TYPES)}, got "
            f"{profile.freeze_type!r}."
        )

    # Mask‑dependent freeze requires a mode profile.
    if profile.freeze_type in ("mask", "inverse_mask"):
        if profile.mode_profile is None:
            raise ValueError(
                f"FreezeProfile '{profile.name}': freeze_type="
                f"'{profile.freeze_type}' requires mode_profile to be set."
            )
        if profile.mode_profile not in mode_profile_names:
            raise ValueError(
                f"FreezeProfile '{profile.name}': mode_profile "
                f"'{profile.mode_profile}' does not exist. "
                f"Available: {sorted(mode_profile_names)}."
            )

    # Validate layer_specs if provided.
    if profile.layer_specs is not None:
        all_names = _collect_layer_names(model)
        n_layers = len(all_names)
        seen: set[int] = set()
        for spec in profile.layer_specs:
            istart, iend = _resolve_layer_spec(spec, n_layers)
            for i in range(istart, iend):
                if i in seen:
                    raise ValueError(
                        f"FreezeProfile '{profile.name}': overlapping "
                        f"LayerSpec at layer {i} ({all_names[i]})."
                    )
                seen.add(i)

    # Validate param_names if provided.
    if profile.param_names is not None:
        all_param_names = set(dict(model.named_parameters()).keys())
        for pname in profile.param_names:
            if pname not in all_param_names:
                raise ValueError(
                    f"FreezeProfile '{profile.name}': param '{pname}' "
                    f"not found in model."
                )


def _validate_training_profile(
    profile: TrainingProfile,
) -> None:
    """Validate a ``TrainingProfile``.

    Raises:
        ValueError: If configuration is invalid.
    """
    if profile.epochs <= 0:
        raise ValueError(
            f"TrainingProfile '{profile.name}': epochs must be > 0, "
            f"got {profile.epochs}."
        )
    if not callable(profile.loss_fn):
        raise ValueError(
            f"TrainingProfile '{profile.name}': loss_fn must be callable."
        )
    if not callable(profile.optimizer_fn):
        raise ValueError(
            f"TrainingProfile '{profile.name}': optimizer_fn must be callable."
        )
    if profile.scheduler_fn is not None and not callable(profile.scheduler_fn):
        raise ValueError(
            f"TrainingProfile '{profile.name}': scheduler_fn must be "
            f"callable or None."
        )
