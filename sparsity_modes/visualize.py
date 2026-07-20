"""
Weight visualization and diffing for sparsity models.

Outputs CSVs that show the first *N* elements of each layer's weight,
ordered by mask grouping so the sparsity pattern is visually obvious
(e.g. for 1:4 every 4 elements shows exactly one kept position).

Each mode gets its own CSV file.  Each layer header includes summary
stats: mode, sparsity, number masked, total elements, and more.

Weights are reshaped to respect the mask axis so that the
``pos_in_group`` column makes visual sense.
"""

from __future__ import annotations

import csv
from typing import Any

import torch
import torch.nn as nn

from .masks import set_active_sparsity
from .freeze import _frozen_params, _element_freeze_masks


def visualize(
    model: nn.Module,
    logger: Any,
    sparsity_keys: list[str] | None = None,
    first_n: int = 50,
) -> None:
    """Write one CSV per registered sparsity key under the run directory.

    For each key the model is temporarily activated so that the correct
    masks are visible.  Each layer's header block includes:

    * sparsity key name
    * sparsity for this layer (``dense``, ``1:4``, ``1:32``, ...)
    * total elements, kept-by-mask, pruned, frozen counts
    * percentage masked / frozen

    Parameters
    ----------
    model:         The ``nn.Module`` whose weights to visualize.
    logger:        An ``OutputLogger`` with ``start_run()`` already called.
    sparsity_keys: Which keys to visualize.  If ``None``, all registered
                   keys are used.
    first_n:       How many elements to print per layer (default 50).
    """
    # Discover sparsity keys if not given
    if sparsity_keys is None:
        keys: set[str] = set()
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "nm_masks"):
                keys.update(module.nm_masks.keys())
        sparsity_keys = sorted(keys)

    # Build layer name → module mapping
    name_to_module = dict(model.named_modules())

    # Collect layer assignments for the header
    layer_map: dict[str, str] = {}
    for mod_name, module in name_to_module.items():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if not hasattr(module, "nm_masks") or not module.nm_masks:
            continue
        layer_map[mod_name] = _module_sparsity_summary(module)

    for key in sparsity_keys:
        layer_assignments: dict[str, str] = {}
        for mod_name in layer_map:
            module = name_to_module[mod_name]
            if key in module.nm_masks:
                layer_assignments[mod_name] = key
            else:
                layer_assignments[mod_name] = "dense"

        freeze_name = _current_freeze_description(model)
        header_lines = _build_visualize_header(
            mode_name=key,
            layer_assignments=layer_assignments,
            freeze=freeze_name,
            commit=logger.commit,
            branch=logger.branch,
        )

        csv_path, csv_file = logger.open_visualize_csv(key, header_lines)

        try:
            set_active_sparsity(model, key)
            for mod_name, sparsity_str in layer_assignments.items():
                module = name_to_module[mod_name]
                weight = module.weight.detach()

                mask = _get_mask(module, key)
                freeze_mask = _get_freeze_mask(module.weight)
                axis_info = _axis_info(module)

                _write_layer_visualization(
                    csv_file=csv_file,
                    weight=weight,
                    mask=mask,
                    freeze_mask=freeze_mask,
                    layer_name=mod_name,
                    sparsity=sparsity_str,
                    mode_name=key,
                    freeze_name=freeze_name,
                    axis_info=axis_info,
                    first_n=first_n,
                )
        finally:
            csv_file.close()


def visualize_diff(
    snapshot_before: dict,
    snapshot_after: dict,
    logger: Any,
    sparsity_key: str,
    model: nn.Module | None = None,
    first_n: int = 50,
) -> None:
    """Write a CSV comparing two parameter snapshots element-by-element.

    Parameters
    ----------
    snapshot_before: Dict from ``parameter_snapshot()`` taken before training.
    snapshot_after:  Dict from ``parameter_snapshot()`` taken after training.
    logger:          An ``OutputLogger`` with ``start_run()`` already called.
    sparsity_key:    Which sparsity mode to use for masks (e.g. ``"1:4"``).
    model:           Live model used for axis info (``_conv_axis``, etc.).
    first_n:         How many elements to print per layer (default 50).
    """
    params_before = snapshot_before["params"]
    params_after = snapshot_after["params"]
    masks = snapshot_before.get("masks", {})
    frozen_names = set(snapshot_before.get("frozen_param_names", []))
    elem_freeze = snapshot_before.get("element_freeze_masks", {})

    # Build module name → axis info from the live model
    mod_axis: dict[str, tuple[str, str] | None] = {}
    if model is not None:
        for mod_name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                mod_axis[mod_name] = _axis_info(module)

    # Build layer assignments: module_name → sparsity string
    layer_assignments: dict[str, str] = {}
    # Group weight params by their module (strip ".weight" suffix)
    weight_params: dict[str, str] = {}  # param_name → module_name
    for pname in params_before:
        if not pname.endswith(".weight"):
            continue
        mod_name = pname.rsplit(".weight", 1)[0]
        if mod_name in masks and sparsity_key in masks[mod_name]:
            layer_assignments[mod_name] = sparsity_key
            weight_params[pname] = mod_name
        else:
            layer_assignments[mod_name] = "dense"
            weight_params[pname] = mod_name

    if not layer_assignments:
        return

    header_lines = _build_diff_header(
        mode_name=sparsity_key,
        layer_assignments=layer_assignments,
        commit=logger.commit,
        branch=logger.branch,
    )

    csv_path, csv_file = logger.open_diff_csv(sparsity_key, header_lines)

    try:
        for pname, mod_name in weight_params.items():
            if pname not in params_before or pname not in params_after:
                continue

            weight_a = params_before[pname]
            weight_b = params_after[pname]
            diff = weight_a - weight_b

            # Mask
            mask = None
            if mod_name in masks and sparsity_key in masks[mod_name]:
                mask = masks[mod_name][sparsity_key]

            # Freeze mask
            freeze_mask = _freeze_mask_for_param(
                pname, weight_a, frozen_names, elem_freeze
            )

            # Axis info
            axis_info = mod_axis.get(mod_name)
            sparsity_str = layer_assignments[mod_name]

            _write_diff_layer(
                csv_file=csv_file,
                weight_a=weight_a,
                weight_b=weight_b,
                diff=diff,
                mask=mask,
                freeze_mask=freeze_mask,
                layer_name=mod_name,
                sparsity=sparsity_str,
                mode_name=sparsity_key,
                axis_info=axis_info,
                first_n=first_n,
            )
    finally:
        csv_file.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _module_sparsity_summary(module: nn.Module) -> str:
    """Return a compact sparsity string for a module's first registered key."""
    if not hasattr(module, "nm_masks"):
        return "dense"
    keys = [k for k in module.nm_masks if k != "dense"]
    if keys:
        return keys[0]
    return "dense"


def _current_freeze_description(model: nn.Module) -> str:
    """Describe the current freeze state in a short string."""
    n_frozen_params = len(_frozen_params)
    n_frozen_elements = sum(int(m.sum().item()) for m in _element_freeze_masks.values())
    if n_frozen_params == 0 and n_frozen_elements == 0:
        return "none"
    parts = []
    if n_frozen_params:
        parts.append(f"{n_frozen_params} frozen params")
    if n_frozen_elements:
        parts.append(f"{n_frozen_elements:,} frozen elements")
    return ", ".join(parts)


def _freeze_mask_for_param(
    pname: str,
    weight: torch.Tensor,
    frozen_names: set[str],
    elem_freeze: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    """Build a freeze mask for a parameter from snapshot data."""
    if pname in frozen_names:
        return torch.ones_like(weight, dtype=torch.bool)
    if pname in elem_freeze:
        return elem_freeze[pname]
    return None


def _get_mask(module: nn.Module, key: str) -> torch.Tensor | None:
    """Return the mask tensor for *key* on *module*, or ``None``."""
    if hasattr(module, "nm_masks") and key in module.nm_masks:
        return module.nm_masks[key]
    return None


def _get_freeze_mask(param: torch.nn.Parameter) -> torch.Tensor | None:
    """Return the boolean freeze mask for *param*, or ``None``.

    Fully-frozen params return an all-True mask.
    """
    if param in _frozen_params:
        return torch.ones_like(param, dtype=torch.bool)
    return _element_freeze_masks.get(param, None)


def _axis_info(module: nn.Module) -> tuple[str, str] | None:
    """Return ``(layer_type, axis)`` for group-aware weight reshaping."""
    if isinstance(module, nn.Conv2d):
        axis = getattr(module, "_conv_axis", "out")
        return ("conv", axis)
    if isinstance(module, nn.Linear):
        orient = getattr(module, "_linear_orientation", "row")
        return ("linear", orient)
    return None


def _group_size_from_sparsity(sparsity: str) -> int | None:
    """Parse ``"1:4"`` → ``4``, ``"1:32"`` → ``32``, ``"dense"`` → ``None``."""
    if sparsity == "dense":
        return None
    parts = sparsity.split(":")
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _reshape_for_axis(
    weight: torch.Tensor,
    mask: torch.Tensor | None,
    freeze_mask: torch.Tensor | None,
    axis_info: tuple[str, str],
    M: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Reshape weight, mask, and freeze_mask so the last dim is the
    mask-axis, split into ``shape = [-1, M]`` where each row is one N:M group.
    """
    layer_type, axis = axis_info

    if weight.ndim == 4:
        a, b, c, d = weight.shape
    else:
        a, b = weight.shape
        c, d = None, None

    if layer_type == "conv" and axis == "out":
        w = weight.permute(1, 2, 3, 0).reshape(-1, a)
        m = mask.permute(1, 2, 3, 0).reshape(-1, a) if mask is not None else None
        fm = freeze_mask.permute(1, 2, 3, 0).reshape(-1, a) if freeze_mask is not None else None
    elif layer_type == "conv" and axis == "in":
        w = weight.permute(0, 2, 3, 1).reshape(-1, b)
        m = mask.permute(0, 2, 3, 1).reshape(-1, b) if mask is not None else None
        fm = freeze_mask.permute(0, 2, 3, 1).reshape(-1, b) if freeze_mask is not None else None
    elif layer_type == "linear" and axis == "row":
        w = weight.reshape(-1, b)
        m = mask.reshape(-1, b) if mask is not None else None
        fm = freeze_mask.reshape(-1, b) if freeze_mask is not None else None
    elif layer_type == "linear" and axis == "col":
        w = weight.T.reshape(-1, a)
        m = mask.T.reshape(-1, a) if mask is not None else None
        fm = freeze_mask.T.reshape(-1, a) if freeze_mask is not None else None
    else:
        w = weight.flatten().unsqueeze(1)
        m = mask.flatten().unsqueeze(1) if mask is not None else None
        fm = freeze_mask.flatten().unsqueeze(1) if freeze_mask is not None else None
        return w, m, fm

    axis_size = w.shape[1]
    if axis_size < M:
        w = w.flatten().unsqueeze(1)
        m = m.flatten().unsqueeze(1) if m is not None else None
        fm = fm.flatten().unsqueeze(1) if fm is not None else None
        return w, m, fm

    w = w.reshape(-1, axis_size // M, M).reshape(-1, M)
    if m is not None:
        m = m.reshape(-1, axis_size // M, M).reshape(-1, M)
    if fm is not None:
        fm = fm.reshape(-1, axis_size // M, M).reshape(-1, M)

    return w, m, fm


# ---------------------------------------------------------------------------
# Header builders
# ---------------------------------------------------------------------------

def _build_visualize_header(
    mode_name: str,
    layer_assignments: dict[str, str],
    freeze: str,
    commit: str,
    branch: str,
) -> list[str]:
    header: list[str] = []
    header.append(f"Model weights -- mode: {mode_name!r}  freeze: {freeze}")
    header.append(f"Commit: {commit}  Branch: {branch}")
    header.append("")
    parts = [f"{n}={s}" for n, s in layer_assignments.items()]
    header.append(f"Layer sparsity: {', '.join(parts)}")
    header.append("")
    header.append("Legend: M=kept by mask, F=frozen, .=trainable")
    return header


def _build_diff_header(
    mode_name: str,
    layer_assignments: dict[str, str],
    commit: str,
    branch: str,
) -> list[str]:
    header: list[str] = []
    header.append(f"Weight diff -- mode: {mode_name!r}")
    header.append(f"Commit: {commit}  Branch: {branch}")
    header.append("")
    parts = [f"{n}={s}" for n, s in layer_assignments.items()]
    header.append(f"Layer sparsity: {', '.join(parts)}")
    header.append("")
    header.append("Legend: M=kept by mask, F=frozen, .=trainable")
    return header


# ---------------------------------------------------------------------------
# Layer writers
# ---------------------------------------------------------------------------

def _write_layer_visualization(
    csv_file: Any,
    weight: torch.Tensor,
    mask: torch.Tensor | None,
    freeze_mask: torch.Tensor | None,
    layer_name: str,
    sparsity: str,
    mode_name: str,
    freeze_name: str,
    axis_info: tuple[str, str] | None,
    first_n: int,
) -> None:
    """Write stats and data rows for a single layer."""
    shape = tuple(weight.shape)
    total = weight.numel()
    M = _group_size_from_sparsity(sparsity)

    n_masked = int(mask.sum().item()) if mask is not None else 0
    n_frozen = int(freeze_mask.sum().item()) if freeze_mask is not None else 0
    pct_masked = (100.0 * n_masked / total) if total > 0 else 0.0
    pct_frozen = (100.0 * n_frozen / total) if total > 0 else 0.0

    f = csv_file
    f.write("\n")
    f.write(f"# Layer: {layer_name}  shape={shape}  sparsity={sparsity}\n")
    f.write(f"# mode={mode_name}  freeze={freeze_name}\n")
    f.write(
        f"# total_elements={total}  kept_by_mask={n_masked}"
        f"  pruned={total - n_masked}  frozen={n_frozen}"
        f"  masked_pct={pct_masked:.2f}%  frozen_pct={pct_frozen:.2f}%\n"
    )

    # Reshape for axis-aware group ordering
    if M is not None and mask is not None and axis_info is not None:
        w_grouped, m_grouped, f_grouped = _reshape_for_axis(
            weight, mask, freeze_mask, axis_info, M
        )
    else:
        w_grouped = weight.flatten().unsqueeze(1)
        m_grouped = None
        f_grouped = freeze_mask.flatten().unsqueeze(1) if freeze_mask is not None else None
        M = 1

    n_groups = w_grouped.shape[0]
    M = min(M, w_grouped.shape[1])
    n_shown = min(first_n, n_groups * M)
    f.write(f"# shown={n_shown}\n")
    f.write("\n")

    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["index", "pos_in_group", "value", "masked", "frozen"])

    row_count = 0
    for group in range(n_groups):
        for pos in range(M):
            if row_count >= n_shown:
                break
            val = w_grouped[group, pos].item()
            mval = "M" if (m_grouped is not None and m_grouped[group, pos].item() == 1) else "."
            fval = "."
            if f_grouped is not None and f_grouped[group, pos].item():
                fval = "F"
            writer.writerow(
                [row_count, f"{pos}/{M}", f"{val:+.4f}", mval, fval]
            )
            row_count += 1
        if row_count >= n_shown:
            break


def _write_diff_layer(
    csv_file: Any,
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    diff: torch.Tensor,
    mask: torch.Tensor | None,
    freeze_mask: torch.Tensor | None,
    layer_name: str,
    sparsity: str,
    mode_name: str,
    axis_info: tuple[str, str] | None,
    first_n: int,
) -> None:
    """Write diff stats and data rows for a single layer."""
    shape = tuple(weight_a.shape)
    total = weight_a.numel()
    M = _group_size_from_sparsity(sparsity)

    diff_flat = diff.flatten()
    n_different = int((diff_flat.abs() > 1e-8).sum().item())
    n_identical = total - n_different
    max_abs_diff_val = diff_flat.abs().max().item()
    n_masked = int(mask.sum().item()) if mask is not None else 0
    n_frozen = int(freeze_mask.sum().item()) if freeze_mask is not None else 0

    f = csv_file
    f.write("\n")
    f.write(f"# Layer: {layer_name}  shape={shape}  sparsity={sparsity}\n")
    f.write(f"# mode={mode_name}\n")
    f.write(
        f"# total_elements={total}  identical={n_identical}  different={n_different}"
        f"  max_abs_diff={max_abs_diff_val:.6e}\n"
    )
    f.write(f"# kept_by_mask={n_masked}  frozen={n_frozen}\n")
    f.write("\n")

    # Reshape for axis-aware group ordering
    if M is not None and mask is not None and axis_info is not None:
        w_a, m_grouped, f_grouped = _reshape_for_axis(weight_a, mask, freeze_mask, axis_info, M)
        w_b, _, _ = _reshape_for_axis(weight_b, mask, None, axis_info, M)
        d_grouped, _, _ = _reshape_for_axis(diff, mask, None, axis_info, M)
    else:
        w_a = weight_a.flatten().unsqueeze(1)
        w_b = weight_b.flatten().unsqueeze(1)
        d_grouped = diff.flatten().unsqueeze(1)
        m_grouped = None
        f_grouped = freeze_mask.flatten().unsqueeze(1) if freeze_mask is not None else None
        M = 1

    n_groups = w_a.shape[0]
    M = min(M, w_a.shape[1])
    n_shown = min(first_n, n_groups * M)

    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["index", "pos_in_group", "before", "after", "diff", "masked", "frozen"])

    row_count = 0
    for group in range(n_groups):
        for pos in range(M):
            if row_count >= n_shown:
                break
            val_a = w_a[group, pos].item()
            val_b = w_b[group, pos].item()
            val_d = d_grouped[group, pos].item()
            mval = "M" if (m_grouped is not None and m_grouped[group, pos].item() == 1) else "."
            fval = "."
            if f_grouped is not None and f_grouped[group, pos].item():
                fval = "F"
            writer.writerow(
                [row_count, f"{pos}/{M}",
                 f"{val_a:+.6f}", f"{val_b:+.6f}", f"{val_d:+.6e}",
                 mval, fval]
            )
            row_count += 1
        if row_count >= n_shown:
            break
