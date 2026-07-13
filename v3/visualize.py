"""
Weight visualization and diffing for sparsity models.

Outputs CSVs that show the first *N* elements of each layer's weight,
ordered by mask grouping so the sparsity pattern is visually obvious
(e.g., for 1:4 every 4 elements shows exactly one kept position).

Each mode gets its own CSV file.  Each layer header includes summary
stats: mode, freeze profile, number masked, total elements, and more.

Weights are reshaped to respect the mask axis (e.g. ``axis='out'`` groups
output channels) so that the ``pos_in_group`` column makes visual sense.
"""

from __future__ import annotations

import csv
from typing import Any

import torch
import torch.nn as nn


def visualize(
    sparsity_model: Any,
    first_n: int = 50,
) -> None:
    """Write one CSV per registered mode profile under the run directory.

    For each mode the model is temporarily activated so that the correct
    masks are visible.  Each layer's header block includes:

    * mode profile name
    * freeze profile name (if active, or ``none``)
    * sparsity for this layer (``dense``, ``1:4``, ``1:32``, ...)
    * total elements, kept-by-mask, pruned, frozen counts
    * percentage masked / frozen

    Parameters
    ----------
    sparsity_model: The ``SparsityModel`` whose weights to visualize.
    first_n:        How many elements to print per layer (default 50).
    """
    logger: Any = sparsity_model._logger
    if logger is None:
        from .output import OutputLogger
        from .profiles import OutputOptions
        logger = OutputLogger(OutputOptions())
        sparsity_model._logger = logger

    if logger.run_dir is None:
        logger.start_run(
            "visualize",
            profiles_info={
                "mode_profiles": sorted(sparsity_model._mode_profiles.keys()),
                "freeze_profiles": sorted(sparsity_model._freeze_profiles.keys()),
                "training_profiles": sorted(sparsity_model._training_profiles.keys()),
            },
            model_info={"model_class": type(sparsity_model._model).__name__},
        )

    mode_names = sorted(sparsity_model._mode_profiles.keys())
    name_to_module = dict(sparsity_model._model.named_modules())

    for mode_name in mode_names:
        layer_map = sparsity_model._mode_layer_map[mode_name]
        profile = sparsity_model._mode_profiles[mode_name]
        freeze_name = sparsity_model._active_freeze or "none"

        # Build compact layer→sparsity summary for the file header.
        layer_assignments: dict[str, str] = {}
        for fqn, s in layer_map.items():
            layer_assignments[fqn] = s if isinstance(s, str) else f"{s[0]}:{s[1]}"

        header_lines = _build_file_header(
            mode_name=mode_name,
            layer_assignments=layer_assignments,
            freeze=freeze_name,
            commit=logger.commit,
            branch=logger.branch,
        )

        csv_path, csv_file = logger.open_visualize_csv(mode_name, header_lines)

        try:
            with sparsity_model.mode(mode_name):
                for module_fqn, sparsity in layer_map.items():
                    module = name_to_module[module_fqn]
                    weight = module.weight.detach()
                    sparsity_str = (
                        sparsity if isinstance(sparsity, str)
                        else f"{sparsity[0]}:{sparsity[1]}"
                    )

                    mask = _get_mask_for_layer(module, sparsity)
                    freeze_mask = _get_freeze_mask_for_param(
                        sparsity_model, module.weight
                    )

                    # Determine axis info for correct group ordering.
                    axis_info = _axis_info_for_module(module, profile)

                    _write_layer_visualization(
                        csv_file=csv_file,
                        weight=weight,
                        mask=mask,
                        freeze_mask=freeze_mask,
                        layer_name=module_fqn,
                        sparsity=sparsity_str,
                        mode_name=mode_name,
                        freeze_name=freeze_name,
                        axis_info=axis_info,
                        first_n=first_n,
                    )
        finally:
            csv_file.close()

    logger.finish_run()


def visualize_diff(
    model_a: Any,
    model_b: Any,
    mode_name: str | None = None,
    first_n: int = 50,
) -> None:
    """Write a CSV comparing two models' weights element‑by‑element.

    Parameters
    ----------
    model_a:    The reference model (or first ``SparsityModel``).
    model_b:    The comparison model (or second ``SparsityModel``).
    mode_name:  If given, only compare using this mode profile's masks.
                Otherwise compares raw weights (all with ``"."`` mask column).
    first_n:    How many elements to print per layer (default 50).
    """
    ...


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_mask_for_layer(
    module: nn.Module,
    sparsity: tuple[int, int] | str,
) -> torch.Tensor | None:
    """Return the mask tensor for *sparsity* on *module*, or ``None`` for dense."""
    if sparsity == "dense":
        return None
    key = f"{sparsity[0]}:{sparsity[1]}"
    if hasattr(module, "nm_masks") and key in module.nm_masks:
        return module.nm_masks[key]
    return None


def _get_freeze_mask_for_param(
    sparsity_model: Any,
    param: torch.nn.Parameter,
) -> torch.Tensor | None:
    """Return the boolean element‑freeze mask for *param*, or ``None``."""
    return sparsity_model._element_freeze_masks.get(param, None)


def _axis_info_for_module(
    module: nn.Module,
    profile: Any,
) -> tuple[str, str] | None:
    """Return ``(layer_type, axis)`` for group‑aware weight reshaping.

    Returns:
        ``(layer_type, axis)`` — e.g. ``("conv", "out")``, ``("linear", "row")``.
        ``None`` if the module type is unrecognised.
    """
    if isinstance(module, nn.Conv2d):
        return ("conv", profile.conv_axis)
    if isinstance(module, nn.Linear):
        return ("linear", profile.linear_orientation)
    return None


def _build_file_header(
    mode_name: str,
    layer_assignments: dict[str, str],
    freeze: str,
    commit: str,
    branch: str,
) -> list[str]:
    """Build the metadata header lines for a visualization CSV file."""
    header: list[str] = []
    header.append(f"Model weights -- mode: {mode_name!r}  freeze: {freeze}")
    header.append(f"Commit: {commit}  Branch: {branch}")
    header.append("")

    parts = [f"{n}={s}" for n, s in layer_assignments.items()]
    header.append(f"Layer sparsity: {', '.join(parts)}")
    header.append("")
    header.append("Legend: M=kept by mask, F=frozen, .=trainable")
    return header


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
    """Write stats (raw text) and data rows (CSV) for a single layer.

    Stats lines are written directly with ``csv_file.write()`` so they
    are never double‑quoted.  Data rows use ``csv.writer`` for proper
    CSV formatting.

    Parameters
    ----------
    csv_file:    Raw file handle open for writing.
    weight:      2‑D or 4‑D weight tensor.
    mask:        Mask tensor of same shape as *weight*, or ``None`` for dense.
    freeze_mask: Boolean tensor of same shape as *weight*, or ``None``.
    layer_name:  Module FQN.
    sparsity:    String like ``"dense"``, ``"1:4"``, ``"1:32"``.
    mode_name:   Active mode profile name.
    freeze_name: Active freeze profile name (or ``"none"``).
    axis_info:   ``("conv", "out")``, ``("linear", "row")``, or ``None`` for dense.
    first_n:     Max elements to print for this layer.
    """
    shape = tuple(weight.shape)
    total = weight.numel()
    n_shown = min(first_n, total)
    M = _group_size_from_sparsity(sparsity)

    # ── Compute stats ────────────────────────────────────────────────
    n_masked = int(mask.sum().item()) if mask is not None else 0
    n_frozen = int(freeze_mask.sum().item()) if freeze_mask is not None else 0
    pct_masked = (100.0 * n_masked / total) if total > 0 else 0.0
    pct_frozen = (100.0 * n_frozen / total) if total > 0 else 0.0

    # ── Stats block (raw write — no csv.writer quoting) ──────────────
    f = csv_file  # shorthand
    f.write("\n")
    f.write(f"# Layer: {layer_name}  shape={shape}  sparsity={sparsity}\n")
    f.write(f"# mode={mode_name}  freeze={freeze_name}\n")
    f.write(
        f"# total_elements={total}  kept_by_mask={n_masked}"
        f"  pruned={total - n_masked}  frozen={n_frozen}"
        f"  masked_pct={pct_masked:.2f}%  frozen_pct={pct_frozen:.2f}%\n"
    )
    f.write(f"# shown={n_shown}\n")
    f.write("\n")

    # ── Column header (CSV row) ──────────────────────────────────────
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["index", "pos_in_group", "value", "masked", "frozen"])

    # ── Reshape for axis‑aware group ordering ────────────────────────
    if M is not None and mask is not None and axis_info is not None:
        w_grouped, m_grouped, f_grouped = _reshape_for_axis(
            weight, mask, freeze_mask, axis_info, M
        )
    else:
        # Dense / no mask — flat iteration.
        w_grouped = weight.flatten().unsqueeze(1)  # [total, 1]
        m_grouped = None
        f_grouped = freeze_mask.flatten().unsqueeze(1) if freeze_mask is not None else None
        M = 1

    n_groups = w_grouped.shape[0]
    n_shown = min(first_n, n_groups * M)

    # ── Data rows (row‑major: groups outer, positions inner) ─────────
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


def _reshape_for_axis(
    weight: torch.Tensor,
    mask: torch.Tensor,
    freeze_mask: torch.Tensor | None,
    axis_info: tuple[str, str],
    M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Reshape weight, mask, and freeze_mask so that the last dim is the
    mask‑axis, split into ``shape = [-1, M]`` where each row is one N:M group.

    Row‑major iteration over the result yields contiguous groups visible as
    adjacent CSV rows.

    Parameters
    ----------
    weight:      Original weight tensor [out, in, kH, kW] or [out_f, in_f].
    mask:        Mask of same shape as *weight*.
    freeze_mask: Boolean tensor of same shape or ``None``.
    axis_info:   ``("conv", "out")``, ``("conv", "in")``, ``("linear", "row")``,
                 or ``("linear", "col")``.
    M:           Group size.

    Returns:
        ``(w, m, fm)`` each of shape ``[-1, M]``.  *fm* may be ``None``.
    """
    layer_type, axis = axis_info
    a, b, c, d = weight.shape if weight.ndim == 4 else (weight.shape[0], weight.shape[1], None, None)

    if layer_type == "conv" and axis == "out":
        # Mask groups consecutive output channels.
        w = weight.permute(1, 2, 3, 0).reshape(-1, a)   # [*in_spatial, out_c]
        m = mask.permute(1, 2, 3, 0).reshape(-1, a)
        fm = freeze_mask.permute(1, 2, 3, 0).reshape(-1, a) if freeze_mask is not None else None
    elif layer_type == "conv" and axis == "in":
        # Mask groups consecutive input channels.
        w = weight.permute(0, 2, 3, 1).reshape(-1, b)   # [*out_spatial, in_c]
        m = mask.permute(0, 2, 3, 1).reshape(-1, b)
        fm = freeze_mask.permute(0, 2, 3, 1).reshape(-1, b) if freeze_mask is not None else None
    elif layer_type == "linear" and axis == "row":
        # Mask groups consecutive input features per output row.
        # weight: [out_f, in_f] — in_f is already the last dim.
        w = weight.reshape(-1, b)   # b = in_f
        m = mask.reshape(-1, b)
        fm = freeze_mask.reshape(-1, b) if freeze_mask is not None else None
    elif layer_type == "linear" and axis == "col":
        # Mask groups consecutive output features per input column.
        # weight: [out_f, in_f] → move out_f to last dim.
        w = weight.T.reshape(-1, a)  # a = out_f
        m = mask.T.reshape(-1, a)
        fm = freeze_mask.T.reshape(-1, a) if freeze_mask is not None else None
    else:
        # Fallback: plain flatten.
        w = weight.flatten().unsqueeze(1)
        m = mask.flatten().unsqueeze(1)
        fm = freeze_mask.flatten().unsqueeze(1) if freeze_mask is not None else None
        return w, m, fm

    # Split the last dim into groups of M.
    axis_size = w.shape[1]
    w = w.reshape(-1, axis_size // M, M).reshape(-1, M)
    m = m.reshape(-1, axis_size // M, M).reshape(-1, M)
    if fm is not None:
        fm = fm.reshape(-1, axis_size // M, M).reshape(-1, M)

    return w, m, fm


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
