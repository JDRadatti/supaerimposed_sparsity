"""
Runtime validation utilities.
Snapshots parameters and verifies frozen positions remain unchanged.
"""
import torch
import torch.nn as nn

from .freeze import is_frozen, _element_freeze_masks, frozen_parameter_names
from .logger import debug


def snapshot_parameters(model: nn.Module):
    """
    Returns a dict:
        {
            "params": {name: tensor.clone()},
            "masks":  {module_name: mask_tensor.clone()},
        }
    """
    snap = {"params": {}, "masks": {}}
    for name, param in model.named_parameters():
        snap["params"][name] = param.detach().clone()
    for mod_name, module in model.named_modules():
        if not hasattr(module, "nm_masks"):
            continue
        mask_dict = {}
        for key, mask in module.nm_masks.items():
            if mask is None:
                continue
            mask_dict[key] = mask.detach().clone()
        if mask_dict:
            snap["masks"][mod_name] = mask_dict
    return snap


def parameter_snapshot(model: nn.Module) -> dict:
    """Comprehensive snapshot of params, masks, and freeze state for comparison.

    Returns:
        ``{"params": {...}, "masks": {...},
           "frozen_param_names": [...], "element_freeze_masks": {...}}``
    """
    snap = snapshot_parameters(model)
    snap["frozen_param_names"] = frozen_parameter_names(model)
    name_to_param = dict(model.named_parameters())
    snap["element_freeze_masks"] = {}
    for name, p in name_to_param.items():
        if p in _element_freeze_masks:
            snap["element_freeze_masks"][name] = _element_freeze_masks[p].clone()
    return snap


def verify_after_training(
    model: nn.Module,
    snapshot: dict,
    active_mask_key: str | None = None,
    frozen_params: list[torch.nn.Parameter] | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
):
    """
    Checks that:
      1) All **masked** entries behave like zero and have NOT changed
         relative to the snapshot taken before training.
      2) All **frozen** parameters are exactly the same as the snapshot.

    Parameters
    ----------
    model : nn.Module
        The model after training (or after the checkpoint you want to test).
    snapshot : dict
        The object returned by ``snapshot_parameters`` taken BEFORE training.
    active_mask_key : str 
        Mask key to verify (e.g. "1:4").
    frozen_params : list[torch.nn.Parameter]
        List of parameters that should be frozen
    atol / rtol : tolerance for ``torch.allclose``.
    """
    #
    # Verify frozen parameters
    #
    for name, p in model.named_parameters():
        if is_frozen(p):
            before = snapshot["params"][name]
            if not torch.allclose(p, before, atol=atol, rtol=rtol):
                raise RuntimeError(
                    f"[VERIFY][FREEZE] Parameter '{name}' changed! before : {before}, after  : {p}"
                )
    #
    # Verify element-wise frozen positions
    #
    if not _element_freeze_masks:
        debug("[VERIFY] No element-freeze masks registered, skipping element check.")
    else:
        name_to_param = dict(model.named_parameters())
        for name, p in name_to_param.items():
            if p not in _element_freeze_masks:
                continue
            freeze_mask = _element_freeze_masks[p]
            if not freeze_mask.any():
                continue

            before = snapshot["params"][name].to(p.device)
            frozen_pos = freeze_mask
            if not torch.allclose(p.data[frozen_pos], before[frozen_pos], atol=atol, rtol=rtol):
                raise RuntimeError(
                    f"[VERIFY][ELEMENT] Element-frozen positions changed in '{name}'.\n"
                    f"   #frozen entries: {frozen_pos.sum().item()}\n"
                    f"   before[:5] : {before[frozen_pos].view(-1)[:5]}\n"
                    f"   after[:5]  : {p.data[frozen_pos].view(-1)[:5]}"
                )
    debug("[VERIFY] All checks passed (frozen + element-freeze).")
    return True


def compare_snapshots(
    model: nn.Module,
    before: dict,
    after: dict,
    sparsity_key: str | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    line_cap: int | None = None,
) -> None:
    """Print a per-element before/after comparison with mask and freeze info.

    Each element is printed on its own line:

        [idx] (before, after, diff, mask, protect)

    where *mask* is 1/0 for weight params (blank for biases) and *protect*
    is ``F`` (frozen) or ``.`` (trainable).

    Args:
        line_cap: If set, only print the first *line_cap* elements per tensor,
                  then append a summary line with the remaining count.
    """
    name_to_param = dict(model.named_parameters())

    debug(f"\n{'=' * 70}")
    debug("PARAMETER COMPARISON")
    debug(f"{'=' * 70}")
    debug("  [idx] (before, after, diff, mask, protect)")
    debug("  mask:    1 = kept by mask, 0 = pruned, '' = not masked")
    debug("  protect: F = frozen, . = trainable")

    device = next(model.parameters()).device

    for name in after["params"]:
        if name not in before["params"]:
            continue

        p = name_to_param[name]
        before_vals = before["params"][name].flatten().to(device)
        after_vals = after["params"][name].flatten().to(device)
        n = before_vals.numel()

        debug(f"\n  {name}  ({n} elem)  shape={tuple(p.shape)}")

        # ---- Mask overlay ----
        mask_flat = None
        if sparsity_key is not None and name.endswith(".weight"):
            mod_name = name.rsplit(".weight", 1)[0]
            mod_masks = before.get("masks", {}).get(mod_name, {})
            if sparsity_key in mod_masks:
                mask_flat = mod_masks[sparsity_key].flatten()

        # ---- Freeze protection ----
        is_full_frozen = name in before.get("frozen_param_names", [])
        elem_freeze_flat = None
        if name in before.get("element_freeze_masks", {}):
            elem_freeze_flat = before["element_freeze_masks"][name].flatten()

        # ---- Per-element lines ----
        diff = after_vals - before_vals
        limit = n if line_cap is None else min(line_cap, n)

        protect = ["."] * limit
        if is_full_frozen:
            protect = ["F"] * limit
        if elem_freeze_flat is not None:
            ef = elem_freeze_flat[:limit]
            for i in range(limit):
                if ef[i].item():
                    protect[i] = "F"

        n_changed = 0
        n_protected = 0

        for idx in range(limit):
            b = before_vals[idx].item()
            a = after_vals[idx].item()
            d = diff[idx].item()
            m = int(mask_flat[idx].item()) if mask_flat is not None else ""
            c = protect[idx]
            if c == "F":
                n_protected += 1
            if not torch.isclose(torch.tensor(d), torch.zeros_like(torch.tensor(d)),
                                 atol=atol, rtol=rtol):
                n_changed += 1

            debug(f"    [{idx:4d}] ({b:+.4f}, {a:+.4f}, {d:+.4f}, {m}, {c})")

        if line_cap is not None and n > line_cap:
            remaining = diff[line_cap:]
            changed_mask = ~torch.isclose(remaining, torch.zeros_like(remaining),
                                          atol=atol, rtol=rtol)
            n_changed += changed_mask.sum().item()
            if is_full_frozen:
                n_protected += n - line_cap
            elif elem_freeze_flat is not None:
                n_protected += elem_freeze_flat[line_cap:].sum().item()
            debug(f"    ... ({n - line_cap} more elements)")

        debug(f"    summary: {n_changed}/{n} changed  |  {n_protected} protected  |  {n - n_protected} trained")

    debug(f"\n{'=' * 70}\n")
