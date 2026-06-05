"""
Parameter freezing utilities.
Handles freezing individual parameters and element-wise positions.
"""
import torch
import torch.nn as nn

from .logger import debug


# Module-level state for frozen params
_frozen_params: set[torch.nn.Parameter] = set()
_element_freeze_masks: dict[torch.nn.Parameter, torch.Tensor] = {}


# ---------------------------------------------------------------------------
# PARAMETER-LEVEL FREEZE
# ---------------------------------------------------------------------------

def register_frozen_param(p: torch.nn.Parameter):
    """Mark a Parameter as frozen (no update ever)."""
    _frozen_params.add(p)
    setattr(p, "_frozen", True)


def unregister_frozen_param(p: torch.nn.Parameter):
    """Remove a Parameter from the frozen set."""
    _frozen_params.discard(p)
    if hasattr(p, "_frozen"):
        delattr(p, "_frozen")


def is_frozen(p: torch.nn.Parameter) -> bool:
    return p in _frozen_params or getattr(p, "_frozen", False)


def _snapshot_frozen_positions() -> dict:
    """Cache current values of all frozen positions (element-wise + full param).

    Used before ``optimizer.step()`` so they can be restored afterwards
    (weight decay would otherwise modify them despite zero gradients).
    """
    snap = {}
    for p, freeze_mask in _element_freeze_masks.items():
        if freeze_mask.any():
            snap[id(p)] = p.data[freeze_mask].clone()
    for p in _frozen_params:
        if id(p) not in snap:
            snap[id(p)] = p.data.clone()
    return snap


def _restore_frozen_positions(snap: dict) -> None:
    """Restore frozen positions (element-wise + full param) from a snapshot."""
    for p, freeze_mask in _element_freeze_masks.items():
        pid = id(p)
        if pid in snap:
            p.data[freeze_mask] = snap[pid]
    for p in _frozen_params:
        pid = id(p)
        if pid in snap:
            p.data[:] = snap[pid]


def clear_all_frozen():
    for p in list(_frozen_params):
        if hasattr(p, "_frozen"):
            delattr(p, "_frozen")
    _frozen_params.clear()
    _element_freeze_masks.clear()
    debug("[FREEZE] All frozen flags cleared.")


def zero_frozen_gradients(optimizer):
    """Iterate through optimizer's param groups and zero grads of frozen params and element-wise frozen positions."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None and is_frozen(p):
                p.grad.zero_()
    for p, freeze_mask in _element_freeze_masks.items():
        if p.grad is not None:
            p.grad[freeze_mask] = 0.0


def all_trainable_parameters(model: torch.nn.Module):
    """
    Returns a list of (name, parameter) tuples for every parameter that
    requires gradients (i.e. `requires_grad == True`).  These are the
    parameters you could freeze if you wanted to.
    """
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def frozen_parameter_names(model: torch.nn.Module):
    """
    Returns a list of the *names* of parameters that have been marked as frozen.
    """
    frozen_names = []
    for name, p in model.named_parameters():
        if is_frozen(p):
            frozen_names.append(name)
    return frozen_names


# ---------------------------------------------------------------------------
# ELEMENT-WISE FREEZE
# ---------------------------------------------------------------------------

def register_element_freeze(param: torch.nn.Parameter, freeze_mask: torch.Tensor):
    """Freeze individual elements of a Parameter (gradient zeroed at those positions).

    Args:
        param: The parameter to freeze elements of.
        freeze_mask: Boolean tensor of the same shape as param. ``True`` = freeze that element.
    """
    freeze_mask = freeze_mask.to(device=param.device, dtype=torch.bool)
    if param in _element_freeze_masks:
        _element_freeze_masks[param] |= freeze_mask
    else:
        _element_freeze_masks[param] = freeze_mask


def clear_element_freeze():
    """Clear all element-level freeze masks."""
    _element_freeze_masks.clear()


# ---------------------------------------------------------------------------
# USER-FACING FREEZE HELPERS
# ---------------------------------------------------------------------------

def freeze_biases(model: torch.nn.Module):
    """Freeze all bias parameters in the model (no gradient updates)."""
    count = 0
    for module in model.modules():
        if hasattr(module, 'bias') and module.bias is not None:
            register_frozen_param(module.bias)
            count += 1
    debug(f"[FREEZE] {count} bias parameters frozen.")


def freeze_mask(model: torch.nn.Module, sparsity_key: str):
    """Freeze the weight elements that are *kept* (mask == 1) under the given sparsity mask.

    After calling this, the positions selected by the N:M mask will not receive gradient updates.
    Useful for e.g. fine-tuning only the pruned (inverse-mask) weights.
    """
    count = 0
    for module in model.modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if not hasattr(module, 'nm_masks') or sparsity_key not in module.nm_masks:
            continue
        mask = module.nm_masks[sparsity_key]
        if mask is None:
            continue
        register_element_freeze(module.weight, mask.bool())
        count += 1
    debug(f"[FREEZE] {count} modules mask '{sparsity_key}' frozen.")


def freeze_inverse_mask(model: torch.nn.Module, sparsity_key: str):
    """Freeze the weight elements that are *pruned* (mask == 0) under the given sparsity mask.

    After calling this, the positions NOT selected by the N:M mask will not receive gradient updates.
    Useful for e.g. fine-tuning only the sparse (mask) weights.
    """
    count = 0
    for module in model.modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if not hasattr(module, 'nm_masks') or sparsity_key not in module.nm_masks:
            continue
        mask = module.nm_masks[sparsity_key]
        if mask is None:
            continue
        register_element_freeze(module.weight, mask.eq(0))
        count += 1
    debug(f"[FREEZE] {count} modules inverse mask '{sparsity_key}' frozen.")


def freeze_batchnorm(model: torch.nn.Module):
    """Freeze all BatchNorm weight and bias parameters in the model."""
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if module.weight is not None:
                register_frozen_param(module.weight)
                count += 1
            if module.bias is not None:
                register_frozen_param(module.bias)
                count += 1
    debug(f"[FREEZE] {count} BatchNorm parameters frozen.")


def unfreeze_all():
    """Unfreeze everything: clear all frozen params and element-level freeze masks."""
    clear_all_frozen()
    debug("[FREEZE] All parameters unfrozen.")


def freeze_non_masked_layers(model: nn.Module, sparsity_key: str):
    """Freeze the entire weight tensor of Conv/Linear layers not covered by the given mask."""
    count = 0
    for module in model.modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if hasattr(module, 'nm_masks') and sparsity_key in module.nm_masks:
            continue
        if module.weight is not None:
            register_frozen_param(module.weight)
            count += 1
    debug(f"[FREEZE] {count} non-masked layers frozen.")


def freeze_all(model: nn.Module):
    """Freeze every single parameter in the model (full-param freeze).

    Useful as a sanity check: training with everything frozen should leave
    every weight unchanged and accuracy should match the pre-training eval.
    """
    count = 0
    for p in model.parameters():
        register_frozen_param(p)
        count += 1
    debug(f"[FREEZE] All {count} parameters frozen.")
