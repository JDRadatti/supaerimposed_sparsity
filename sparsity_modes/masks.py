"""
Sparsity mask injection and activation.
Handles registering masks on modules and switching between sparsity modes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._sparse import _nm_mask_conv2d_axis, _nm_mask_linear
from .logger import debug, warning


def inject_switchable_mask(
    model: nn.Module,
    sparsity_key: str,
    N: int,
    M: int,
    conv_axis: str = "out",
    linear_orientation: str = "row",
):
    """
    Registers an N:M sparsity mask for a given key into eligible Conv2d / Linear layers.
    Skips the very first and very last such layers.

    Args:
        model:               The target ``nn.Module``.
        sparsity_key:        A unique identifier for this sparsity mode (e.g., "1:4").
        N:                   Number of weights to keep per group.
        M:                   Group size.
        conv_axis:           Axis for Conv2d mask (``"out"`` or ``"in"``).
        linear_orientation:  Orientation for Linear mask (``"row"`` or ``"col"``).
    """
    # Identify first & last Conv/Linear to skip
    conv_linear = [
        m
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
        and hasattr(m, "weight")
    ]
    first_last = set(conv_linear[:1] + conv_linear[-1:]) if conv_linear else set()

    for module in conv_linear:
        if module in first_last:
            continue

        # Create a dictionary to hold masks if it doesn't exist
        if not hasattr(module, 'nm_masks'):
            module.nm_masks = {}
            module.nm_masks['dense'] = None

        # Generate the N:M Mask
        if sparsity_key not in module.nm_masks:
            w = module.weight.detach()
            if isinstance(module, nn.Conv2d):
                mask = _nm_mask_conv2d_axis(w, N, M, axis=conv_axis)
            else: # nn.Linear
                mask = _nm_mask_linear(w, N, M, orientation=linear_orientation)
            module.nm_masks[sparsity_key] = mask.type_as(w)

        _patch_forward_for_masks(module)


def _patch_forward_for_masks(module: nn.Module) -> None:
    """Patch forward on a module that already has ``nm_masks`` set up.

    If the module already has ``nm_masks`` and isn't already patched, this
    overrides ``module.forward`` with a closure that applies the active mask.
    """
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return
    if not hasattr(module, 'nm_masks'):
        return

    if not hasattr(module, '_original_forward'):
        module._original_forward = module.forward

    if not hasattr(module, 'active_mask_key'):
        module.active_mask_key = 'dense'

    def _make_forward(m):
        def _forward(input):
            current_mask_key = getattr(m, 'active_mask_key', 'dense')

            if current_mask_key == 'dense':
                return m._original_forward(input)

            mask = m.nm_masks.get(current_mask_key)
            if mask is None:
                warning(f"Mask '{current_mask_key}' not found for module. Falling back to dense.")
                return m._original_forward(input)

            masked_weight = m.weight * mask.to(m.weight.device)

            if isinstance(m, nn.Linear):
                return F.linear(input, masked_weight, m.bias)
            return F.conv2d(input, masked_weight, m.bias,
                            m.stride, m.padding, m.dilation, m.groups)
        return _forward

    module.__dict__['forward'] = _make_forward(module)


def set_active_sparsity(model: nn.Module, sparsity_key: str):
    """
    Sets the active sparsity mask key for all eligible modules in the model.
    'sparsity_key' can be a registered key (e.g., "1:4") or "dense".
    If 'dense' is passed, it ensures no specific mask is active, effectively
    restoring the module to its original forward behavior.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'nm_masks'):
            
            if sparsity_key == 'dense':
                if not hasattr(module, 'active_mask_key') or module.active_mask_key != 'dense':
                    module.active_mask_key = 'dense'
            elif sparsity_key in module.nm_masks:
                if not hasattr(module, 'active_mask_key') or module.active_mask_key != sparsity_key:
                    module.active_mask_key = sparsity_key
            else:
                pass # Keep the current active mask if the requested key isn't available.


def get_active_sparsity(model: nn.Module) -> str:
    """
    Returns the active sparsity key for the first eligible module found that has 'active_mask_key'.
    Assumes all eligible modules in a model are configured to use the same active mask.
    Returns 'dense' if no specific mask is active or if no configured modules are found.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'active_mask_key'):
            return module.active_mask_key
    return "dense"


def _debug_mask_state(model: nn.Module, tag: str = "") -> None:
    """Print the active mask key and patch state of every module with nm_masks."""
    debug(f"\n[DEBUG MASK STATE{(' ' + tag) if tag else ''}]")
    total_conv_linear = sum(
        1 for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, 'weight')
    )
    total_masked = 0
    for name, module in model.named_modules():
        if not hasattr(module, 'nm_masks'):
            continue
        total_masked += 1
        keys = sorted(module.nm_masks.keys())
        active = getattr(module, 'active_mask_key', 'N/A')
        is_patched = (hasattr(module, '_original_forward') and
                      module.forward is not module._original_forward)
        debug(f"  {name}: keys={keys} active={active} patched={is_patched}")
    debug(f"  [{total_masked}/{total_conv_linear} Conv/Linear layers have nm_masks]")
