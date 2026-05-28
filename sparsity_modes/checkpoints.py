"""
Checkpoint save and load utilities.
Persists model state, masks, freeze info, and BN buffers.
"""
import torch
import torch.nn as nn
import torch.optim as optim

from .masks import _patch_forward_for_masks
from .freeze import (
    clear_all_frozen,
    clear_element_freeze,
    register_frozen_param,
    _element_freeze_masks,
)
from .bn_buffers import _bn_buffers


def save_checkpoint(
    model: nn.Module,
    path: str,
    epoch: int | None = None,
    optimizer: optim.Optimizer | None = None,
    metrics: dict | None = None,
):
    """Save model state, masks, and freezing info to a checkpoint file.

    Args:
        model: The model to save.
        path: File path for the checkpoint.
        epoch: Optional epoch number to store in the checkpoint.
        optimizer: Optional optimizer whose state dict will be saved.
        metrics: Optional dict of metrics to store.
    """
    checkpoint: dict = {
        'model_state_dict': model.state_dict(),
        'masks': {},
        'frozen_param_names': [],
        'element_freeze_masks': {},
        'bn_buffers': {},
    }

    # Save masks
    for mname, module in model.named_modules():
        if not hasattr(module, 'nm_masks'):
            continue
        saved = {k: v.cpu() for k, v in module.nm_masks.items() if v is not None}
        if saved:
            checkpoint['masks'][mname] = saved

    # Save frozen param names
    from .freeze import frozen_parameter_names
    checkpoint['frozen_param_names'] = frozen_parameter_names(model)

    # Save element freeze masks
    name_to_param = dict(model.named_parameters())
    for name, p in name_to_param.items():
        if p in _element_freeze_masks:
            checkpoint['element_freeze_masks'][name] = _element_freeze_masks[p].cpu()

    # Persist per-key BN buffers (cpu)
    for skey, mdict in _bn_buffers.items():
        checkpoint['bn_buffers'][skey] = {
            mname: {bk: bv.cpu() for bk, bv in bd.items()}
            for mname, bd in mdict.items()
        }

    if epoch is not None:
        checkpoint['epoch'] = epoch
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, path)
    print(f"[SAVE] Checkpoint saved to {path}")
    if metrics and 'eval' in metrics:
        print(f"  eval summary:")
        for key, (loss, acc) in metrics['eval'].items():
            print(f"    {key:<10}  loss={loss:.4f}  acc={acc:.2f}%")


def load_checkpoint(
    model: nn.Module,
    path: str,
    strict: bool = True,
    device: torch.device | None = None,
) -> dict:
    """Load model state, masks, and freezing info from a checkpoint file.

    Args:
        model: The model to load into (should have the same architecture).
        path: File path of the checkpoint.
        strict: Whether to enforce strict key matching in ``load_state_dict``.
        device: Device to load tensors onto (default: model's current device).

    Returns:
        The checkpoint dict (useful for accessing epoch, metrics, etc.).
    """
    map_location = device
    checkpoint = torch.load(path, weights_only=True, map_location=map_location)

    # Restore nm_masks and patch forwards
    for mname, module in model.named_modules():
        saved_masks = checkpoint.get('masks', {}).get(mname)
        if saved_masks is None:
            continue
        if not hasattr(module, 'nm_masks'):
            module.nm_masks = {'dense': None}
        for key, mask in saved_masks.items():
            module.nm_masks[key] = mask.to(module.weight.device)
        _patch_forward_for_masks(module)

    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    # Restore frozen info
    clear_all_frozen()
    name_to_param = dict(model.named_parameters())
    for name in checkpoint.get('frozen_param_names', []):
        if name in name_to_param:
            register_frozen_param(name_to_param[name])

    clear_element_freeze()
    for name, mask in checkpoint.get('element_freeze_masks', {}).items():
        if name in name_to_param:
            _element_freeze_masks[name_to_param[name]] = mask.bool()

    # Restore per-key BN buffers
    _bn_buffers.clear()
    for skey, mdict in checkpoint.get('bn_buffers', {}).items():
        _bn_buffers[skey] = mdict  # kept on cpu; moved to active device at restore time
    if _bn_buffers:
        print(f"  Restored BN buffers for keys: {list(_bn_buffers.keys())}")
    else:
        print(f"  [BN] No BN buffers found in checkpoint")

    print(f"[LOAD] Checkpoint loaded from {path}")
    saved_metrics = checkpoint.get('metrics', {})
    if 'eval' in saved_metrics:
        print(f"  original eval:")
        for key, (loss, acc) in saved_metrics['eval'].items():
            print(f"    {key:<10}  loss={loss:.4f}  acc={acc:.2f}%")
    return checkpoint
