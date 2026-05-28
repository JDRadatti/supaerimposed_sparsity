"""
BatchNorm buffer management for per-sparsity-mode statistics.
Each sparsity mode maintains its own running_mean and running_var.
"""
import torch
import torch.nn as nn

from .masks import get_active_sparsity


# Module-level state for BN buffers
_bn_buffers: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
"""Maps sparsity_key -> module_name -> {running_mean, running_var, num_batches_tracked}."""


def _collect_bn_stats(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    """Capture current BN running stats for every BatchNorm module.

    Returns:
        ``{module_name: {"running_mean": tensor, "running_var": tensor,
                         "num_batches_tracked": tensor}}``
    """
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            stats[name] = {
                "running_mean": module.running_mean.detach().clone(),
                "running_var": module.running_var.detach().clone(),
                "num_batches_tracked": module.num_batches_tracked.detach().clone(),
            }
    return stats


def _apply_bn_stats(model: nn.Module, bn_dict: dict[str, dict[str, torch.Tensor]]) -> None:
    """Restore BN running stats from a dict produced by ``_collect_bn_stats``."""
    for name, module in model.named_modules():
        if name in bn_dict:
            module.running_mean.copy_(bn_dict[name]["running_mean"])
            module.running_var.copy_(bn_dict[name]["running_var"])
            module.num_batches_tracked.copy_(bn_dict[name]["num_batches_tracked"])


def save_active_bn_stats(model: nn.Module) -> None:
    """Save the current BN buffers under the currently active sparsity key."""
    key = get_active_sparsity(model)
    _bn_buffers[key] = _collect_bn_stats(model)
    n = len(_bn_buffers[key])
    print(f"[BN] Saved BN buffers for key '{key}' ({n} BN layers).")


def restore_bn_stats(model: nn.Module, sparsity_key: str) -> None:
    """Restore BN buffers for *sparsity_key*.  No‑op if no buffers have been saved."""
    if sparsity_key in _bn_buffers:
        _apply_bn_stats(model, _bn_buffers[sparsity_key])
        print(f"[BN] Restored BN buffers for key '{sparsity_key}'.")
    else:
        print(f"[BN] No saved BN buffers for key '{sparsity_key}', using current.")
