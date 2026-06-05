"""
Evaluation utilities for testing all sparsity modes.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import vgg
from .masks import set_active_sparsity, get_active_sparsity, _debug_mask_state
from .bn_buffers import restore_bn_stats
from .logger import debug


def evaluate_all_sparsity_modes(
        model: nn.Module, 
        loader: DataLoader, 
        device: torch.device,
    ) -> dict[str, tuple[float, float]]:
    """
    Evaluates the model on all registered sparsity modes plus dense mode.

    For each sparsity key the corresponding saved BN buffers (if any)
    are restored so that BN running statistics match that mask's activations.

    Args:
        model: The nn.Module to evaluate.
        loader: The DataLoader for evaluation.
        device: The torch.device to use.

    Returns:
        A dictionary mapping sparsity key (e.g., ``"dense"``, ``"1:4"``)
        to a tuple of (average_loss, accuracy).
    """
    results = {}
    
    # Get all registered sparsity keys
    keys = set()
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'nm_masks'):
            keys.update(module.nm_masks.keys())
            break 
    sparsity_keys = sorted(list(keys))
    if 'dense' not in sparsity_keys:
        sparsity_keys.insert(0, 'dense')

    original_active_key = get_active_sparsity(model)

    for key in sparsity_keys:
        set_active_sparsity(model, key)
        restore_bn_stats(model, key)
        _debug_mask_state(model, f"evaluating key='{key}'")
        avg_loss, accuracy = vgg.evaluate(model, loader, device)
        results[key] = (avg_loss, accuracy)
        debug(f"[EVAL] Sparsity '{key}': Avg Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
    
    set_active_sparsity(model, original_active_key)
    restore_bn_stats(model, original_active_key)
    return results
