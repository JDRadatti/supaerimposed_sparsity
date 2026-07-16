"""
Utilities for testing inference equivalence between a sparsity‑patched model
and a truly pruned reference model (weights = zero at pruned positions).
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ._sparse import _nm_mask_conv2d_axis, _nm_mask_linear


def build_truly_pruned(
    model: nn.Module,
    mode_profile: Any,
) -> nn.Module:
    """Create a deep copy of *model* with masked weights set to exactly zero.

    Uses the same N:M mask primitives as ``SparsityModel`` to compute masks,
    then bakes the zeros directly into the weights.  The returned model does
    **not** use forward patching — zeros are real.

    Comparing its outputs against a ``SparsityModel`` in the same mode
    verifies that the patched forward is numerically equivalent to true
    pruning.

    Parameters
    ----------
    model:        Source ``nn.Module`` (e.g. a newly built VGG16).
    mode_profile: A ``ModeProfile`` describing which layers get which N:M mask.

    Returns:
        A new ``nn.Module`` with pruned weights clipped to zero.

    Raises:
        ValueError: If *mode_profile* fails validation against *model*.
    """
    pruned: nn.Module = copy.deepcopy(model)

    # Use the same validation to get the layer→sparsity map.
    from .profiles import _validate_mode_profile
    layer_map = _validate_mode_profile(pruned, mode_profile)

    name_to_module = dict(pruned.named_modules())

    for module_fqn, sparsity in layer_map.items():
        if sparsity == "dense":
            continue

        module = name_to_module[module_fqn]
        N, M = sparsity
        w = module.weight.data

        if isinstance(module, nn.Conv2d):
            mask = _nm_mask_conv2d_axis(w, N, M, axis=mode_profile.conv_axis)
        else:
            mask = _nm_mask_linear(w, N, M, orientation=mode_profile.linear_orientation)

        w.mul_(mask.to(w.device))

    return pruned


def compare_models(
    model_a: nn.Module,
    model_b: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Compare the outputs of two models across a dataloader.

    Runs both models on every batch and computes per‑sample statistics.

    Parameters
    ----------
    model_a:  First model.
    model_b:  Second model.
    loader:   DataLoader providing inputs.
    device:   Device to run on (default: auto‑detect from *model_a*).
    atol:     Absolute tolerance for ``torch.allclose``.
    rtol:     Relative tolerance for ``torch.allclose``.
    max_batches:  Stop after this many batches (``None`` = full dataloader).

    Returns:
        A dict with keys:
        - ``"all_close"``: ``bool`` — ``True`` if all outputs match within tol.
        - ``"max_abs_diff"``: ``float`` — largest per‑element difference.
        - ``"max_rel_diff"``: ``float`` — largest relative difference (where
          reference != 0).
        - ``"n_samples"``: ``int`` — total samples compared.
        - ``"n_different"``: ``int`` — samples where outputs differ.
    """
    if device is None:
        device = next(model_a.parameters()).device

    model_a = model_a.to(device).eval()
    model_b = model_b.to(device).eval()

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    n_samples = 0
    n_different = 0
    all_close = True

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)

            out_a = model_a(inputs)
            out_b = model_b(inputs)

            abs_diff = (out_a - out_b).abs()
            batch_max = abs_diff.max().item()
            if batch_max > max_abs_diff:
                max_abs_diff = batch_max

            # Relative difference (avoid division by zero).
            a_abs = out_a.abs()
            nonzero = a_abs > 1e-6
            if nonzero.any():
                rel_diff = (abs_diff[nonzero] / a_abs[nonzero]).max().item()
                if rel_diff > max_rel_diff:
                    max_rel_diff = rel_diff

            n_samples += inputs.size(0)
            if not torch.allclose(out_a, out_b, atol=atol, rtol=rtol):
                all_close = False
                n_different += inputs.size(0)

    return {
        "all_close": all_close,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "n_samples": n_samples,
        "n_different": n_different,
    }


def compare_single_input(
    model_a: nn.Module,
    model_b: nn.Module,
    x: torch.Tensor,
    device: torch.device | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Run both models on a single input tensor — no dataloader required.

    Parameters
    ----------
    model_a:  First model.
    model_b:  Second model.
    x:        Input tensor of shape ``[batch, *dims]``.
    device:   Device to run on (default: auto‑detect from *model_a*).
    atol:     Absolute tolerance.
    rtol:     Relative tolerance.

    Returns:
        Dict with ``"all_close"`` and ``"max_abs_diff"``.
    """
    if device is None:
        device = next(model_a.parameters()).device

    model_a = model_a.to(device).eval()
    model_b = model_b.to(device).eval()
    x = x.to(device)

    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)

    abs_diff = (out_a - out_b).abs()

    return {
        "all_close": torch.allclose(out_a, out_b, atol=atol, rtol=rtol),
        "max_abs_diff": abs_diff.max().item(),
    }
