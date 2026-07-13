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


def build_truly_pruned(
    model: nn.Module,
    mode_profile: Any,
) -> nn.Module:
    """Create a deep copy of *model* with masked weights set to exactly zero.

    The returned model does **not** use forward patching — the zeros are
    baked into the weights.  Comparing its outputs against a `SparsityModel`
    in the same mode verifies that the patched forward is numerically
    equivalent to true pruning.

    Parameters
    ----------
    model:        The source `nn.Module`.
    mode_profile: A `ModeProfile` describing which layers get which N:M mask.

    Returns:
        A new `nn.Module` with weights zeroed at pruned positions.
    """
    # 1. Deep‑copy the model.
    pruned: nn.Module = copy.deepcopy(model)

    # 2. For each layer covered by the mode_profile's layer_specs:
    #    a. Compute the N:M mask from the current weight.
    #    b. Multiply the weight by the mask in‑place.
    # 3. Return the pruned model (no forward patching needed — zeros are real).

    ...


def compare_models(
    model_a: nn.Module,
    model_b: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare the outputs of two models across a dataloader.

    Runs both models on every batch and computes per‑sample statistics.

    Parameters
    ----------
    model_a:  First model.
    model_b:  Second model.
    loader:   DataLoader providing inputs.
    device:   Device to run on (default: auto‑detect from model_a).
    atol:     Absolute tolerance for `torch.allclose`.
    rtol:     Relative tolerance for `torch.allclose`.

    Returns:
        A dict with keys:
        - `"all_close"`: `bool` — `True` if all outputs match within tol.
        - `"max_abs_diff"`: `float` — largest per‑element difference.
        - `"max_rel_diff"`: `float` — largest relative difference (where
          reference != 0).
        - `"n_samples"`: `int` — total samples compared.
        - `"n_different"`: `int` — samples where outputs differ.
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
        for batch in loader:
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
            mask = a_abs > 1e-6
            if mask.any():
                rel_diff = (abs_diff[mask] / a_abs[mask]).max().item()
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
