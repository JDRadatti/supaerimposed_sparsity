"""
gate_losses.py - Pluggable loss functions for the gate model.

Each function returns {"loss": Tensor, "bce": float, "aux": float}.
"""

import torch
import torch.nn.functional as F


def baseline_loss(logits, labels):
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    return {"loss": bce, "bce": bce.item(), "aux": 0.0}


def entropy_regularized_loss(logits, labels, p_true, lambda_ent=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    p_hat = torch.sigmoid(logits).mean()
    aux = (p_hat - p_true) ** 2
    loss = bce + lambda_ent * aux
    return {"loss": loss, "bce": bce.item(), "aux": aux.item()}


def accuracy_weighted_loss(logits, labels, pos_weight=2.0):
    weight = torch.tensor(pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
    return {"loss": bce, "bce": bce.item(), "aux": 0.0}


def confidence_calibrated_loss(logits, labels, lambda_conf=0.05):
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    with torch.no_grad():
        wrong = (logits > 0) != labels.bool()
    if wrong.any():
        aux = torch.abs(logits[wrong]).mean()
    else:
        aux = torch.tensor(0.0, device=logits.device)
    loss = bce + lambda_conf * aux
    return {"loss": loss, "bce": bce.item(), "aux": aux.item()}


def flops_aware_loss(logits, labels, lambda_flops=0.1):
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    p_dense = torch.sigmoid(logits).mean()
    loss = bce + lambda_flops * p_dense
    return {"loss": loss, "bce": bce.item(), "aux": p_dense.item()}


_LOSS_REGISTRY = {
    "baseline": baseline_loss,
    "entropy_regularized": entropy_regularized_loss,
    "accuracy_weighted": accuracy_weighted_loss,
    "confidence_calibrated": confidence_calibrated_loss,
    "flops_aware": flops_aware_loss,
}


def compute_loss(logits, labels, loss_mode="baseline", **kwargs):
    if loss_mode not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss_mode '{loss_mode}'. "
            f"Available: {list(_LOSS_REGISTRY.keys())}"
        )
    return _LOSS_REGISTRY[loss_mode](logits, labels, **kwargs)
