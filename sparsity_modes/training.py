"""
Training loop for sparsity modes.
Handles training with frozen parameters and per-mode BN buffers.
"""
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import vgg
from .masks import inject_switchable_mask, set_active_sparsity
from .freeze import (
    register_frozen_param,
    zero_frozen_gradients,
    _snapshot_frozen_positions,
    _restore_frozen_positions,
    clear_all_frozen,
)
from .bn_buffers import save_active_bn_stats, restore_bn_stats
from .evaluation import evaluate_all_sparsity_modes
from .validation import snapshot_parameters, verify_after_training


def train(
    model: nn.Module,
    epochs: int = 10,
    N: int = 0,
    M: int = 0,
    verify_frequency: int = 5,
    freeze_params: list[torch.nn.Parameter] | None = None,
    # ------------------------------------------------------------------
    # Optional improvements (backward-compatible defaults)
    # ------------------------------------------------------------------
    lr: float = 0.1,
    scheduler_fn: Callable | None = None,
    label_smoothing: float = 0.0,
    cutout: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #
    # Sparsity injection
    #
    sparsity_key = f"{N}:{M}"
    if N != 0 and M != 0:
        inject_switchable_mask(
            model, 
            sparsity_key, 
            N=N, 
            M=M, 
            conv_axis="out", 
            linear_orientation="row"
        )
        set_active_sparsity(model, sparsity_key)
        restore_bn_stats(model, sparsity_key)
        print(f"[TRAIN] Sparsity mode '{sparsity_key}' active.")
    else:
        set_active_sparsity(model, "dense")
        restore_bn_stats(model, "dense")
        print("[TRAIN] Dense mode.")

    #
    # Freeze registration
    #
    if freeze_params:
        for p in freeze_params:
            register_frozen_param(p)
        print(f"[TRAIN] {len(freeze_params)} parameters frozen.")
    else:
        print("[TRAIN] No frozen parameters.")

    #
    # Optimizer / scheduler / dataloader
    #
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    if scheduler_fn is not None:
        scheduler = scheduler_fn(optimizer)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
    train_loader, val_loader = vgg.create_dataloaders("./data", 128, 4, cutout=cutout)

    #
    # Take Snapshot
    #
    snapshot = snapshot_parameters(model)

    #
    # Main training loop
    #
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, targets, label_smoothing=label_smoothing)
            loss.backward()

            zero_frozen_gradients(optimizer)
            frozen_snap = _snapshot_frozen_positions()
            optimizer.step()
            _restore_frozen_positions(frozen_snap)

            loss_sum += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += images.size(0)

        scheduler.step()

        val_loss, val_acc = vgg.evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: "
              f"Train Acc {100*correct/total:.2f}% | "
              f"Val Acc {val_acc:.2f}%")

        # 
        # Verify snapshot
        #
        if epoch % verify_frequency == 0:
            verify_after_training(
                model,
                snapshot,
                active_mask_key=sparsity_key,
                frozen_params=freeze_params,
            )
            snapshot = snapshot_parameters(model)

    #
    # Save BN buffers for the active key
    # 
    save_active_bn_stats(model)

    #
    # Clean‑up
    # 
    clear_all_frozen()
    return model
