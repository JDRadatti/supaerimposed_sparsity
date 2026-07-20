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
from .logger import debug, info


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
    on_epoch: Callable[[int, float, float, float, float, float], None] | None = None,
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
        debug(f"[TRAIN] Sparsity mode '{sparsity_key}' active.")
    else:
        set_active_sparsity(model, "dense")
        restore_bn_stats(model, "dense")
        debug("[TRAIN] Dense mode.")

    #
    # Freeze registration
    #
    if freeze_params:
        for p in freeze_params:
            register_frozen_param(p)
        debug(f"[TRAIN] {len(freeze_params)} parameters frozen.")
    else:
        debug("[TRAIN] No frozen parameters.")

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
        current_lr = scheduler.get_last_lr()[0]
        info(f"Epoch {epoch:3d}:  train_loss={loss_sum / total:.4f}  train_acc={100.0 * correct / total:.2f}%"
             f"  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  lr={current_lr:.6f}")

        if on_epoch is not None:
            on_epoch(
                epoch,
                loss_sum / total,
                100.0 * correct / total,
                val_loss,
                val_acc,
                current_lr,
            )

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


def train_joint(
    model: nn.Module,
    sparsity_keys: list[str],
    epochs: int = 200,
    lr: float = 0.1,
    label_smoothing: float = 0.1,
    cutout: int = 16,
    scheduler_fn: Callable | None = None,
    verify_frequency: int = 5,
    on_epoch: Callable[[int, float, float, float, float, float], None] | None = None,
):
    """Joint training of all sparsity modes (Slimmable Neural Networks approach).

    For each batch, computes the loss for every sparsity mode and accumulates
    gradients across all modes before a single optimizer step.  No parameters
    are frozen — every weight is updated by every mode.  Each mode maintains
    its own BatchNorm running statistics (Switchable BN).

    Args:
        model: The nn.Module to train.
        sparsity_keys: List of sparsity keys to jointly train, e.g.
            ``["1:32", "1:16", "1:8", "1:4", "dense"]``.  Must already
            have masks injected via ``inject_switchable_mask``.
        epochs: Number of training epochs.
        lr: Initial learning rate.
        label_smoothing: Label smoothing factor for cross-entropy loss.
        cutout: Cutout size (0 to disable).
        scheduler_fn: ``(optimizer) -> scheduler`` factory.  Defaults to
            ``StepLR(step_size=6, gamma=0.5)``.
        verify_frequency: Verify frozen positions every N epochs (unused
            in joint training but kept for API compatibility).
        on_epoch: Callback ``(epoch, train_loss, train_acc, val_loss,
            val_acc, lr)`` invoked after each epoch.

    Returns:
        The trained model (same object, modified in-place).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #
    # Restore initial BN stats for all modes (fresh start if none saved yet)
    #
    for key in sparsity_keys:
        restore_bn_stats(model, key)
    debug(f"[JOINT] Initialized BN stats for {len(sparsity_keys)} modes: {sparsity_keys}")

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
    # Main training loop
    #
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()

            for key in sparsity_keys:
                set_active_sparsity(model, key)
                restore_bn_stats(model, key)
                outputs = model(images)
                loss = F.cross_entropy(outputs, targets, label_smoothing=label_smoothing)
                loss.backward()

                # Capture the BN running stats updated by this forward pass
                save_active_bn_stats(model)

            optimizer.step()

            # Use the last mode's outputs for training metrics
            loss_sum += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += images.size(0)

        scheduler.step()

        val_loss, val_acc = vgg.evaluate(model, val_loader, device)
        current_lr = scheduler.get_last_lr()[0]
        info(f"Epoch {epoch:3d}:  train_loss={loss_sum / total:.4f}  train_acc={100.0 * correct / total:.2f}%"
             f"  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  lr={current_lr:.6f}")

        if on_epoch is not None:
            on_epoch(
                epoch,
                loss_sum / total,
                100.0 * correct / total,
                val_loss,
                val_acc,
                current_lr,
            )

    return model
