"""
Phase 4: Create Mixed Weights + Retrain with Frozen Weights

Workflow:
1. Load sparse fine-tuned model from phase3_sparse.pt
2. Load masks from phase2_masks.pt
3. Create mixed weights: w_orig at mask positions + w' at non-mask positions
4. Evaluate (4A) - accuracy BEFORE training
5. Freeze sparse weights + biases + BN
6. Retrain with only mask=0 weights updating
7. Evaluate (4B) - accuracy AFTER training
8. Compare (4A) vs (4B)
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from copy import deepcopy

from vgg import VGG16
from sparse import save_original_weights, create_mixed_weights, apply_nm_sparsity
from train import (
    set_seed,
    create_dataloaders,
    evaluate,
    train_one_epoch,
    append_loss_history,
)


def get_trainable_params(
    model: nn.Module, masks: dict, freeze_biases=True, freeze_bn=True
):
    """
    Split parameters into trainable (mask=0 weights) and frozen (mask=1 weights, biases, BN).
    Returns two lists of parameters for creating optimizer parameter groups.
    
    This properly handles weight decay by excluding frozen params from the optimizer
    or giving them zero weight decay.
    """
    trainable_params = []  # mask=0 weights (to be retrained)
    frozen_params = []     # mask=1 weights (sparse, frozen), biases, BN

    for name, param in model.named_parameters():
        is_weight = ".weight" in name and ".bias" not in name
        is_bias = ".bias" in name and "bn" not in name.lower()
        is_bn = "bn" in name.lower() or "batchnorm" in name.lower()

        module_name = name.replace(".weight", "").replace(".bias", "")
        mask = masks.get(module_name, None)

        if mask is not None and is_weight:
            # Split this parameter into trainable (mask=0) and frozen (mask=1) portions
            # We'll handle this by creating a parameter group with requires_grad=False
            # for the frozen portion... but PyTorch doesn't support partial parameter freezing
            # 
            # Alternative: Use gradient hooks + restore weights after optimizer step
            param.requires_grad = True
            frozen_params.append(param)
        elif is_bias and freeze_biases:
            param.requires_grad = False
            frozen_params.append(param)
        elif is_bn and freeze_bn:
            param.requires_grad = False
            frozen_params.append(param)
        else:
            trainable_params.append(param)

    return trainable_params, frozen_params


def freeze_sparse_weights(
    model: nn.Module, masks: dict, optimizer=None, freeze_biases=True, freeze_bn=True
):
    """
    Freeze weights at mask=1 positions (Phase 3 sparse weights).
    Returns a callback to restore frozen weights after each optimizer step
    (to protect against weight decay and momentum).
    If optimizer is provided, also resets momentum buffers for mask=1 positions.
    """
    # Store (param, mask, original_values) for mask=1 weights that need protection
    frozen_store = []

    for name, param in model.named_parameters():
        is_weight = ".weight" in name and ".bias" not in name
        is_bias = ".bias" in name and "bn" not in name.lower()
        is_bn = "bn" in name.lower() or "batchnorm" in name.lower()

        module_name = name.replace(".weight", "").replace(".bias", "")
        mask = masks.get(module_name, None)

        if mask is not None and is_weight:
            mask = mask.to(param.device)
            # Save original mask=1 values (Phase 3 weights, to be preserved)
            orig_values = param.data.clone()
            frozen_store.append((param, mask, orig_values))
            # Zero gradients at mask=1 positions (so only mask=0 get gradients)
            grad_mask = (mask == 0).float()  # 1 at mask=0, 0 at mask=1
            param.register_hook(lambda grad, gm=grad_mask: grad * gm)

        elif is_bias and freeze_biases:
            param.requires_grad = False

        elif is_bn and freeze_bn:
            param.requires_grad = False

    frozen_weight = sum((m == 1).sum().item() for m in masks.values())
    trainable_weight = sum((m == 0).sum().item() for m in masks.values())
    print(
        f"[Freeze] Weights: {int(frozen_weight):,} frozen, {int(trainable_weight):,} trainable"
    )

    # Return a callback that restores mask=1 weights after each optimizer step
    # This is needed because SGD weight_decay still updates params even when grad=0
    def post_step_callback():
        for param, mask, orig_values in frozen_store:
            with torch.no_grad():
                # Restore mask=1 positions to original Phase 3 values
                mask_1 = mask == 1
                param.data[mask_1] = orig_values[mask_1]
                # Update stored values for mask=0 positions (they were trained)
                orig_values[mask == 0] = param.data[mask == 0]

        # Reset momentum buffer for mask=1 positions to prevent accumulation
        if optimizer is not None:
            for param, mask, _ in frozen_store:
                if param in optimizer.state:
                    state = optimizer.state[param]
                    if 'momentum_buffer' in state:
                        # Zero out momentum buffer at mask=1 positions
                        state['momentum_buffer'][mask == 1] = 0

    return post_step_callback


def run(epochs=30, lr=0.01, checkpoint_interval=5):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data"

    runs_dir = Path("runs/week3")
    sparse_path = runs_dir / "phase3_sparse.pt"
    masks_path = runs_dir / "phase2_masks.pt"
    checkpoint_path = runs_dir / "phase4_mixed.pt"
    loss_csv_path = runs_dir / "phase4_loss.csv"

    print("=" * 60)
    print("PHASE 4: Mixed Weights + Retrain")
    print(f"Device: {device}, LR: {lr}")
    print("=" * 60)

    # Load masks
    print(f"Loading masks from {masks_path}...")
    masks_data = torch.load(masks_path, map_location="cpu")
    masks = masks_data["masks"]
    N, M = masks_data["N"], masks_data["M"]
    print(f"Loaded {N}:{M} masks for {len(masks)} layers")

    # Check if already complete
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if ckpt.get("epoch", 0) >= epochs:
            print(f"Phase 4 already complete ({ckpt['epoch']}/{epochs} epochs)")
            model = VGG16(
                num_classes=10, batch_norm=True, classifier_hidden=512, dropout_p=0.5
            )
            model.load_state_dict(ckpt["model"])
            model = model.to(device)

            # Evaluate without masks (all weights)
            val_loss, val_acc = evaluate(
                model, create_dataloaders(data_dir, 128)[1], device
            )
            print(f"4B - After retrain (all weights): val_acc = {val_acc:.2f}%")

            # Evaluate with masks applied (sparse inference)
            model_masked = deepcopy(model)
            apply_nm_sparsity(model_masked, masks)
            val_loss_masked, val_acc_masked = evaluate(
                model_masked, create_dataloaders(data_dir, 128)[1], device
            )
            print(f"4B - After retrain (with masks): val_acc = {val_acc_masked:.2f}%")

            print(f"\nCOMPARISON:")
            print(f"  (4B) After retrain (all weights): {val_acc:.2f}%")
            print(f"  (4B) After retrain (with masks): {val_acc_masked:.2f}%")
            return

    # Load sparse fine-tuned model
    print(f"Loading sparse model from {sparse_path}...")
    model = VGG16(num_classes=10, batch_norm=True, classifier_hidden=512, dropout_p=0.5)
    ckpt = torch.load(sparse_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"Loaded sparse model")

    # Load DENSE model and save those as original
    dense_model = VGG16(
        num_classes=10, batch_norm=True, classifier_hidden=512, dropout_p=0.5
    )
    dense_path = runs_dir / "phase1_dense.pt"
    dense_ckpt = torch.load(dense_path, map_location="cpu")
    dense_model.load_state_dict(dense_ckpt["model"])
    save_original_weights(dense_model, buffer_name="orig_weights")
    print(f"Saved dense weights as original (from phase1)")

    # Copy saved weights info to model so create_mixed_weights can access them
    model._saved_weights_info = dense_model._saved_weights_info

    # Create mixed weights: dense at mask=1 + sparse at mask=0
    print(f"Creating mixed weights...")
    create_mixed_weights(model, masks)

    # Save BN stats BEFORE training (will be restored for sparse evaluation)
    bn_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_stats[name] = (module.running_mean.clone(), module.running_var.clone())
    print(f"Saved BN stats for {len(bn_stats)} BatchNorm layers")

    # Move model to device FIRST
    model = model.to(device)

    # Evaluate BEFORE training (4A) - sparse accuracy
    val_loader = create_dataloaders(data_dir, 128)[1]
    from copy import deepcopy
    from sparse import apply_nm_sparsity
    model_masked = deepcopy(model)
    # BN stats are already from pre-training, no need to restore
    apply_nm_sparsity(model_masked, masks)
    val_loss_before, val_acc_before = evaluate(model_masked, val_loader, device)
    print(f"4A - Before retrain (sparse): val_acc = {val_acc_before:.2f}%")

    # Freeze sparse weights, biases, BN
    print(f"Freezing sparse weights, biases, and BN...")
    
    train_loader, val_loader = create_dataloaders(data_dir, 128)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Now pass optimizer to freeze_sparse_weights to reset momentum buffers
    post_step_callback = freeze_sparse_weights(
        model, masks, optimizer=optimizer, freeze_biases=True, freeze_bn=True
    )

    history = []

    print(f"Starting retraining from epoch 1/{epochs}")
    start_time = time.time()

    val_loss = 0.0
    val_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            post_step_callback=post_step_callback
        )
        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, device)

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_data)

        print(
            f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f}/{train_acc:.2f}% | "
            f"Val: {val_loss:.4f}/{val_acc:.2f}%"
        )

        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )
            print(f"  Checkpoint saved at epoch {epoch+1}")

        append_loss_history(loss_csv_path, [epoch_data])

    elapsed = time.time() - start_time
    print(f"\nPhase 4 complete in {elapsed:.1f}s")

    # Evaluate AFTER training (4B) - without masks (all weights)
    val_loss_after, val_acc_after = evaluate(model, val_loader, device)
    print(f"4B - After retrain (all weights): val_acc = {val_acc_after:.2f}%")

    # Evaluate with masks applied (sparse inference)
    # Restore BN stats to Phase 4 pre-training values (saved before training)
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and name in bn_stats:
            module.running_mean, module.running_var = bn_stats[name]

    model_masked = deepcopy(model)
    apply_nm_sparsity(model_masked, masks)
    val_loss_masked, val_acc_masked = evaluate(model_masked, val_loader, device)
    print(f"4B - After retrain (with masks): val_acc = {val_acc_masked:.2f}%")

    # Compare
    print(f"\nCOMPARISON:")
    print(f"  (4A) Before retrain: {val_acc_before:.2f}%")
    print(f"  (4B) After retrain (all weights): {val_acc_after:.2f}%")
    print(f"  (4B) After retrain (with masks): {val_acc_masked:.2f}%")

    torch.save(
        {
            "epoch": epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_acc": val_acc,
        },
        checkpoint_path,
    )

    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    run()
