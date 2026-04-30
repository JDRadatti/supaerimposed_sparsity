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


def freeze_sparse_weights(
    model: nn.Module, masks: dict, freeze_biases=True, freeze_bn=True
):
    """
    Freeze weights at mask=1 positions, biases, and optionally BatchNorm parameters.
    After this, only weights at mask=0 positions are trainable.
    """
    for name, param in model.named_parameters():
        is_weight = ".weight" in name and ".bias" not in name
        is_bias = ".bias" in name and "bn" not in name.lower()
        is_bn = "bn" in name.lower() or "batchnorm" in name.lower()

        module_name = name.replace(".weight", "").replace(".bias", "")
        mask = masks.get(module_name, None)

        if mask is not None and is_weight:
            mask_binary = mask.clone()
            mask_binary[mask == 1] = 0
            mask_binary[mask == 0] = 1

            param.requires_grad = True
            param.register_hook(lambda grad, mb=mask_binary: grad.mul_(mb))

        elif is_bias and freeze_biases:
            param.requires_grad = False

        elif is_bn and freeze_bn:
            param.requires_grad = False

    frozen_weight = sum(m.sum().item() for m in masks.values())
    trainable_weight = sum((1 - m).sum().item() for m in masks.values())
    print(
        f"[Freeze] Weights: {int(frozen_weight):,} frozen, {int(trainable_weight):,} trainable"
    )


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

    # Evaluate BEFORE training (4A)
    val_loader = create_dataloaders(data_dir, 128)[1]
    val_loss_before, val_acc_before = evaluate(model, val_loader, device)
    print(f"4A - Before retrain: val_acc = {val_acc_before:.2f}%")

    # Move model to device FIRST, then freeze weights
    model = model.to(device)

    # Freeze sparse weights, biases, BN
    print(f"Freezing sparse weights, biases, and BN...")
    freeze_sparse_weights(model, masks, freeze_biases=True, freeze_bn=True)

    train_loader, val_loader = create_dataloaders(data_dir, 128)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = []

    print(f"Starting retraining from epoch 1/{epochs}")
    start_time = time.time()

    val_loss = 0.0
    val_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
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
