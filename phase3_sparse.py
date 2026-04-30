"""
Phase 3: Sparse Fine-tune

Workflow:
1. Load dense model from phase1_dense.pt
2. Load and apply 1:4 masks from phase2_masks.pt
3. Fine-tune with lower LR (only non-masked weights update)
4. Save to phase3_sparse.pt
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from vgg import VGG16
from sparse import apply_nm_sparsity
from train import (
    set_seed,
    create_dataloaders,
    evaluate,
    train_one_epoch,
    append_loss_history,
)


def run(epochs=30, lr=0.01, checkpoint_interval=5):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data"

    runs_dir = Path("runs/week3")
    dense_path = runs_dir / "phase1_dense.pt"
    masks_path = runs_dir / "phase2_masks.pt"
    checkpoint_path = runs_dir / "phase3_sparse.pt"
    loss_csv_path = runs_dir / "phase3_loss.csv"

    print("=" * 60)
    print("PHASE 3: Sparse Fine-tune (30 epochs)")
    print(f"Device: {device}, LR: {lr}")
    print("=" * 60)

    # Check if already complete
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if ckpt.get("epoch", 0) >= epochs:
            print(f"Phase 3 already complete ({ckpt['epoch']}/{epochs} epochs)")
            model = VGG16(
                num_classes=10, batch_norm=True, classifier_hidden=512, dropout_p=0.5
            )
            model.load_state_dict(ckpt["model"])
            val_loss, val_acc = evaluate(
                model, create_dataloaders(data_dir, 128)[1], device
            )
            print(f"Final val acc: {val_acc:.2f}%")
            return

    # Load dense model
    print(f"Loading model from {dense_path}...")
    model = VGG16(num_classes=10, batch_norm=True, classifier_hidden=512, dropout_p=0.5)
    ckpt = torch.load(dense_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"Loaded dense model")

    # Load masks
    print(f"Loading masks from {masks_path}...")
    masks_data = torch.load(masks_path, map_location="cpu")
    masks = masks_data["masks"]
    N, M = masks_data["N"], masks_data["M"]
    print(f"Loaded {N}:{M} masks for {len(masks)} layers")

    # Apply masks to model weights
    print(f"Applying masks to model weights...")
    apply_nm_sparsity(model, masks)

    model = model.to(device)

    train_loader, val_loader = create_dataloaders(data_dir, 128)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = []

    print(f"Starting sparse fine-tuning from epoch 1/{epochs}")
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
            f"Val: {val_loss:.4f}/{val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.5f}"
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
    print(f"\nPhase 3 complete in {elapsed:.1f}s")
    print(f"Final val acc: {val_acc:.2f}%")

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
    print(f"Loss history saved to {loss_csv_path}")


if __name__ == "__main__":
    run()

