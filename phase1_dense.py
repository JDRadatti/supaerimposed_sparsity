"""
Phase 1: Train Dense VGG-16

Trains a VGG-16 model from scratch on CIFAR-10.
Saves checkpoint and loss history to runs/week3/
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from vgg import VGG16
from train import (
    set_seed,
    create_dataloaders,
    evaluate,
    train_one_epoch,
    load_loss_history,
    append_loss_history,
)


def run(seed, epochs, batch_size, checkpoint_interval):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data"

    runs_dir = Path("runs/week3")
    runs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = runs_dir / "phase1_dense.pt"
    loss_csv_path = runs_dir / "phase1_loss.csv"

    print("=" * 60)
    print("PHASE 1: Train Dense VGG-16 (30 epochs)")
    print(f"Device: {device}")
    print("=" * 60)

    model = VGG16(num_classes=10, batch_norm=True, classifier_hidden=512, dropout_p=0.5)

    ckpt = None
    start_epoch = 0
    history = []

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)

        if start_epoch >= epochs:
            print(f"Phase 1 already complete ({start_epoch}/{epochs} epochs)")
            val_loss, val_acc = evaluate(
                model, create_dataloaders(data_dir, batch_size)[1], device
            )
            print(f"Final val acc: {val_acc:.2f}%")
            return

        print(f"Resuming from epoch {start_epoch}/{epochs}")
        history = load_loss_history(loss_csv_path)

    else:
        print("Starting fresh from epoch 1")

    model = model.to(device)

    train_loader, val_loader = create_dataloaders(data_dir, batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if ckpt is not None:
        optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        scheduler.load_state_dict(ckpt.get("scheduler", scheduler.state_dict()))
        for _ in range(start_epoch):
            scheduler.step()

    print(f"Starting training from epoch {start_epoch + 1}/{epochs}")
    start_time = time.time()

    val_loss = 0.0
    val_acc = 0.0

    for epoch in range(start_epoch, epochs):
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
    print(f"\nPhase 1 complete in {elapsed:.1f}s")
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
