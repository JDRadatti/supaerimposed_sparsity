"""
Shared training utilities for ML experiments.

Contains common functions used across multiple phases:
- set_seed
- create_dataloaders
- evaluate
- train_one_epoch
- load_loss_history
- append_loss_history
"""

import time
import csv
import random
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(data_dir: str, batch_size: int = 128):
    """
    Create dataloaders for training and testing CIFAR-10.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size for dataloaders

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Precomputed mean and std for VGG16 on CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # Transforms for training
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Transforms for testing
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model on given dataloader.

    Args:
        model: PyTorch model
        loader: DataLoader to evaluate on
        device: Device to run evaluation on

    Returns:
        Tuple of (average_loss, accuracy_percent)
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += images.size(0)

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model
        loader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Tuple of (average_train_loss, train_accuracy_percent)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += images.size(0)

    return running_loss / total, 100.0 * correct / total


def load_loss_history(csv_path: Path) -> List[Dict]:
    """
    Load loss history from CSV file.

    Args:
        csv_path: Path to loss history CSV

    Returns:
        List of dictionaries with epoch data
    """
    if not csv_path.exists():
        return []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["train_loss"] = float(row["train_loss"])
            row["train_acc"] = float(row["train_acc"])
            row["val_loss"] = float(row["val_loss"])
            row["val_acc"] = float(row["val_acc"])
            row["lr"] = float(row["lr"])
            rows.append(row)
        return rows


def append_loss_history(csv_path: Path, history: List[Dict]):
    """
    Append loss history to CSV file.

    Args:
        csv_path: Path to loss history CSV
        history: List of dictionaries with epoch data
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    mode = "a" if not write_header else "w"
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(history)