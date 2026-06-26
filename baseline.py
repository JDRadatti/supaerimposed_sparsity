"""
baseline.py - Train dense and 1:32 baselines, evaluate and compare.

Saves checkpoints and comparison files to ./baseline/.
Uses prune_nm() for pruning (no sparsity_modes framework).
"""
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.optim import lr_scheduler

from vgg import build_model, create_dataloaders, evaluate
from sparsity_modes._sparse import _nm_mask_conv2d_axis, _nm_mask_linear

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.datasets\.cifar")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs_baseline"
DENSE_CKPT = f"{BASE_DIR}/dense.pt"
SPARSE_1_32_CKPT = f"{BASE_DIR}/sparse_1_32.pt"
DENSE_EPOCHS = 90
SPARSE_EPOCHS = 30


@torch.no_grad()
def prune_nm(model, N, M, conv_axis="out", linear_orientation="row"):
    """Apply N:M pruning to eligible Conv2d/Linear layers (skip first & last)."""
    conv_linear = [
        m for m in model.modules()
        if (isinstance(m, (nn.Conv2d, nn.Linear))) and hasattr(m, "weight")
    ]
    first_last = set(conv_linear[:1] + conv_linear[-1:]) if conv_linear else set()

    for module in conv_linear:
        if module in first_last:
            continue
        w = module.weight.detach()
        if isinstance(module, nn.Conv2d):
            mask = _nm_mask_conv2d_axis(w, N, M, axis=conv_axis)
        else:
            mask = _nm_mask_linear(w, N, M, orientation=linear_orientation)
        prune.custom_from_mask(module, name="weight", mask=mask)


def _snapshot_params(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _write_comparison(path, before, after):
    with open(path, "w") as f:
        f.write("Parameter comparison (before -> after fine-tuning):\n\n")
        total_changed = 0
        total_params = 0
        for name in after:
            if name not in before:
                continue
            b = before[name].flatten()
            a = after[name].flatten()
            n = b.numel()
            changed = (~torch.isclose(b, a, atol=1e-8, rtol=1e-5)).sum().item()
            total_changed += changed
            total_params += n
            f.write(f"  {name}  ({n} elem)  changed={changed}/{n}\n")
        f.write(f"\n  Total: {total_changed}/{total_params} parameters changed\n")
    print(f"  [COMPARISON] saved to {path}")


def train_model(model, epochs, lr=0.1, label_smoothing=0.0, cutout=0,
                scheduler_fn=None, freeze_conv1=False):
    """Train model with standard SGD loop."""
    if freeze_conv1:
        for i, child in enumerate(model.features):
            if i >= 7:
                break
            for p in child.parameters():
                p.requires_grad = False
    model = model.to(DEVICE)
    train_loader, val_loader = create_dataloaders("./data", 128, 4, cutout=cutout)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    if scheduler_fn is not None:
        scheduler = scheduler_fn(optimizer)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, targets, label_smoothing=label_smoothing)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += images.size(0)
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, DEVICE)
        print(f"  Epoch {epoch:2d}/{epochs}: Train Acc {100*correct/total:.2f}% | Val Acc {val_acc:.2f}%")

    return model


def main():
    print("=" * 55)
    print("  BASELINE: Dense + 1:32 Pruned")
    print("=" * 55)

    os.makedirs(BASE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Train dense baseline
    # ------------------------------------------------------------------
    if os.path.exists(DENSE_CKPT):
        print(f"\n[1] Loading cached dense baseline from {DENSE_CKPT}...")
        model_dense = build_model().to(DEVICE)
        model_dense.load_state_dict(torch.load(DENSE_CKPT, map_location=DEVICE)["model_state_dict"])
    else:
        print(f"\n[1] Training dense baseline ({DENSE_EPOCHS} epochs)...")
        model_dense = build_model().to(DEVICE)
        snap_before = _snapshot_params(model_dense)
        train_model(
            model_dense, DENSE_EPOCHS, lr=0.1, label_smoothing=0.1, cutout=16,
            scheduler_fn=lambda opt: lr_scheduler.CosineAnnealingLR(opt, T_max=DENSE_EPOCHS),
        )
        _, val_loader = create_dataloaders("./data", 128, 4)
        dense_loss, dense_acc = evaluate(model_dense, val_loader, DEVICE)
        print(f"  Dense: loss={dense_loss:.4f}  acc={dense_acc:.2f}%")
        torch.save({"model_state_dict": model_dense.state_dict(), "loss": dense_loss, "acc": dense_acc}, DENSE_CKPT)
        print(f"  Saved to {DENSE_CKPT}")
        snap_after = _snapshot_params(model_dense)
        _write_comparison(f"{BASE_DIR}/dense_comparison.txt", snap_before, snap_after)

    # ------------------------------------------------------------------
    # 2. Prune 1:32 and fine-tune
    # ------------------------------------------------------------------
    _, val_loader = create_dataloaders("./data", 128, 4)

    if os.path.exists(SPARSE_1_32_CKPT):
        print(f"\n[2] Loading cached 1:32 checkpoint from {SPARSE_1_32_CKPT}...")
        model_sparse = build_model().to(DEVICE)
        state = torch.load(SPARSE_1_32_CKPT, map_location=DEVICE)
        model_sparse.load_state_dict(state["model_state_dict"])
    else:
        print(f"\n[2] Pruning 1:32 and fine-tuning ({SPARSE_EPOCHS} epochs)...")
        model_sparse = build_model().to(DEVICE)
        model_sparse.load_state_dict(
            torch.load(DENSE_CKPT, map_location=DEVICE)["model_state_dict"]
        )
        snap_before = _snapshot_params(model_sparse)
        prune_nm(model_sparse, N=1, M=32, conv_axis="out", linear_orientation="row")
        snap_after_prune = _snapshot_params(model_sparse)
        _write_comparison(f"{BASE_DIR}/prune_comparison.txt", snap_before, snap_after_prune)
        train_model(
            model_sparse, SPARSE_EPOCHS, lr=0.01,
            scheduler_fn=lambda opt: lr_scheduler.CosineAnnealingLR(opt, T_max=SPARSE_EPOCHS),
            freeze_conv1=True,
        )
        snap_after = _snapshot_params(model_sparse)
        _write_comparison(f"{BASE_DIR}/sparse_1_32_comparison.txt", snap_after_prune, snap_after)
        sparse_loss, sparse_acc = evaluate(model_sparse, val_loader, DEVICE)
        print(f"  1:32: loss={sparse_loss:.4f}  acc={sparse_acc:.2f}%")
        torch.save({
            "model_state_dict": model_sparse.state_dict(),
            "loss": sparse_loss, "acc": sparse_acc,
        }, SPARSE_1_32_CKPT)
        print(f"  Saved to {SPARSE_1_32_CKPT}")

    # ------------------------------------------------------------------
    # 3. Final evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  RESULTS")
    print("=" * 55)

    dense_loss, dense_acc = evaluate(model_dense, val_loader, DEVICE)
    print(f"  Dense:  loss={dense_loss:.4f}  acc={dense_acc:.2f}%")

    sparse_loss, sparse_acc = evaluate(model_sparse, val_loader, DEVICE)
    print(f"  1:32:   loss={sparse_loss:.4f}  acc={sparse_acc:.2f}%")
    print("=" * 55)

    eval_path = f"{BASE_DIR}/results.txt"
    with open(eval_path, "w") as f:
        f.write("=" * 55 + "\n")
        f.write("  BASELINE RESULTS\n")
        f.write("=" * 55 + "\n")
        f.write(f"  Dense:  loss={dense_loss:.4f}  acc={dense_acc:.2f}%\n")
        f.write(f"  1:32:   loss={sparse_loss:.4f}  acc={sparse_acc:.2f}%\n")
        f.write("=" * 55 + "\n")
    print(f"  Results saved to {eval_path}")


if __name__ == "__main__":
    main()
