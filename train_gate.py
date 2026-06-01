#!/usr/bin/env python3
"""
Train a binary gate to predict whether to use 1:32 or dense mode.

Pipeline:
  1. Load the fully-trained progressive model (all masks + BN buffers).
  2. Generate (feature, binary_label) data from 1:32 mode on the train set.
  3. Train two gate architectures (linear, linear+hidden).
  4. Evaluate gated inference on the test set.
"""
import os
import warnings

import torch
from torch.utils.data import DataLoader, random_split

from vgg import evaluate as eval_model
from vgg import build_model, create_dataloaders
import sparsity_modes as su
from gate import GateModel, generate_gate_data, train_gate, evaluate_gated, _quiet_restore_bn

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.datasets\.cifar")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs"
CKPT_PATH = f"{BASE_DIR}/progressive_sparse_dense_final.pt"


THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def print_threshold_table(name, all_results):
    print(f"\n  {'=' * 68}")
    print(f"  {name}")
    print(f"  {'=' * 68}")
    print(f"  {'thresh':>7} {'1:32->':>8} {'1:32 ok':>8} {'1:32%':>7} "
          f"{'dense->':>8} {'dense ok':>8} {'dense%':>7} "
          f"{'overall':>8}")
    print(f"  {'─' * 68}")
    for t, counts, correct, _ in all_results:
        total = sum(counts.values())
        overall = 100.0 * sum(correct.values()) / total
        c32, cd = counts["1:32"], counts["dense"]
        ok32, okd = correct["1:32"], correct["dense"]
        p32 = 100.0 * ok32 / c32 if c32 else 0.0
        pd = 100.0 * okd / cd if cd else 0.0
        print(f"  {t:>7.2f} {c32:>8d} {ok32:>8d} {p32:>6.2f}% "
              f"{cd:>8d} {okd:>8d} {pd:>6.2f}% {overall:>7.2f}%")
    print(f"  {'─' * 68}")
    print(f"  {'':>7} {'':>8} {'':>8} {'':>7} "
          f"{'':>8} {'':>8} {'':>7} {'87.48%':>7}  (always 1:32)")
    print(f"  {'':>7} {'':>8} {'':>8} {'':>7} "
          f"{'':>8} {'':>8} {'':>7} {'92.20%':>7}  (always dense)")


def main():
    print("=" * 60)
    print("  GATE TRAINING — Binary (1:32 vs dense)")
    print("=" * 60)

    # 1. Load model
    print(f"\n[1] Loading model from {CKPT_PATH}...")
    model = build_model().to(DEVICE)
    su.load_checkpoint(model, CKPT_PATH)
    su.unfreeze_all()
    model.requires_grad_(False)
    su.set_active_sparsity(model, "dense")
    su.restore_bn_stats(model, "dense")

    # Quick sanity eval
    _, val_loader = create_dataloaders("./data", batch_size=128, num_workers=4)
    su.set_active_sparsity(model, "1:32")
    _quiet_restore_bn(model, "1:32")
    loss_32, acc_32 = eval_model(model, val_loader, DEVICE)
    su.set_active_sparsity(model, "dense")
    _quiet_restore_bn(model, "dense")
    loss_d, acc_d = eval_model(model, val_loader, DEVICE)
    print(f"  Baseline — 1:32: {acc_32:.2f}%  |  dense: {acc_d:.2f}%")

    # 2. Create data loaders
    print("\n[2] Creating data loaders...")
    train_loader, val_loader = create_dataloaders("./data", batch_size=128, num_workers=4)

    # 3. Generate gate training data
    print("\n[3] Generating gate training data...")
    dataset = generate_gate_data(model, train_loader, DEVICE)
    pos = dataset.tensors[1].sum().item()
    neg = len(dataset) - pos
    print(f"  Samples: {len(dataset)}  (1:32 correct: {int(pos)} / {100*pos/len(dataset):.1f}%"
          f"  |  1:32 wrong: {int(neg)} / {100*neg/len(dataset):.1f}%)")

    # 4. Split into train / val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    gate_train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    gate_val_loader = DataLoader(val_ds, batch_size=128, num_workers=2)

    # 5. GateA — Linear(16384 -> 1)
    print(f"\n[4] GateA — Linear (MaxPool2d → Flatten → Linear)")
    print(f"  Params: {sum(p.numel() for p in GateModel(0).parameters())}")
    gate_a = GateModel(hidden_dim=0)
    gate_a = train_gate(gate_a, gate_train_loader, gate_val_loader, DEVICE)
    results_a = []
    for t in THRESHOLDS:
        r = evaluate_gated(model, gate_a, val_loader, DEVICE, threshold=t)
        results_a.append((t, *r))
    print_threshold_table("GateA (MaxPool → Linear)", results_a)

    # 6. GateB — Linear(16384 -> 64 -> ReLU -> 1)
    print(f"\n[5] GateB — Linear (MaxPool2d → Flatten → Linear 64 → ReLU → Linear 1)")
    print(f"  Params: {sum(p.numel() for p in GateModel(64).parameters())}")
    gate_b = GateModel(hidden_dim=64)
    gate_b = train_gate(gate_b, gate_train_loader, gate_val_loader, DEVICE)
    results_b = []
    for t in THRESHOLDS:
        r = evaluate_gated(model, gate_b, val_loader, DEVICE, threshold=t)
        results_b.append((t, *r))
    print_threshold_table("GateB (MaxPool → Linear 64 → ReLU → Linear 1)", results_b)


if __name__ == "__main__":
    main()
