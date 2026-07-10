"""
gate_v3.py — Feature-Based Dynamic Gate (SGS Frontier Approach)

Like gate_v2, but the gate takes backbone stem features instead of raw images.
The stem (block 1, always dense) extracts features that are shared with
the backbone, giving the gate richer input for routing decisions.

Training: SGS-style online frontier label generation
  - Run frozen backbone in all 5 sparsity modes per batch
  - Assign each sample the cheapest mode it is correct in
  - Gate takes stem features (backbone.features[:7]) → predicts mode
  - Cross-entropy loss between gate prediction and frontier label

=================================================================
  RESULTS
=================================================================
  Gate Accuracy:          19.55%
  Gated System Accuracy:  91.46%
  Oracle Accuracy:        96.78%
  Weighted Cost:          42.3%

  Baselines:
    Always dense:  92.20%
    Always 1:32:   88.26%
  Improvement over 1:32:  +3.20%
  Gap to dense:           0.74%
  Gap to oracle:          5.32%
  Mode Distribution:
    dense    35.6%
    1:4      11.9%
    1:8      17.3%
    1:16     17.2%
    1:32     18.0%

  Results saved →  ./runs_gate_v3/results.txt
  Checkpoint saved → ./runs_gate_v3/gate_checkpoint.pt
=================================================================
"""
import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from vgg import build_model, create_dataloaders
from sparsity_modes import load_checkpoint, set_active_sparsity, restore_bn_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs_gate_v3"
CHECKPOINT_PATH = "./runs/progressive_sparse_dense_final.pt"
DATA_DIR = "./data"

MODES = ["dense", "1:4", "1:8", "1:16", "1:32"]
NUM_MODES = len(MODES)
COSTS = torch.tensor([1.0, 0.25, 0.125, 0.0625, 0.03125])

GATE_EPOCHS = 30
GATE_LR = 1e-3
GATE_BATCH_SIZE = 128
GATE_WARMUP = 3
GATE_WEIGHT_DECAY = 1e-4

STEM_SLICE = slice(None, 24)  # backbone.features[:24] → blocks 1-3 incl. 3rd maxpool → [B, 256, 4, 4]
SUFFIX_SLICE = slice(24, None)  # backbone.features[24:] → blocks 4-5


class DynamicGate(nn.Module):
    """MLP gate: backbone block-3 features → 5 sparsity mode logits (~42K params)."""

    def __init__(self, in_channels: int = 256, num_modes: int = NUM_MODES):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_modes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = x.flatten(1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------


def load_backbone(path: str = CHECKPOINT_PATH):
    log.info("Loading backbone from %s", path)
    model = build_model()
    ckpt = load_checkpoint(model, path)
    eval_metrics = ckpt.get("metrics", {}).get("eval", {})
    if eval_metrics:
        parts = [f"{k}: acc={v[1]:.2f}%" for k, v in sorted(eval_metrics.items())]
        log.info("  Loaded eval: %s", " | ".join(parts))
    return model


# ---------------------------------------------------------------------------
# Stem feature extraction
# ---------------------------------------------------------------------------


def extract_stem(backbone, images, device):
    """Run backbone stem (block 1, always dense) to get features for gate input."""
    set_active_sparsity(backbone, "dense")
    restore_bn_stats(backbone, "dense", verbose=False)
    return backbone.features[STEM_SLICE](images)


def run_suffix(backbone, stem_features):
    """Run backbone suffix (features[7:], avgpool, classifier)."""
    x = backbone.features[SUFFIX_SLICE](stem_features)
    x = backbone.avgpool(x)
    x = torch.flatten(x, 1)
    return backbone.classifier(x)


# ---------------------------------------------------------------------------
# Per-mode output computation
# ---------------------------------------------------------------------------


def compute_per_mode_outputs(backbone, images, targets, device):
    """Run frozen backbone in all 5 sparsity modes.

    Returns:
        per_mode_outputs:  [B, 5, 10]  logits per sample per mode
        per_mode_correct:   [B, 5]      boolean correctness
    """
    B = images.size(0)
    per_mode_outputs = torch.zeros(B, NUM_MODES, 10, device=device)
    per_mode_correct = torch.zeros(B, NUM_MODES, dtype=torch.bool, device=device)

    with torch.no_grad():
        for mode_idx, mode_key in enumerate(MODES):
            set_active_sparsity(backbone, mode_key)
            restore_bn_stats(backbone, mode_key, verbose=False)
            backbone.eval()
            outputs = backbone(images)
            per_mode_outputs[:, mode_idx] = outputs
            per_mode_correct[:, mode_idx] = outputs.argmax(dim=1) == targets

    return per_mode_outputs, per_mode_correct


# ---------------------------------------------------------------------------
# Frontier label generation (SGS)
# ---------------------------------------------------------------------------


def generate_frontier_labels(correct_per_mode: torch.Tensor) -> torch.Tensor:
    """Assign each sample the cheapest sparsity mode where it is correct.

    Checks modes from cheapest (1:32, index 4) to most expensive (1:4, index 1).
    Falls back to dense (index 0) when no mode is correct.
    """
    B = correct_per_mode.size(0)
    labels = torch.zeros(B, dtype=torch.long, device=correct_per_mode.device)

    for mode_idx in range(NUM_MODES - 1, 0, -1):  # 4, 3, 2, 1
        mask = correct_per_mode[:, mode_idx] & (labels == 0)
        labels[mask] = mode_idx

    return labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _compute_static_baselines(backbone, loader, device):
    """Per-mode accuracy and oracle upper bound (no gate)."""
    per_mode_correct_total = torch.zeros(NUM_MODES, device=device)
    per_mode_total = torch.zeros(NUM_MODES, device=device)
    total = 0
    oracle_correct = 0

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            B = images.size(0)

            _, correct_batch = compute_per_mode_outputs(backbone, images, targets, device)

            for idx in range(NUM_MODES):
                per_mode_correct_total[idx] += correct_batch[:, idx].sum().item()
                per_mode_total[idx] += B

            oracle_correct += correct_batch.any(dim=1).sum().item()
            total += B

    per_mode_acc = {
        MODES[i]: (per_mode_correct_total[i] / per_mode_total[i] * 100).item()
        for i in range(NUM_MODES)
    }
    oracle_acc = oracle_correct / total * 100
    return per_mode_acc, oracle_acc


def evaluate_gate(backbone, gate, loader, device):
    """Evaluate gate accuracy and gated system accuracy.

    Gated inference: stem (dense) → gate → suffix at predicted sparsity.
    """
    backbone.eval()
    gate.eval()

    total = 0
    gate_correct = 0
    gated_correct = 0
    oracle_correct = 0
    mode_counts = torch.zeros(NUM_MODES, device=device)

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            B = images.size(0)

            # Frontier labels and oracle (full backbone in all modes)
            per_mode_outputs, per_mode_correct = compute_per_mode_outputs(
                backbone, images, targets, device
            )
            frontier_labels = generate_frontier_labels(per_mode_correct)

            # Gate: stem (dense) → predict mode
            stem_features = extract_stem(backbone, images, device)
            logits = gate(stem_features)
            gate_choices = logits.argmax(dim=1)

            gate_correct += (gate_choices == frontier_labels).sum().item()

            # True gated inference: suffix at predicted sparsity
            for i, mode_key in enumerate(MODES):
                mask = gate_choices == i
                n = mask.sum().item()
                if n == 0:
                    continue
                set_active_sparsity(backbone, mode_key)
                restore_bn_stats(backbone, mode_key, verbose=False)
                outputs = run_suffix(backbone, stem_features[mask])
                gated_preds = outputs.argmax(dim=1)
                gated_correct += (gated_preds == targets[mask]).sum().item()

            oracle_correct += per_mode_correct.any(dim=1).sum().item()

            mode_counts += torch.bincount(gate_choices, minlength=NUM_MODES)

            total += B

    return {
        "gate_acc": gate_correct / total * 100,
        "gated_acc": gated_correct / total * 100,
        "oracle_acc": oracle_correct / total * 100,
        "mode_distribution": {
            MODES[i]: (mode_counts[i] / total * 100).item()
            for i in range(NUM_MODES)
        },
        "weighted_cost": sum(
            (mode_counts[i] / total).item() * COSTS[i].item()
            for i in range(NUM_MODES)
        ) * 100,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _compute_class_weights(backbone, loader, device):
    """Compute inverse-frequency class weights from frontier label distribution."""
    counts = torch.zeros(NUM_MODES, device=device)
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            _, correct = compute_per_mode_outputs(backbone, images, targets, device)
            labels = generate_frontier_labels(correct)
            counts += torch.bincount(labels, minlength=NUM_MODES)

    counts = counts.clamp(min=1)
    weights = counts.sum() / (NUM_MODES * counts)
    weights = weights / weights.sum() * NUM_MODES  # normalise so mean = 1
    return weights


def train_gate(backbone, gate, train_loader, val_loader, epochs=GATE_EPOCHS,
               lr=GATE_LR, device=None, class_weights=None):
    """Train gate using SGS frontier labels with stem features as input."""
    if device is None:
        device = DEVICE

    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    gate = gate.to(device)
    loss_fn = F.cross_entropy
    if class_weights is not None:
        class_weights = class_weights.to(device)
        loss_fn = lambda logits, targets: F.cross_entropy(logits, targets, weight=class_weights)

    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=lr, weight_decay=GATE_WEIGHT_DECAY
    )
    warmup = lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=GATE_WARMUP
    )
    cosine = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - GATE_WARMUP
    )
    scheduler = lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[GATE_WARMUP]
    )

    best_gated_acc = 0.0
    best_state = None
    best_metrics = None

    log_lines = [
        "epoch,loss,gate_acc,gated_acc,oracle_acc,weighted_cost,"
        + ",".join(f"frac_{m.replace(':', '_')}" for m in MODES)
    ]

    for epoch in range(epochs):
        gate.train()
        total_loss = 0.0
        n_samples = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            # Extract stem features (always dense)
            stem_features = extract_stem(backbone, images, device)

            # Compute frontier labels (full backbone in all modes)
            _, per_mode_correct = compute_per_mode_outputs(
                backbone, images, targets, device
            )
            frontier_labels = generate_frontier_labels(per_mode_correct)

            # Gate prediction from stem features
            logits = gate(stem_features)
            loss = loss_fn(logits, frontier_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            n_samples += images.size(0)

        scheduler.step()
        avg_loss = total_loss / n_samples

        val_metrics = evaluate_gate(backbone, gate, val_loader, device)

        log.info(
            "Epoch %2d/%d: Loss=%.4f | GateAcc=%.2f%% | GatedAcc=%.2f%% | "
            "Oracle=%.2f%% | Cost=%.1f%%",
            epoch + 1, epochs, avg_loss,
            val_metrics["gate_acc"], val_metrics["gated_acc"],
            val_metrics["oracle_acc"], val_metrics["weighted_cost"],
        )
        dist = val_metrics["mode_distribution"]
        log.info(
            "  Modes: %s",
            " | ".join(f"{m}={dist[m]:.1f}%" for m in MODES),
        )

        csv_row = [
            str(epoch + 1),
            f"{avg_loss:.6f}",
            f"{val_metrics['gate_acc']:.2f}",
            f"{val_metrics['gated_acc']:.2f}",
            f"{val_metrics['oracle_acc']:.2f}",
            f"{val_metrics['weighted_cost']:.1f}",
        ]
        csv_row.extend(
            f"{val_metrics['mode_distribution'][m]:.1f}" for m in MODES
        )
        log_lines.append(",".join(csv_row))

        if val_metrics["gated_acc"] > best_gated_acc:
            best_gated_acc = val_metrics["gated_acc"]
            best_state = {k: v.cpu().clone() for k, v in gate.state_dict().items()}
            best_metrics = val_metrics

    if best_state is not None:
        gate.load_state_dict(best_state)

    return gate, best_gated_acc, best_metrics, log_lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    print("=" * 65)
    print("  DYNAMIC GATE v3 — Feature-Based SGS Frontier Approach")
    print("=" * 65)
    print(f"  Device:      {DEVICE}")
    print(f"  Backbone:    {CHECKPOINT_PATH}")
    print(f"  Modes:       {MODES}")
    print(f"  Stem slice:  backbone.features[:24] → [B, 256, 4, 4]")
    print(f"  Gate:        avgpool(1×1) → Linear(256→128→64→5) + BN")
    print(f"  Optimizer:   AdamW (lr={GATE_LR}, wd={GATE_WEIGHT_DECAY})")
    print(f"  Scheduler:   {GATE_WARMUP}ep warmup + cosine to {GATE_EPOCHS}ep")
    print(f"  Gate params: ~42K")
    print()

    backbone = load_backbone(CHECKPOINT_PATH)
    backbone = backbone.to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    log.info("Creating data loaders …")
    train_loader, val_loader = create_dataloaders(
        DATA_DIR, batch_size=GATE_BATCH_SIZE, num_workers=4, cutout=0
    )

    log.info("Computing static per-mode baselines …")
    per_mode_acc, oracle_acc = _compute_static_baselines(
        backbone, val_loader, DEVICE
    )
    print("  Static per-mode accuracies:")
    for m in MODES:
        print(f"    {m:<8} {per_mode_acc[m]:.2f}%")
    print(f"  Oracle accuracy (upper bound):  {oracle_acc:.2f}%")
    print()

    gate = DynamicGate()
    n_params = sum(p.numel() for p in gate.parameters())
    log.info("Gate created (%s params)", n_params)

    log.info("Computing class weights from frontier label distribution …")
    class_weights = _compute_class_weights(backbone, train_loader, DEVICE)
    log.info("  Class weights: %s", [f"{w:.2f}" for w in class_weights.tolist()])

    log.info("Training gate …")
    gate, best_gated_acc, best_metrics, log_lines = train_gate(
        backbone, gate, train_loader, val_loader,
        epochs=GATE_EPOCHS, lr=GATE_LR, device=DEVICE,
        class_weights=class_weights,
    )

    ckpt_path = os.path.join(BASE_DIR, "gate_checkpoint.pt")
    torch.save(
        {
            "gate_state_dict": gate.state_dict(),
            "best_gated_acc": best_gated_acc,
            "best_metrics": best_metrics,
        },
        ckpt_path,
    )
    log.info("Checkpoint saved → %s", ckpt_path)

    log_path = os.path.join(BASE_DIR, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log.info("Training log saved → %s", log_path)

    log.info("Final evaluation …")
    final_metrics = evaluate_gate(backbone, gate, val_loader, DEVICE)

    static_dense = per_mode_acc["dense"]
    static_1_32 = per_mode_acc["1:32"]

    results_path = os.path.join(BASE_DIR, "results.txt")
    with open(results_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("  DYNAMIC GATE v3 — Feature-Based SGS Frontier Approach\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("Static Per-Mode Accuracies:\n")
        for m in MODES:
            f.write(f"  {m:<8} {per_mode_acc[m]:.2f}%\n")
        f.write(f"\n  Oracle Accuracy (upper bound):  {oracle_acc:.2f}%\n")
        f.write("\n" + "─" * 65 + "\n\n")
        f.write("Gating Results:\n")
        f.write(f"  Gate Accuracy:          {final_metrics['gate_acc']:.2f}%\n")
        f.write(f"  Gated System Accuracy:  {final_metrics['gated_acc']:.2f}%\n")
        f.write(f"  Weighted Cost:          {final_metrics['weighted_cost']:.1f}%\n")
        f.write(f"\n  Static baselines:\n")
        f.write(f"    Always dense:  {static_dense:.2f}%\n")
        f.write(f"    Always 1:32:   {static_1_32:.2f}%\n")
        f.write(f"  Improvement over 1:32:  {final_metrics['gated_acc'] - static_1_32:+.2f}%\n")
        f.write(f"  Gap to dense:           {static_dense - final_metrics['gated_acc']:.2f}%\n")
        f.write(f"  Gap to oracle:          {oracle_acc - final_metrics['gated_acc']:.2f}%\n")
        f.write("\nMode Distribution:\n")
        for m in MODES:
            f.write(f"  {m:<8} {final_metrics['mode_distribution'][m]:.1f}%\n")
        f.write("\n" + "=" * 65 + "\n")

    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Gate Accuracy:          {final_metrics['gate_acc']:.2f}%")
    print(f"  Gated System Accuracy:  {final_metrics['gated_acc']:.2f}%")
    print(f"  Oracle Accuracy:        {final_metrics['oracle_acc']:.2f}%")
    print(f"  Weighted Cost:          {final_metrics['weighted_cost']:.1f}%")
    print()
    print("  Baselines:")
    print(f"    Always dense:  {static_dense:.2f}%")
    print(f"    Always 1:32:   {static_1_32:.2f}%")
    print(f"  Improvement over 1:32:  {final_metrics['gated_acc'] - static_1_32:+.2f}%")
    print(f"  Gap to dense:           {static_dense - final_metrics['gated_acc']:.2f}%")
    print(f"  Gap to oracle:          {oracle_acc - final_metrics['gated_acc']:.2f}%")
    print("  Mode Distribution:")
    for m in MODES:
        print(f"    {m:<8} {final_metrics['mode_distribution'][m]:.1f}%")
    print()
    print(f"  Results saved →  {results_path}")
    print(f"  Checkpoint saved → {ckpt_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
