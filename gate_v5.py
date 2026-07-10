"""
gate_v5.py — Class-Weighted Gumbel-Softmax Gate with Balanced Batch Sampling

Improvements over v4:
  - Balanced batch sampling (WeightedRandomSampler) forces equal frontier-class
    representation in every batch, preventing the CE loss from gravitating to
    the majority class (88% of samples have frontier = 1:32)
  - Mild class-weighted CE (freq^−0.5) as secondary protection
  - Full-backbone evaluation eliminates BN mismatch from stem/suffix split
    (v4's stem ran dense but suffix ran sparse, dropping ~5% accuracy)
  - L2 logit regularisation prevents gate-logit unbounded growth
  - Milder entropy schedule allows convergence to a stable distribution
  - Longer warmup (20 vs 10 epochs) lets per-sample features develop first
  - Lower learning rate (5e-4 vs 1e-3) for stability

Loss:  weighted_CE + λc·cost − λe·H(p) + λb·balance + λr·|logits|²
"""
from datetime import datetime
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from vgg import build_model
from sparsity_modes import load_checkpoint, set_active_sparsity, restore_bn_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs_gate_v5"
CHECKPOINT_PATH = "./runs/progressive_sparse_dense_final.pt"
DATA_DIR = "./data"

MODES = ["dense", "1:4", "1:8", "1:16", "1:32"]
NUM_MODES = len(MODES)

# --- Split-aware costs (same as v4) ---
COSTS = torch.tensor([1.0000, 0.3501, 0.2417, 0.1876, 0.1605])

STEM_SLICE = slice(None, 7)
SUFFIX_SLICE = slice(7, None)

# --- Training schedule ---
GATE_EPOCHS = 60
WARMUP_EPOCHS = 20

TAU_START = 5.0
TAU_MID = 1.5
TAU_END = 0.2
COST_LAMBDA_START = 0.0
COST_LAMBDA_MID = 0.1
COST_LAMBDA_END = 0.3
ENT_LAMBDA_START = 0.05
ENT_LAMBDA_MID = 0.01
ENT_LAMBDA_END = 0.001
BALANCE_LAMBDA = 0.02
LOGIT_REG_LAMBDA = 0.0005
MIN_CKPT_EPOCH = 20
PARETO_BETA = 0.1

GATE_LR = 5e-4
GATE_BATCH_SIZE = 128
GATE_WARMUP = 5
GATE_WEIGHT_DECAY = 1e-4


# ---------------------------------------------------------------------------
# Gate architecture (same conv gate as v4)
# ---------------------------------------------------------------------------


class DynamicGate(nn.Module):
    def __init__(self, in_channels: int = 64, num_modes: int = NUM_MODES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_modes)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Schedule helpers (same as v4)
# ---------------------------------------------------------------------------


def _linear_schedule(t, start, end):
    return start + (end - start) * t


def _cosine_schedule(t, start, end):
    return end + (start - end) * (1 + math.cos(math.pi * t)) / 2


def get_schedule_params(epoch: int):
    if GATE_EPOCHS <= WARMUP_EPOCHS:
        t = epoch / max(GATE_EPOCHS - 1, 1)
        return {
            "tau": _linear_schedule(t, TAU_START, TAU_END),
            "lambda_cost": _linear_schedule(t, COST_LAMBDA_START, COST_LAMBDA_END),
            "lambda_ent": _linear_schedule(t, ENT_LAMBDA_START, ENT_LAMBDA_END),
        }
    if epoch < WARMUP_EPOCHS:
        t = epoch / max(WARMUP_EPOCHS - 1, 1)
        return {
            "tau": _linear_schedule(t, TAU_START, TAU_MID),
            "lambda_cost": _linear_schedule(t, COST_LAMBDA_START, COST_LAMBDA_MID),
            "lambda_ent": _linear_schedule(t, ENT_LAMBDA_START, ENT_LAMBDA_MID),
        }
    phase2_epochs = GATE_EPOCHS - WARMUP_EPOCHS
    t = (epoch - WARMUP_EPOCHS) / max(phase2_epochs - 1, 1)
    return {
        "tau": _linear_schedule(t, TAU_MID, TAU_END),
        "lambda_cost": _cosine_schedule(t, COST_LAMBDA_MID, COST_LAMBDA_END),
        "lambda_ent": _cosine_schedule(t, ENT_LAMBDA_MID, ENT_LAMBDA_END),
    }


# ---------------------------------------------------------------------------
# Backbone helpers
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


def extract_stem(backbone, images, device):
    set_active_sparsity(backbone, "dense")
    restore_bn_stats(backbone, "dense", verbose=False)
    return backbone.features[STEM_SLICE](images)


# ---------------------------------------------------------------------------
# Per-mode output (used by both training and evaluation)
# ---------------------------------------------------------------------------


def compute_per_mode_outputs(backbone, images, targets, device):
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


def generate_frontier_labels(correct_per_mode):
    B = correct_per_mode.size(0)
    labels = torch.zeros(B, dtype=torch.long, device=correct_per_mode.device)
    for mode_idx in range(NUM_MODES - 1, 0, -1):
        mask = correct_per_mode[:, mode_idx] & (labels == 0)
        labels[mask] = mode_idx
    return labels


# ---------------------------------------------------------------------------
# Balanced data loader  (WeightedRandomSampler per frontier class)
# ---------------------------------------------------------------------------


def build_balanced_loader(backbone, device, batch_size):
    """Collect frontier labels for the full training set and build a
    WeightedRandomSampler-based DataLoader so every batch sees ~equal
    representation of each frontier class.

    Returns (train_loader, val_loader, class_weights).
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=train_tf)
    val_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=False, transform=val_tf)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Pass 1: collect frontier labels with frozen backbone
    temp_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    all_labels = []
    class_counts = torch.zeros(NUM_MODES, dtype=torch.float32, device=device)

    with torch.no_grad():
        for images, targets in temp_loader:
            images, targets = images.to(device), targets.to(device)
            _, correct = compute_per_mode_outputs(backbone, images, targets, device)
            frontier = generate_frontier_labels(correct)
            all_labels.extend(frontier.cpu().tolist())
            class_counts += torch.bincount(frontier, minlength=NUM_MODES).float()

    freqs = class_counts / class_counts.sum()
    class_weights = 1.0 / (freqs.sqrt() + 1e-8)
    class_weights = class_weights / class_weights.sum() * NUM_MODES
    log.info(
        "Frontier labels: %s",
        " | ".join(f"{MODES[i]}={freqs[i]*100:.1f}%% (√w={class_weights[i]:.2f})" for i in range(NUM_MODES)),
    )

    # Per-sample weight = 1 / class_count → each class has equal total weight
    per_sample = [1.0 / (class_counts[label].item() + 1e-6) for label in all_labels]
    sampler = WeightedRandomSampler(per_sample, num_samples=len(all_labels), replacement=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)

    log.info("Balanced sampler created (%d samples)", len(all_labels))
    return train_loader, val_loader, class_weights


# ---------------------------------------------------------------------------
# Multi-component loss
# ---------------------------------------------------------------------------


def compute_gate_loss(logits, frontier_labels, costs, schedule_params, class_weights):
    B = logits.size(0)
    device = logits.device
    costs = costs.to(device)

    tau = schedule_params["tau"]
    gumbel = F.gumbel_softmax(logits, tau=tau, hard=True, dim=1)
    probs = F.softmax(logits, dim=1)

    # Class-weighted CE
    w = class_weights.to(device)
    ce = F.cross_entropy(logits, frontier_labels, reduction="none")
    L_ce = (ce * w[frontier_labels]).mean()

    # Expected cost
    L_cost = (probs * costs).sum(dim=1).mean()

    # Entropy maximisation
    L_entropy = (probs * (probs + 1e-8).log()).sum(dim=1).mean()

    # Load balancing
    f = gumbel.float().mean(dim=0)
    p = probs.mean(dim=0)
    L_balance = (f * p).sum()

    # L2 penalty on logits
    L_logit = (logits ** 2).mean()

    loss = (L_ce
            + schedule_params["lambda_cost"] * L_cost
            + schedule_params["lambda_ent"] * L_entropy
            + BALANCE_LAMBDA * L_balance
            + LOGIT_REG_LAMBDA * L_logit)

    with torch.no_grad():
        H = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
        f_acc = (gumbel.argmax(dim=1) == frontier_labels).float().mean().item()

    return {
        "loss": loss,
        "ce": L_ce.item(),
        "cost_term": L_cost.item(),
        "entropy_mean": H.item(),
        "balance_term": L_balance.item(),
        "frontier_acc": f_acc,
    }


# ---------------------------------------------------------------------------
# Evaluation  (uses full-backbone forward — no BN mismatch)
# ---------------------------------------------------------------------------


def _compute_static_baselines(backbone, loader, device):
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
    return per_mode_acc, oracle_correct / total * 100


def evaluate_gate(backbone, gate, loader, device):
    backbone.eval()
    gate.eval()
    total = 0
    gate_correct = 0
    gated_correct = 0
    oracle_correct = 0
    mode_counts = torch.zeros(NUM_MODES, device=device)
    per_mode_gated_correct = torch.zeros(NUM_MODES, device=device)
    per_mode_total = torch.zeros(NUM_MODES, device=device)

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            B = images.size(0)

            per_mode_outputs, per_mode_correct = compute_per_mode_outputs(
                backbone, images, targets, device)
            frontier_labels = generate_frontier_labels(per_mode_correct)

            stem_features = extract_stem(backbone, images, device)
            logits = gate(stem_features)
            gate_choices = logits.argmax(dim=1)
            gate_correct += (gate_choices == frontier_labels).sum().item()

            # Full-backbone forward per mode — avoids stem/suffix BN mismatch
            for i in range(NUM_MODES):
                mask = gate_choices == i
                n = mask.sum().item()
                if n == 0:
                    continue
                per_mode_total[i] += n
                preds = per_mode_outputs[:, i][mask].argmax(dim=1)
                per_mode_gated_correct[i] += (preds == targets[mask]).sum().item()
                gated_correct += (preds == targets[mask]).sum().item()

            oracle_correct += per_mode_correct.any(dim=1).sum().item()
            mode_counts += torch.bincount(gate_choices, minlength=NUM_MODES)
            total += B

    gated_acc = gated_correct / total * 100
    weighted_cost = sum(
        (mode_counts[i] / total).item() * COSTS[i].item() for i in range(NUM_MODES)
    ) * 100
    per_mode_gated = {}
    for i in range(NUM_MODES):
        if per_mode_total[i] > 0:
            per_mode_gated[MODES[i]] = (per_mode_gated_correct[i] / per_mode_total[i] * 100).item()
        else:
            per_mode_gated[MODES[i]] = 0.0

    return {
        "gate_acc": gate_correct / total * 100,
        "gated_acc": gated_acc,
        "oracle_acc": oracle_correct / total * 100,
        "weighted_cost": weighted_cost,
        "pareto_score": gated_acc - PARETO_BETA * weighted_cost,
        "mode_distribution": {
            MODES[i]: (mode_counts[i] / total * 100).item() for i in range(NUM_MODES)
        },
        "per_mode_gated_acc": per_mode_gated,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_gate(backbone, gate, train_loader, val_loader, class_weights,
               epochs=GATE_EPOCHS, lr=GATE_LR, device=None):
    if device is None:
        device = DEVICE

    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    gate = gate.to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=GATE_WEIGHT_DECAY)
    warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=GATE_WARMUP)
    cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - GATE_WARMUP)
    scheduler = lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[GATE_WARMUP])

    best_pareto = -float("inf")
    best_state = None
    best_metrics = None
    costs = COSTS.clone()

    csv_fields = [
        "epoch", "loss", "ce", "cost_term", "entropy", "balance", "frontier_acc",
        "tau", "lambda_cost", "lambda_ent",
        "gate_acc", "gated_acc", "oracle_acc", "weighted_cost", "pareto",
    ]
    for m in MODES:
        csv_fields.append(f"frac_{m.replace(':', '_')}")
    for m in MODES:
        csv_fields.append(f"acc_{m.replace(':', '_')}")
    log_lines = [",".join(csv_fields)]

    for epoch in range(epochs):
        sched = get_schedule_params(epoch)
        gate.train()
        epoch_loss = epoch_ce = epoch_cost = epoch_ent = epoch_bal = epoch_facc = 0.0
        n_samples = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            stem_features = extract_stem(backbone, images, device)
            _, per_mode_correct = compute_per_mode_outputs(backbone, images, targets, device)
            frontier_labels = generate_frontier_labels(per_mode_correct)
            logits = gate(stem_features)
            result = compute_gate_loss(logits, frontier_labels, costs, sched, class_weights)

            optimizer.zero_grad()
            result["loss"].backward()
            optimizer.step()

            b = images.size(0)
            epoch_loss += result["loss"].item() * b
            epoch_ce += result["ce"] * b
            epoch_cost += result["cost_term"] * b
            epoch_ent += result["entropy_mean"] * b
            epoch_bal += result["balance_term"] * b
            epoch_facc += result["frontier_acc"] * b
            n_samples += b

        scheduler.step()
        avg_loss = epoch_loss / n_samples
        avg_ce = epoch_ce / n_samples
        avg_cost = epoch_cost / n_samples
        avg_ent = epoch_ent / n_samples
        avg_bal = epoch_bal / n_samples
        avg_facc = epoch_facc / n_samples * 100

        val_metrics = evaluate_gate(backbone, gate, val_loader, device)

        log.info(
            "Epoch %2d/%d: τ=%.2f λc=%.3f λe=%.4f | "
            "Loss=%.4f CE=%.4f Cost=%.4f H=%.4f Bal=%.4f FAcc=%.1f%% | "
            "GateAcc=%.2f%% GatedAcc=%.2f%% Cost=%.1f%% Pareto=%.2f",
            epoch + 1, epochs,
            sched["tau"], sched["lambda_cost"], sched["lambda_ent"],
            avg_loss, avg_ce, avg_cost, avg_ent, avg_bal, avg_facc,
            val_metrics["gate_acc"], val_metrics["gated_acc"],
            val_metrics["weighted_cost"], val_metrics["pareto_score"],
        )
        dist = val_metrics["mode_distribution"]
        log.info("  Modes: %s", " | ".join(f"{m}={dist[m]:.1f}%" for m in MODES))
        per_acc = val_metrics["per_mode_gated_acc"]
        log.info("  Per-mode acc: %s", " | ".join(f"{m}={per_acc[m]:.1f}%" for m in MODES))

        csv_row = [
            str(epoch + 1),
            f"{avg_loss:.6f}", f"{avg_ce:.6f}", f"{avg_cost:.6f}",
            f"{avg_ent:.6f}", f"{avg_bal:.6f}", f"{avg_facc:.2f}",
            f"{sched['tau']:.3f}", f"{sched['lambda_cost']:.4f}", f"{sched['lambda_ent']:.4f}",
            f"{val_metrics['gate_acc']:.2f}", f"{val_metrics['gated_acc']:.2f}",
            f"{val_metrics['oracle_acc']:.2f}", f"{val_metrics['weighted_cost']:.1f}",
            f"{val_metrics['pareto_score']:.2f}",
        ]
        csv_row.extend(f"{val_metrics['mode_distribution'][m]:.1f}" for m in MODES)
        csv_row.extend(f"{val_metrics['per_mode_gated_acc'][m]:.1f}" for m in MODES)
        log_lines.append(",".join(csv_row))

        if epoch + 1 >= MIN_CKPT_EPOCH and val_metrics["pareto_score"] > best_pareto:
            best_pareto = val_metrics["pareto_score"]
            best_state = {k: v.cpu().clone() for k, v in gate.state_dict().items()}
            best_metrics = val_metrics
            log.info("  *** New best checkpoint (epoch %d, Pareto=%.2f) ***", epoch + 1, best_pareto)

    if best_state is not None:
        gate.load_state_dict(best_state)
        log.info("Loaded best checkpoint (Pareto=%.2f)", best_pareto)
    else:
        best_metrics = evaluate_gate(backbone, gate, val_loader, device)
        log.info("No checkpoint saved; using final model")

    return gate, best_pareto, best_metrics, log_lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    print("=" * 65)
    print("  DYNAMIC GATE v5 — Balanced-Sampling + Class-Weighted Gumbel-Softmax")
    print("=" * 65)
    print(f"  Device:      {DEVICE}")
    print(f"  Backbone:    {CHECKPOINT_PATH}")
    print(f"  Modes:       {MODES}")
    print(f"  Costs:       {[f'{c:.4f}' for c in COSTS.tolist()]}")
    print(f"  Stem:        features[:7] → block 1 → [B, 64, 16, 16]")
    print(f"  Gate:        Conv(64→32→64→128) → GAP → Linear(128,5)")
    print(f"  Sampling:    WeightedRandomSampler (balanced frontier labels)")
    print(f"  Class wts:   1/√freq  (mild, secondary)")
    print(f"  Loss:        wCE + λc·cost − λe·H(p) + λb·balance + λr·|logits|²")
    print(f"  Schedule:    τ={TAU_START}→{TAU_MID}→{TAU_END}")
    print(f"               λc={COST_LAMBDA_START}→{COST_LAMBDA_MID}→{COST_LAMBDA_END}")
    print(f"               λe={ENT_LAMBDA_START}→{ENT_LAMBDA_MID}→{ENT_LAMBDA_END}")
    print(f"               λr={LOGIT_REG_LAMBDA}")
    print(f"  Ckpt guard:  epoch >= {MIN_CKPT_EPOCH}")
    print(f"  Pareto β:    {PARETO_BETA}")
    print()

    backbone = load_backbone(CHECKPOINT_PATH)
    backbone = backbone.to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    log.info("Building balanced data loader …")
    train_loader, val_loader, class_weights = build_balanced_loader(
        backbone, DEVICE, GATE_BATCH_SIZE)

    log.info("Computing static per-mode baselines …")
    per_mode_acc, oracle_acc = _compute_static_baselines(backbone, val_loader, DEVICE)
    print("  Static per-mode accuracies:")
    for m in MODES:
        print(f"    {m:<8} {per_mode_acc[m]:.2f}%")
    print(f"  Oracle accuracy (upper bound):  {oracle_acc:.2f}%")
    print()

    gate = DynamicGate()
    n_params = sum(p.numel() for p in gate.parameters())
    log.info("Gate created (%s params)", n_params)

    log.info("Training gate …")
    gate, best_pareto, best_metrics, log_lines = train_gate(
        backbone, gate, train_loader, val_loader, class_weights,
        epochs=GATE_EPOCHS, lr=GATE_LR, device=DEVICE)

    ckpt_path = os.path.join(BASE_DIR, "gate_checkpoint.pt")
    torch.save({
        "gate_state_dict": gate.state_dict(),
        "best_pareto": best_pareto,
        "best_metrics": best_metrics,
    }, ckpt_path)
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
        f.write("  DYNAMIC GATE v5 — Balanced-Sampling + Class-Weighted Gumbel-Softmax\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("Improvements over v4:\n")
        f.write("  - WeightedRandomSampler: balanced frontier labels in every batch\n")
        f.write("  - Class-weighted CE (1/√freq) as secondary protection\n")
        f.write("  - Full-backbone evaluation (no stem/suffix BN mismatch)\n")
        f.write("  - L2 logit regularisation\n")
        f.write("  - Milder entropy schedule\n\n")
        f.write("Configuration:\n")
        f.write(f"  Stem slice:  features[:7] → block 1 → [B, 64, 16, 16]\n")
        f.write(f"  Costs:       {[f'{c:.4f}' for c in COSTS.tolist()]}\n")
        f.write(f"  Gate params: {n_params}\n")
        f.write(f"  Sampling:    WeightedRandomSampler (balanced frontier labels)\n")
        f.write(f"  Class wts:   1/√freq (mild, secondary)\n")
        f.write(f"  Loss:        wCE + λc·cost − λe·H(p) + λb·balance + λr·|logits|²\n")
        f.write(f"  Schedule:    τ {TAU_START}→{TAU_MID}→{TAU_END}\n")
        f.write(f"               λc {COST_LAMBDA_START}→{COST_LAMBDA_MID}→{COST_LAMBDA_END}\n")
        f.write(f"               λe {ENT_LAMBDA_START}→{ENT_LAMBDA_MID}→{ENT_LAMBDA_END}\n")
        f.write(f"               λr {LOGIT_REG_LAMBDA}\n")
        f.write(f"  Ckpt guard:  epoch >= {MIN_CKPT_EPOCH}\n")
        f.write(f"  Pareto β:    {PARETO_BETA}\n")
        f.write("\nStatic Per-Mode Accuracies:\n")
        for m in MODES:
            f.write(f"  {m:<8} {per_mode_acc[m]:.2f}%\n")
        f.write(f"\n  Oracle Accuracy (upper bound):  {oracle_acc:.2f}%\n")
        f.write("\n" + "─" * 65 + "\n\n")
        f.write("Gating Results:\n")
        f.write(f"  Gate Accuracy:          {final_metrics['gate_acc']:.2f}%\n")
        f.write(f"  Gated System Accuracy:  {final_metrics['gated_acc']:.2f}%\n")
        f.write(f"  Weighted Cost:          {final_metrics['weighted_cost']:.1f}%\n")
        f.write(f"  Pareto Score:           {final_metrics['pareto_score']:.2f}\n")
        f.write(f"\n  Static baselines:\n")
        f.write(f"    Always dense:  {static_dense:.2f}% (cost 100.0%)\n")
        f.write(f"    Always 1:32:   {static_1_32:.2f}% (cost {COSTS[4].item()*100:.1f}%)\n")
        f.write(f"  Improvement over 1:32:  {final_metrics['gated_acc'] - static_1_32:+.2f}%\n")
        f.write(f"  Gap to dense:           {static_dense - final_metrics['gated_acc']:.2f}%\n")
        f.write(f"  Gap to oracle:          {oracle_acc - final_metrics['gated_acc']:.2f}%\n")
        f.write("\nMode Distribution:\n")
        for m in MODES:
            f.write(f"  {m:<8} {final_metrics['mode_distribution'][m]:.1f}%\n")
        f.write("\nPer-Mode Gated Accuracy:\n")
        for m in MODES:
            f.write(f"  {m:<8} {final_metrics['per_mode_gated_acc'][m]:.1f}%\n")
        f.write("\n" + "=" * 65 + "\n")

    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Gate Accuracy:          {final_metrics['gate_acc']:.2f}%")
    print(f"  Gated System Accuracy:  {final_metrics['gated_acc']:.2f}%")
    print(f"  Oracle Accuracy:        {final_metrics['oracle_acc']:.2f}%")
    print(f"  Weighted Cost:          {final_metrics['weighted_cost']:.1f}%")
    print(f"  Pareto Score:           {final_metrics['pareto_score']:.2f}")
    print()
    print("  Baselines:")
    print(f"    Always dense:  {static_dense:.2f}% (cost 100.0%)")
    print(f"    Always 1:32:   {static_1_32:.2f}% (cost {COSTS[4].item()*100:.1f}%)")
    print(f"  Improvement over 1:32:  {final_metrics['gated_acc'] - static_1_32:+.2f}%")
    print(f"  Gap to dense:           {static_dense - final_metrics['gated_acc']:.2f}%")
    print(f"  Gap to oracle:          {oracle_acc - final_metrics['gated_acc']:.2f}%")
    print("  Mode Distribution:")
    for m in MODES:
        print(f"    {m:<8} {final_metrics['mode_distribution'][m]:.1f}%")
    print("  Per-Mode Gated Accuracy:")
    for m in MODES:
        print(f"    {m:<8} {final_metrics['per_mode_gated_acc'][m]:.1f}%")
    print()
    print(f"  Results saved →  {results_path}")
    print(f"  Checkpoint saved → {ckpt_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
