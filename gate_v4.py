"""
gate_v4.py — Budget-Constrained Gumbel-Softmax Gate with Shallow Stem

Trains a gate to route each CIFAR-10 sample to the cheapest N:M sparsity
mode the frozen backbone classifies it correctly, while explicitly
penalising expected compute cost, maximising entropy, and balancing
mode usage to prevent gate collapse.

Key differences from V3:
  - Stem = features[:7] (block 1 only, 12.5% of FLOPs) instead of [:24]
  - Conv-based gate (64→32→64→128) instead of pure MLP
  - Gumbel-Softmax during training with temperature annealing
  - Multi-component loss: CE + cost + entropy + load balancing
  - Split-aware costs that account for always-dense stem + gate overhead
  - Pareto checkpoint selection with minimum-epoch guard
  - Removed class weights (caused V3's gate-accuracy collapse)
"""
import math
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
BASE_DIR = "./runs_gate_v4"
CHECKPOINT_PATH = "./runs/progressive_sparse_dense_final.pt"
DATA_DIR = "./data"

MODES = ["dense", "1:4", "1:8", "1:16", "1:32"]
NUM_MODES = len(MODES)

# --- Split-aware costs ---
# Stem (features[:7], always dense):         39.52M MACs
# Gate (small conv net):                      2.65M MACs
# Suffix masked convs + Lin(512,512):        273.94M MACs (at dense sparsity)
# Last linear (always dense):                 0.01M MACs
# Total dense:                               316.12M MACs
#
# Per-mode fractions:
#   dense: (39.52 + 2.65 + 273.94 + 0.01) / 316.12 = 1.0000
#   1:4:   (39.52 + 2.65 + 68.49  + 0.01) / 316.12 = 0.3501
#   1:8:   (39.52 + 2.65 + 34.24  + 0.01) / 316.12 = 0.2417
#   1:16:  (39.52 + 2.65 + 17.12  + 0.01) / 316.12 = 0.1876
#   1:32:  (39.52 + 2.65 +  8.56  + 0.01) / 316.12 = 0.1605
COSTS = torch.tensor([1.0000, 0.3501, 0.2417, 0.1876, 0.1605])

STEM_SLICE = slice(None, 7)    # backbone.features[:7]  → block 1 → [B, 64, 16, 16]
SUFFIX_SLICE = slice(7, None)  # backbone.features[7:]  → blocks 2–5

# --- Training schedule ---
GATE_EPOCHS = 30

# Phase 1 (epochs 0–9): exploration
WARMUP_EPOCHS = 10
TAU_START = 5.0                             # Gumbel temperature
TAU_MID = 1.0                               # τ at end of phase 1
TAU_END = 0.1                               # τ at end of training
COST_LAMBDA_START = 0.0                     # λ_cost phase 1 start
COST_LAMBDA_MID = 0.1                       # λ_cost at end of phase 1
COST_LAMBDA_END = 0.3                       # λ_cost at end of training
ENT_LAMBDA_START = 0.1                      # λ_ent phase 1 start
ENT_LAMBDA_MID = 0.05                       # λ_ent at end of phase 1
ENT_LAMBDA_END = 0.005                      # λ_ent at end of training
BALANCE_LAMBDA = 0.01                       # λ_bal (fixed)
MIN_CKPT_EPOCH = 10                         # minimum epoch before saving best
PARETO_BETA = 0.1                           # accuracy-vs-cost tradeoff in checkpoint score

GATE_LR = 1e-3
GATE_BATCH_SIZE = 128
GATE_WARMUP = 3
GATE_WEIGHT_DECAY = 1e-4


# ---------------------------------------------------------------------------
# Gate architecture
# ---------------------------------------------------------------------------


class DynamicGate(nn.Module):
    """Small conv gate: block-1 stem features → 5 sparsity mode logits (~112K params).

    Input:  [B, 64, 16, 16]  (stem output)
    Output: [B, 5]  logits
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------


def _linear_schedule(t, start, end):
    return start + (end - start) * t


def _cosine_schedule(t, start, end):
    return end + (start - end) * (1 + math.cos(math.pi * t)) / 2


def get_schedule_params(epoch: int):
    """Compute annealed τ, λ_cost, λ_ent for the current epoch.

    Phase 1: epochs 0 .. WARMUP_EPOCHS-1 (linear)
    Phase 2: epochs WARMUP_EPOCHS .. GATE_EPOCHS-1 (linear for τ, cosine for λ's)
    """
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
# Stem / suffix helpers
# ---------------------------------------------------------------------------


def extract_stem(backbone, images, device):
    set_active_sparsity(backbone, "dense")
    restore_bn_stats(backbone, "dense", verbose=False)
    return backbone.features[STEM_SLICE](images)


def run_suffix(backbone, stem_features):
    x = backbone.features[SUFFIX_SLICE](stem_features)
    x = backbone.avgpool(x)
    x = torch.flatten(x, 1)
    return backbone.classifier(x)


# ---------------------------------------------------------------------------
# Per-mode output computation
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


# ---------------------------------------------------------------------------
# Frontier label generation (SGS)
# ---------------------------------------------------------------------------


def generate_frontier_labels(correct_per_mode: torch.Tensor) -> torch.Tensor:
    B = correct_per_mode.size(0)
    labels = torch.zeros(B, dtype=torch.long, device=correct_per_mode.device)

    for mode_idx in range(NUM_MODES - 1, 0, -1):  # 4, 3, 2, 1
        mask = correct_per_mode[:, mode_idx] & (labels == 0)
        labels[mask] = mode_idx

    return labels


# ---------------------------------------------------------------------------
# Multi-component loss
# ---------------------------------------------------------------------------


def compute_gate_loss(logits, frontier_labels, costs, schedule_params):
    """
    L = L_ce + λ_cost * L_cost - λ_ent * H(probs) + λ_bal * L_balance

    Returns dict of scalar tensors and float components for logging.
    """
    B = logits.size(0)
    device = logits.device
    costs = costs.to(device)

    # Gumbel-Softmax sample (exploration + load balancing)
    tau = schedule_params["tau"]
    gumbel = F.gumbel_softmax(logits, tau=tau, hard=True, dim=1)  # [B, 5] one-hot

    probs = F.softmax(logits, dim=1)  # [B, 5]

    # CE loss on raw logits
    ce = F.cross_entropy(logits, frontier_labels)

    # Expected computational cost
    L_cost = (probs * costs).sum(dim=1).mean()

    # Entropy maximisation (negative entropy → smaller loss = more exploration)
    L_entropy = (probs * (probs + 1e-8).log()).sum(dim=1).mean()  # -H average

    # Load balancing (Switch Transformer style)
    f = gumbel.float().mean(dim=0)       # [5]  fraction assigned per mode (hard)
    p = probs.mean(dim=0)                # [5]  mean softmax probability per mode
    L_balance = (f * p).sum()

    lambda_cost = schedule_params["lambda_cost"]
    lambda_ent = schedule_params["lambda_ent"]

    loss = ce + lambda_cost * L_cost + lambda_ent * L_entropy + BALANCE_LAMBDA * L_balance

    with torch.no_grad():
        H = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()  # mean entropy (≥ 0)

    return {
        "loss": loss,
        "ce": ce.item(),
        "cost_term": L_cost.item(),
        "entropy_mean": H.item(),
        "balance_term": L_balance.item(),
    }


# ---------------------------------------------------------------------------
# Evaluation
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
    oracle_acc = oracle_correct / total * 100
    return per_mode_acc, oracle_acc


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
                backbone, images, targets, device
            )
            frontier_labels = generate_frontier_labels(per_mode_correct)

            stem_features = extract_stem(backbone, images, device)
            logits = gate(stem_features)
            gate_choices = logits.argmax(dim=1)

            gate_correct += (gate_choices == frontier_labels).sum().item()

            for i, mode_key in enumerate(MODES):
                mask = gate_choices == i
                n = mask.sum().item()
                if n == 0:
                    continue
                per_mode_total[i] += n
                set_active_sparsity(backbone, mode_key)
                restore_bn_stats(backbone, mode_key, verbose=False)
                outputs = run_suffix(backbone, stem_features[mask])
                gated_preds = outputs.argmax(dim=1)
                per_mode_gated_correct[i] += (gated_preds == targets[mask]).sum().item()
                gated_correct += (gated_preds == targets[mask]).sum().item()

            oracle_correct += per_mode_correct.any(dim=1).sum().item()

            mode_counts += torch.bincount(gate_choices, minlength=NUM_MODES)

            total += B

    gated_acc = gated_correct / total * 100
    weighted_cost = sum(
        (mode_counts[i] / total).item() * COSTS[i].item()
        for i in range(NUM_MODES)
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
            MODES[i]: (mode_counts[i] / total * 100).item()
            for i in range(NUM_MODES)
        },
        "per_mode_gated_acc": per_mode_gated,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_gate(backbone, gate, train_loader, val_loader, epochs=GATE_EPOCHS,
               lr=GATE_LR, device=None):
    if device is None:
        device = DEVICE

    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    gate = gate.to(device)
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

    best_pareto = -float("inf")
    best_state = None
    best_metrics = None
    costs = COSTS.clone()

    csv_fields = [
        "epoch", "loss", "ce", "cost_term", "entropy", "balance",
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
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_cost = 0.0
        epoch_ent = 0.0
        epoch_bal = 0.0
        n_samples = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            stem_features = extract_stem(backbone, images, device)

            _, per_mode_correct = compute_per_mode_outputs(
                backbone, images, targets, device
            )
            frontier_labels = generate_frontier_labels(per_mode_correct)

            logits = gate(stem_features)
            result = compute_gate_loss(logits, frontier_labels, costs, sched)

            optimizer.zero_grad()
            result["loss"].backward()
            optimizer.step()

            b = images.size(0)
            epoch_loss += result["loss"].item() * b
            epoch_ce += result["ce"] * b
            epoch_cost += result["cost_term"] * b
            epoch_ent += result["entropy_mean"] * b
            epoch_bal += result["balance_term"] * b
            n_samples += b

        scheduler.step()
        avg_loss = epoch_loss / n_samples
        avg_ce = epoch_ce / n_samples
        avg_cost = epoch_cost / n_samples
        avg_ent = epoch_ent / n_samples
        avg_bal = epoch_bal / n_samples

        val_metrics = evaluate_gate(backbone, gate, val_loader, device)

        log.info(
            "Epoch %2d/%d: τ=%.2f λc=%.3f λe=%.4f | "
            "Loss=%.4f CE=%.4f Cost=%.4f H=%.4f Bal=%.4f | "
            "GateAcc=%.2f%% GatedAcc=%.2f%% Cost=%.1f%% Pareto=%.2f",
            epoch + 1, epochs,
            sched["tau"], sched["lambda_cost"], sched["lambda_ent"],
            avg_loss, avg_ce, avg_cost, avg_ent, avg_bal,
            val_metrics["gate_acc"], val_metrics["gated_acc"],
            val_metrics["weighted_cost"], val_metrics["pareto_score"],
        )
        dist = val_metrics["mode_distribution"]
        log.info(
            "  Modes: %s",
            " | ".join(f"{m}={dist[m]:.1f}%" for m in MODES),
        )
        per_acc = val_metrics["per_mode_gated_acc"]
        log.info(
            "  Per-mode acc: %s",
            " | ".join(f"{m}={per_acc[m]:.1f}%" for m in MODES),
        )

        csv_row = [
            str(epoch + 1),
            f"{avg_loss:.6f}", f"{avg_ce:.6f}", f"{avg_cost:.6f}",
            f"{avg_ent:.6f}", f"{avg_bal:.6f}",
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
        log.info("No checkpoint saved (none met min-epoch guard); using final model")

    return gate, best_pareto, best_metrics, log_lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    print("=" * 65)
    print("  DYNAMIC GATE v4 — Budget-Constrained Gumbel-Softmax")
    print("=" * 65)
    print(f"  Device:      {DEVICE}")
    print(f"  Backbone:    {CHECKPOINT_PATH}")
    print(f"  Modes:       {MODES}")
    print(f"  Costs:       {[f'{c:.4f}' for c in COSTS.tolist()]}")
    print(f"  Stem slice:  features[:7]  → block 1 → [B, 64, 16, 16]")
    print(f"  Suffix:      features[7:]  → blocks 2–5 + classifier")
    print(f"  Gate:        Conv(64→32→64→128) → GAP → Linear(128,5)")
    print(f"  Loss:        CE + λc·cost − λe·H(p) + λb·balance")
    print(f"  Schedule:    τ: {TAU_START}→{TAU_MID}→{TAU_END}")
    print(f"               λ_cost: {COST_LAMBDA_START}→{COST_LAMBDA_MID}→{COST_LAMBDA_END}")
    print(f"               λ_ent:  {ENT_LAMBDA_START}→{ENT_LAMBDA_MID}→{ENT_LAMBDA_END}")
    print(f"  Ckpt guard:  epoch >= {MIN_CKPT_EPOCH}")
    print(f"  Pareto β:    {PARETO_BETA}")
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

    log.info("Training gate …")
    gate, best_pareto, best_metrics, log_lines = train_gate(
        backbone, gate, train_loader, val_loader,
        epochs=GATE_EPOCHS, lr=GATE_LR, device=DEVICE,
    )

    ckpt_path = os.path.join(BASE_DIR, "gate_checkpoint.pt")
    torch.save(
        {
            "gate_state_dict": gate.state_dict(),
            "best_pareto": best_pareto,
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
        f.write("  DYNAMIC GATE v4 — Budget-Constrained Gumbel-Softmax\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("Configuration:\n")
        f.write(f"  Stem slice:  features[:7] → block 1 → [B, 64, 16, 16]\n")
        f.write(f"  Costs:       {[f'{c:.4f}' for c in COSTS.tolist()]}\n")
        f.write(f"  Gate params: {n_params}\n")
        f.write(f"  Loss:        CE + λc·cost − λe·H(p) + λb·balance\n")
        f.write(f"  Schedule:    τ {TAU_START}→{TAU_MID}→{TAU_END}\n")
        f.write(f"               λc {COST_LAMBDA_START}→{COST_LAMBDA_MID}→{COST_LAMBDA_END}\n")
        f.write(f"               λe {ENT_LAMBDA_START}→{ENT_LAMBDA_MID}→{ENT_LAMBDA_END}\n")
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
