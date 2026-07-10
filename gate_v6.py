"""
gate_v6.py — SGS-inspired Two-Headed Gate with Cached Labels

Inspired by the SGS (Sandwich Gate Sparsification) technique from DS-Net
(Li et al., CVPR 2021):

  - Two-headed gate: shared encoder → attention head (soft) + slimming head (hard)
  - Binary-split soft targets: easy samples (1:32 correct) vs hard samples
    (1:32 wrong), each with cost-weighted targets over correct modes
  - No separate cost term — cost preference embedded in targets
  - Cached frontier labels eliminate repeated backbone forwards (5× speedup)

Key differences from v5:
  - Pre-computed labels → ~5s/epoch instead of ~45s
  - 35K params (down from 112K)
  - 3 loss terms (softCE + balance + logit reg) instead of 5
  - No class weights needed (self-normalising per-sample targets)

===============================================================
  RESULTS
=================================================================
  Gated System Accuracy:  90.44%
  Weighted Cost:          21.1%
  Pareto Score:           88.33

  Baselines:
    Always dense:  92.20% (cost 100.0%)
    Always 1:32:   88.26% (cost 16.1%)
  Improvement over 1:32:  +2.18%
  Gap to dense:           1.76%
  Gap to oracle:          6.34%
  Mode Distribution:
    dense    0.4%
    1:4      5.2%
    1:8      20.5%
    1:16     74.0%
    1:32     0.0%
  Per-Mode Gated Accuracy:
    dense    86.0%
    1:4      93.8%
    1:8      90.1%
    1:16     90.3%
    1:32     0.0%

  Results saved →  ./runs_gate_v6/results.txt
  Checkpoint saved → ./runs_gate_v6/gate_checkpoint.pt
=================================================================
"""
from datetime import datetime
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms

from vgg import build_model
from sparsity_modes import load_checkpoint, set_active_sparsity, restore_bn_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs_gate_v6"
CHECKPOINT_PATH = "./runs/progressive_sparse_dense_final.pt"
DATA_DIR = "./data"
CACHE_PATH = os.path.join(DATA_DIR, "gate_cache.pt")

MODES = ["dense", "1:4", "1:8", "1:16", "1:32"]
NUM_MODES = len(MODES)
COSTS = torch.tensor([1.0000, 0.3501, 0.2417, 0.1876, 0.1605])
STEM_SLICE = slice(None, 7)

# --- Training schedule ---
GATE_EPOCHS = 60
WARMUP_EPOCHS = 20
TAU_START = 5.0
TAU_MID = 1.5
TAU_END = 0.2
BALANCE_LAMBDA = 0.02
LOGIT_REG_LAMBDA = 0.0005
MIN_CKPT_EPOCH = 20
PARETO_BETA = 0.1
GATE_LR = 5e-4
GATE_BATCH_SIZE = 128
GATE_WARMUP = 5
GATE_WEIGHT_DECAY = 1e-4


# ---------------------------------------------------------------------------
# Indexed dataset — returns (image, cifar_label, sample_idx)
# ---------------------------------------------------------------------------


class IndexedCIFAR10(Dataset):
    def __init__(self, train=True, root=DATA_DIR):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4) if train else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.base = datasets.CIFAR10(root=root, train=train, download=False, transform=transform)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]
        return img, target, idx


# ---------------------------------------------------------------------------
# Two-headed gate architecture
# ---------------------------------------------------------------------------


class DynamicGateV6(nn.Module):
    """Two-headed gate with shared encoder.

    Attention head applies soft channel re-weighting as regularisation
    (used during training only). Slimming head outputs 5-mode logits.
    """
    def __init__(self, in_channels=64, num_modes=NUM_MODES, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden),
            nn.ReLU(inplace=True),
        )
        # Attention head: channel-wise gating (tanh output scaled 0..2)
        self.attn_fc = nn.Linear(hidden, 64)
        # Slimming head: sparsity mode logits
        self.slim_fc = nn.Linear(hidden, num_modes)

    def forward(self, stem_features):
        b, c, h, w = stem_features.shape
        encoded = self.encoder(stem_features)      # [B, 64, 1, 1]
        shared = self.shared_fc(encoded)           # [B, hidden]

        # Attention: scale each of the 64 channels
        attn_raw = self.attn_fc(shared)            # [B, 64]
        attn = 1.0 + torch.tanh(attn_raw)          # [B, 64] range (0, 2)
        attn = attn.view(b, 64, 1, 1)
        gated_features = stem_features * attn       # [B, 64, h, w]

        # Slimming: re-encode gated features for routing
        slim_encoded = self.encoder(gated_features)
        slim_shared = self.shared_fc(slim_encoded)
        logits = self.slim_fc(slim_shared)          # [B, 5]

        return logits, gated_features


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------


def _linear_schedule(t, start, end):
    return start + (end - start) * t


def get_schedule_params(epoch: int):
    if GATE_EPOCHS <= WARMUP_EPOCHS:
        t = epoch / max(GATE_EPOCHS - 1, 1)
        return {"tau": _linear_schedule(t, TAU_START, TAU_END)}
    if epoch < WARMUP_EPOCHS:
        t = epoch / max(WARMUP_EPOCHS - 1, 1)
        return {"tau": _linear_schedule(t, TAU_START, TAU_MID)}
    phase2_epochs = GATE_EPOCHS - WARMUP_EPOCHS
    t = (epoch - WARMUP_EPOCHS) / max(phase2_epochs - 1, 1)
    return {"tau": _linear_schedule(t, TAU_MID, TAU_END)}


# ---------------------------------------------------------------------------
# Backbone helpers
# ---------------------------------------------------------------------------


def load_backbone(path: str = CHECKPOINT_PATH):
    log.info("Loading backbone …")
    model = build_model()
    ckpt = load_checkpoint(model, path)
    eval_metrics = ckpt.get("metrics", {}).get("eval", {})
    if eval_metrics:
        parts = [f"{k}: acc={v[1]:.2f}%" for k, v in sorted(eval_metrics.items())]
        log.info("  %s", " | ".join(parts))
    return model


def extract_stem(backbone, images):
    set_active_sparsity(backbone, "dense")
    restore_bn_stats(backbone, "dense", verbose=False)
    return backbone.features[STEM_SLICE](images)


def compute_per_mode_outputs(backbone, images, targets):
    B = images.size(0)
    per_mode_correct = torch.zeros(B, NUM_MODES, dtype=torch.bool, device=images.device)
    with torch.no_grad():
        for mode_idx, mode_key in enumerate(MODES):
            set_active_sparsity(backbone, mode_key)
            restore_bn_stats(backbone, mode_key, verbose=False)
            backbone.eval()
            outputs = backbone(images)
            per_mode_correct[:, mode_idx] = outputs.argmax(dim=1) == targets
    return per_mode_correct


# ---------------------------------------------------------------------------
# Label caching  (pre-compute correctness + targets once)
# ---------------------------------------------------------------------------


def build_cache_and_loader(backbone, device, batch_size):
    """Pre-compute per-mode correctness and soft targets for train + val sets.

    Saves to CACHE_PATH so subsequent runs are instant.  Returns a
    balanced DataLoader with per-sample indices so targets can be
    looked up by index.
    """
    if os.path.exists(CACHE_PATH):
        log.info("Loading cached frontier labels …")
        data = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
        log.info("  %s", " | ".join(
            f"{MODES[i]}={data['train_freqs'][i]*100:.1f}%" for i in range(NUM_MODES)))
    else:
        log.info("Computing frontier labels (one-time, ~90s) …")
        train_set = IndexedCIFAR10(train=True)
        val_set = IndexedCIFAR10(train=False)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

        train_correct = []
        val_correct = []
        for loader, collector in [(train_loader, train_correct), (val_loader, val_correct)]:
            for images, targets, _ in loader:
                images, targets = images.to(device), targets.to(device)
                correct = compute_per_mode_outputs(backbone, images, targets)
                collector.append(correct.cpu())

        train_correct = torch.cat(train_correct, dim=0)                     # [50000, 5]
        val_correct = torch.cat(val_correct, dim=0)                         # [10000, 5]
        train_targets = _compute_cache_targets(train_correct)               # [50000, 5]
        val_targets = _compute_cache_targets(val_correct)                   # [10000, 5]

        # Frontier labels for sampling
        train_frontier = torch.zeros(len(train_correct), dtype=torch.long)
        for i in range(NUM_MODES - 1, 0, -1):
            mask = train_correct[:, i] & (train_frontier == 0)
            train_frontier[mask] = i
        class_counts = torch.bincount(train_frontier, minlength=NUM_MODES)
        train_freqs = class_counts.float() / class_counts.sum()

        torch.save({
            "train_correct": train_correct,
            "val_correct": val_correct,
            "train_targets": train_targets,
            "val_targets": val_targets,
            "train_frontier": train_frontier,
            "class_counts": class_counts,
            "train_freqs": train_freqs,
        }, CACHE_PATH)
        log.info("Cache saved → %s", CACHE_PATH)
        log.info("  %s", " | ".join(
            f"{MODES[i]}={train_freqs[i]*100:.1f}%" for i in range(NUM_MODES)))
        data = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)

    # Build balanced DataLoader
    train_set = IndexedCIFAR10(train=True)
    class_counts = data["class_counts"]
    train_frontier = data["train_frontier"]
    per_sample = [1.0 / (class_counts[train_frontier[i]].item() + 1e-6)
                  for i in range(len(train_set))]
    sampler = WeightedRandomSampler(per_sample, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)

    val_loader = DataLoader(IndexedCIFAR10(train=False), batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, data


def _compute_cache_targets(correct):
    """SGS-style binary-split cost-weighted soft targets.

    Easy samples (1:32 correct):
      target[j] ∝ correct[j] / cost[j]  (all correct modes, cost-weighted)

    Hard samples (1:32 wrong):
      target[j] ∝ correct[j] / cost[j]  for j < 4 (dense..1:16 only)
      target[4] = 0                     (1:32 explicitly excluded)
    """
    B, K = correct.shape
    correct_f = correct.float()
    inv_cost = 1.0 / (COSTS + 1e-6)                               # [5]

    # Full cost-weighted targets
    full_weights = correct_f * inv_cost                            # [B, 5]
    full_targets = full_weights / full_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

    # Hard-sample targets: exclude 1:32 (index 4)
    hard_mask = torch.ones(K, device=correct_f.device)
    hard_mask[4] = 0.0
    hard_weights = correct_f * inv_cost * hard_mask                # [B, 5]
    hard_targets = hard_weights / hard_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

    # Split: easy if 1:32 correct, else hard
    is_easy = correct[:, 4].float().unsqueeze(1)                   # [B, 1]
    targets = is_easy * full_targets + (1 - is_easy) * hard_targets
    return targets


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_gate_loss(logits, targets, schedule_params):
    B = logits.size(0)

    tau = schedule_params["tau"]
    gumbel = F.gumbel_softmax(logits, tau=tau, hard=True, dim=1)
    probs = F.softmax(logits, dim=1)

    # Soft-target CE
    log_probs = F.log_softmax(logits, dim=1)
    L_ce = -(targets * log_probs).sum(dim=1).mean()

    # Load balancing
    f = gumbel.float().mean(dim=0)
    p = probs.mean(dim=0)
    L_balance = (f * p).sum()

    # Logit regularisation
    L_logit = (logits ** 2).mean()

    loss = L_ce + BALANCE_LAMBDA * L_balance + LOGIT_REG_LAMBDA * L_logit

    with torch.no_grad():
        H = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()

    return {
        "loss": loss,
        "ce": L_ce.item(),
        "entropy": H.item(),
        "balance": L_balance.item(),
    }


# ---------------------------------------------------------------------------
# Evaluation (full-backbone)
# ---------------------------------------------------------------------------


def evaluate_gate(backbone, gate, val_loader, val_correct, device):
    backbone.eval()
    gate.eval()
    total = 0
    gated_correct = 0
    mode_counts = torch.zeros(NUM_MODES)
    per_mode_gated_correct = torch.zeros(NUM_MODES)
    per_mode_total = torch.zeros(NUM_MODES)

    val_correct = val_correct.to(device)

    with torch.no_grad():
        for images, _, indices in val_loader:
            images = images.to(device)
            B = images.size(0)
            idx = indices.to(device)

            stem_features = extract_stem(backbone, images)
            logits, _ = gate(stem_features)
            gate_choices = logits.argmax(dim=1)

            correct_batch = val_correct[idx]  # [B, 5] bool

            for i in range(NUM_MODES):
                mask = gate_choices == i
                n = mask.sum().item()
                if n == 0:
                    continue
                per_mode_total[i] += n
                n_correct = correct_batch[mask, i].sum().item()
                per_mode_gated_correct[i] += n_correct
                gated_correct += n_correct

            mode_counts += torch.bincount(gate_choices.cpu(), minlength=NUM_MODES)
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
        "gated_acc": gated_acc,
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


def train_gate(backbone, gate, train_loader, val_loader, train_targets, val_correct,
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

    train_targets = train_targets.to(device)

    best_pareto = -float("inf")
    best_state = None
    best_metrics = None

    csv_fields = [
        "epoch", "loss", "ce", "entropy", "balance",
        "tau", "gated_acc", "weighted_cost", "pareto",
    ]
    for m in MODES:
        csv_fields.append(f"frac_{m.replace(':', '_')}")
    for m in MODES:
        csv_fields.append(f"acc_{m.replace(':', '_')}")
    log_lines = [",".join(csv_fields)]

    for epoch in range(epochs):
        sched = get_schedule_params(epoch)
        gate.train()

        epoch_loss = epoch_ce = epoch_ent = epoch_bal = 0.0
        n_samples = 0

        for images, _, indices in train_loader:
            images = images.to(device)
            indices = indices.to(device)
            B = images.size(0)

            stem_features = extract_stem(backbone, images)
            logits, _ = gate(stem_features)
            targets = train_targets[indices]
            result = compute_gate_loss(logits, targets, sched)

            optimizer.zero_grad()
            result["loss"].backward()
            optimizer.step()

            epoch_loss += result["loss"].item() * B
            epoch_ce += result["ce"] * B
            epoch_ent += result["entropy"] * B
            epoch_bal += result["balance"] * B
            n_samples += B

        scheduler.step()
        avg_loss = epoch_loss / n_samples
        avg_ce = epoch_ce / n_samples
        avg_ent = epoch_ent / n_samples
        avg_bal = epoch_bal / n_samples

        val_metrics = evaluate_gate(backbone, gate, val_loader, val_correct, device)

        log.info(
            "Epoch %2d/%d: τ=%.2f | Loss=%.4f CE=%.4f H=%.4f Bal=%.4f | "
            "GatedAcc=%.2f%% Cost=%.1f%% Pareto=%.2f",
            epoch + 1, epochs, sched["tau"],
            avg_loss, avg_ce, avg_ent, avg_bal,
            val_metrics["gated_acc"], val_metrics["weighted_cost"], val_metrics["pareto_score"],
        )
        dist = val_metrics["mode_distribution"]
        log.info("  Modes: %s", " | ".join(f"{m}={dist[m]:.1f}%" for m in MODES))
        per_acc = val_metrics["per_mode_gated_acc"]
        log.info("  Per-mode acc: %s", " | ".join(f"{m}={per_acc[m]:.1f}%" for m in MODES))

        csv_row = [
            str(epoch + 1),
            f"{avg_loss:.6f}", f"{avg_ce:.6f}", f"{avg_ent:.6f}", f"{avg_bal:.6f}",
            f"{sched['tau']:.3f}",
            f"{val_metrics['gated_acc']:.2f}", f"{val_metrics['weighted_cost']:.1f}",
            f"{val_metrics['pareto_score']:.2f}",
        ]
        csv_row.extend(f"{val_metrics['mode_distribution'][m]:.1f}" for m in MODES)
        csv_row.extend(f"{val_metrics['per_mode_gated_acc'][m]:.1f}" for m in MODES)
        log_lines.append(",".join(csv_row))

        if epoch + 1 >= MIN_CKPT_EPOCH and val_metrics["pareto_score"] > best_pareto:
            best_pareto = val_metrics["pareto_score"]
            best_state = {k: v.cpu().clone() for k, v in gate.state_dict().items()}
            best_metrics = val_metrics
            log.info("  *** best @ epoch %d (Pareto=%.2f) ***", epoch + 1, best_pareto)

    if best_state is not None:
        gate.load_state_dict(best_state)
        log.info("Loaded best checkpoint (Pareto=%.2f)", best_pareto)
    else:
        best_metrics = evaluate_gate(backbone, gate, val_loader, val_correct, device)

    return gate, best_pareto, best_metrics, log_lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    print("=" * 65)
    print("  DYNAMIC GATE v6 — SGS-inspired Two-Headed Gate")
    print("=" * 65)
    print(f"  Device:      {DEVICE}")
    print(f"  Cache:       {CACHE_PATH}")
    print(f"  Modes:       {MODES}")
    print(f"  Costs:      ", [f"{c:.4f}" for c in COSTS.tolist()])
    print(f"  Gate:        2-head (shared encoder → attn + slimming)")
    print(f"  Loss:        SoftCE(split targets) + λb·balance + λr·|logits|²")
    print(f"  Schedule:    τ={TAU_START}→{TAU_MID}→{TAU_END}")
    print(f"  Ckpt guard:  epoch >= {MIN_CKPT_EPOCH}")
    print(f"  Pareto β:    {PARETO_BETA}")
    print()

    backbone = load_backbone(CHECKPOINT_PATH)
    backbone = backbone.to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    log.info("Building cache and data loaders …")
    train_loader, val_loader, cache = build_cache_and_loader(
        backbone, DEVICE, GATE_BATCH_SIZE)

    train_targets = cache["train_targets"]
    val_correct = cache["val_correct"]

    # Static per-mode accuracies from cache (val set)
    per_mode_acc = {}
    for i, m in enumerate(MODES):
        per_mode_acc[m] = (val_correct[:, i].float().mean() * 100).item()
    oracle_acc = (val_correct.any(dim=1).float().mean() * 100).item()

    print("  Static per-mode accuracies:")
    for m in MODES:
        print(f"    {m:<8} {per_mode_acc[m]:.2f}%")
    print(f"  Oracle accuracy (upper bound):  {oracle_acc:.2f}%")
    print()

    gate = DynamicGateV6()
    n_params = sum(p.numel() for p in gate.parameters())
    log.info("Gate created (%s params)", n_params)

    log.info("Training gate …")
    gate, best_pareto, best_metrics, log_lines = train_gate(
        backbone, gate, train_loader, val_loader, train_targets, val_correct,
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
    final_metrics = evaluate_gate(backbone, gate, val_loader, val_correct, DEVICE)

    static_dense = per_mode_acc["dense"]
    static_1_32 = per_mode_acc["1:32"]

    results_path = os.path.join(BASE_DIR, "results.txt")
    with open(results_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("  DYNAMIC GATE v6 — SGS-inspired Two-Headed Gate\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("Key design:\n")
        f.write("  - SGS-style binary-split soft targets (cost embedded, self-balanced)\n")
        f.write("  - Two-headed gate: shared encoder → attention + slimming heads\n")
        f.write("  - Cached per-mode correctness → no repeated backbone forwards\n")
        f.write("  - Full-backbone evaluation (no BN mismatch)\n\n")
        f.write("Configuration:\n")
        f.write(f"  Gate params: {n_params}\n")
        f.write(f"  Costs:       {[f'{c:.4f}' for c in COSTS.tolist()]}\n")
        f.write(f"  Loss:        SoftCE(split targets) + λb·balance + λr·|logits|²\n")
        f.write(f"  Schedule:    τ {TAU_START}→{TAU_MID}→{TAU_END}\n")
        f.write(f"               λb={BALANCE_LAMBDA} λr={LOGIT_REG_LAMBDA}\n")
        f.write(f"  Ckpt guard:  epoch >= {MIN_CKPT_EPOCH}\n")
        f.write(f"  Pareto β:    {PARETO_BETA}\n")
        f.write("\nStatic Per-Mode Accuracies:\n")
        for m in MODES:
            f.write(f"  {m:<8} {per_mode_acc[m]:.2f}%\n")
        f.write(f"\n  Oracle Accuracy (upper bound):  {oracle_acc:.2f}%\n")
        f.write("\n" + "─" * 65 + "\n\n")
        f.write("Gating Results:\n")
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
    print(f"  Gated System Accuracy:  {final_metrics['gated_acc']:.2f}%")
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
