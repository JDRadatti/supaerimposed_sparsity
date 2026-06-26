"""
gate.py - Train a binary gate to predict whether an input should run
in sparse (1:32) or dense mode.

Uses baseline dense + sparse models to generate labels:
  Label 1 (needs dense): dense correct AND sparse incorrect
  Label 0 (sparse OK):  everything else

Gate is a CNN trained directly on raw 3x32x32 images:
  configurable conv blocks -> GAP -> Dropout -> Linear(n,1).
Trained with BCEWithLogitsLoss (+ optional loss variants).
"""

import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import datasets, transforms

from vgg import build_model
from baseline import prune_nm
from gate_losses import compute_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs_gate"
DENSE_CKPT = "./runs_baseline/dense.pt"
SPARSE_CKPT = "./runs_baseline/sparse_1_32.pt"
GATE_CHANNELS = [32, 64, 128, 256, 512]
GATE_EPOCHS = 100
GATE_LR = 1e-3
GATE_BATCH_SIZE = 128
GATE_WARMUP_EPOCHS = 5
GATE_WEIGHT_DECAY = 1e-4

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class AugmentedTensorDataset(TensorDataset):
    def __init__(self, images, labels, train=True):
        super().__init__(images, labels)
        if train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = None

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        if self.transform:
            img = self.transform(img)
        return img, label


class GateModel(nn.Module):
    def __init__(self, in_channels=3, channels=None):
        super().__init__()
        if channels is None:
            channels = GATE_CHANNELS

        layers = []
        c_in = in_channels
        for c_out in channels[:-1]:
            layers.extend([
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            c_in = c_out

        last = channels[-1]
        layers.extend([
            nn.Conv2d(c_in, last, 3, padding=1),
            nn.BatchNorm2d(last),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        ])

        self.features = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(last, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.fc(x)


def load_dense_model():
    log.info("Loading dense baseline model...")
    model = build_model().to(DEVICE)
    state = torch.load(DENSE_CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    log.info(f"  Dense model loaded (acc={state.get('acc', 'N/A')})")
    return model


def load_sparse_model():
    log.info("Loading sparse 1:32 baseline model...")
    model = build_model().to(DEVICE)
    dense_state = torch.load(DENSE_CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(dense_state["model_state_dict"])
    prune_nm(model, N=1, M=32, conv_axis="out", linear_orientation="row")
    sparse_state = torch.load(SPARSE_CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(sparse_state["model_state_dict"])
    model.eval()
    log.info(f"  Sparse model loaded (acc={sparse_state.get('acc', 'N/A')})")
    return model


def create_eval_loaders(data_dir, batch_size, num_workers):
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


@torch.no_grad()
def collect_predictions(dense_model, sparse_model, loader):
    dense_model.eval()
    sparse_model.eval()

    all_images = []
    all_dense_preds = []
    all_sparse_preds = []
    all_targets = []

    for images, targets_batch in loader:
        all_images.append(images.cpu())

        dense_out = dense_model(images.to(DEVICE))
        all_dense_preds.append(dense_out.argmax(1).cpu())

        sparse_out = sparse_model(images.to(DEVICE))
        all_sparse_preds.append(sparse_out.argmax(1).cpu())

        all_targets.append(targets_batch)

    return (
        torch.cat(all_images, dim=0),
        torch.cat(all_dense_preds, dim=0),
        torch.cat(all_sparse_preds, dim=0),
        torch.cat(all_targets, dim=0),
    )


def generate_labels(dense_preds, sparse_preds, targets):
    dense_correct = dense_preds == targets
    sparse_correct = sparse_preds == targets

    needs_dense = dense_correct & ~sparse_correct
    labels = needs_dense.long()

    n_needs_dense = labels.sum().item()
    n_sparse_ok = len(labels) - n_needs_dense
    n_both_correct = (dense_correct & sparse_correct).sum().item()
    n_dense_only = needs_dense.sum().item()
    n_sparse_only = (sparse_correct & ~dense_correct).sum().item()
    n_both_wrong = (~dense_correct & ~sparse_correct).sum().item()

    log.info(f"  Label distribution ({len(labels)} samples):")
    log.info(f"    sparse-OK (label 0): {n_sparse_ok} ({100*n_sparse_ok/len(labels):.1f}%)")
    log.info(f"    needs-dense (label 1): {n_needs_dense} ({100*n_needs_dense/len(labels):.1f}%)")
    log.info(f"  Breakdown:")
    log.info(f"    Both models correct:  {n_both_correct}")
    log.info(f"    Dense only correct:   {n_dense_only}")
    log.info(f"    Sparse only correct:  {n_sparse_only}")
    log.info(f"    Both wrong:           {n_both_wrong}")

    return labels


def train_gate(gate, train_images, train_labels,
               val_images, val_labels,
               val_dense_preds, val_sparse_preds, val_targets,
               loss_mode="baseline", log_path=None, **loss_kwargs):
    gate = gate.to(DEVICE)

    optimizer = torch.optim.AdamW(gate.parameters(), lr=GATE_LR, weight_decay=GATE_WEIGHT_DECAY)

    warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=GATE_WARMUP_EPOCHS)
    cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=GATE_EPOCHS - GATE_WARMUP_EPOCHS)
    scheduler = lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[GATE_WARMUP_EPOCHS])

    train_dataset = AugmentedTensorDataset(train_images, train_labels.float(), train=True)
    n_pos = train_labels.sum().item()
    n_neg = len(train_labels) - n_pos
    sample_weights = torch.where(train_labels == 1, n_neg / n_pos, 1.0)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=GATE_BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)

    val_images_dev = val_images.to(DEVICE)
    val_labels_bool = val_labels.bool()
    val_dense_preds_dev = val_dense_preds.to(DEVICE)
    val_sparse_preds_dev = val_sparse_preds.to(DEVICE)
    val_targets_dev = val_targets.to(DEVICE)

    best_gated_acc = 0.0
    best_state = None

    log_lines = []
    header = "epoch,total_loss,bce_loss,aux_loss,val_acc,gate_acc0,gate_acc1,gated_acc,frac_routed_dense"
    log_lines.append(header)

    for epoch in range(1, GATE_EPOCHS + 1):
        gate.train()
        epoch_total = 0.0
        epoch_bce = 0.0
        epoch_aux = 0.0
        n_samples = 0
        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad()
            logits = gate(batch_imgs).squeeze(1)
            result = compute_loss(logits, batch_labels, loss_mode=loss_mode, **loss_kwargs)
            result["loss"].backward()
            optimizer.step()
            epoch_total += result["loss"].item() * batch_imgs.size(0)
            epoch_bce += result["bce"] * batch_imgs.size(0)
            epoch_aux += result["aux"] * batch_imgs.size(0)
            n_samples += batch_imgs.size(0)

        scheduler.step()
        epoch_total /= n_samples
        epoch_bce /= n_samples
        epoch_aux /= n_samples

        gate.eval()
        with torch.no_grad():
            val_logits = gate(val_images_dev).squeeze(1).cpu()
            val_preds = val_logits > 0

        val_acc = (val_preds == val_labels_bool).float().mean().item() * 100

        mask0 = val_labels == 0
        mask1 = val_labels == 1
        acc_0 = (val_preds[mask0] == 0).float().mean().item() * 100 if mask0.any() else 0.0
        acc_1 = (val_preds[mask1] == 1).float().mean().item() * 100 if mask1.any() else 0.0

        gate_sparse = ~val_preds
        gated_preds = torch.where(gate_sparse, val_sparse_preds, val_dense_preds)
        gated_acc = (gated_preds == val_targets).float().mean().item() * 100

        frac_dense = val_preds.float().mean().item() * 100
        n_gate_dense = val_preds.sum().item()

        log.info(f"Epoch {epoch:2d}/{GATE_EPOCHS}: Loss={epoch_total:.4f} (BCE={epoch_bce:.4f} Aux={epoch_aux:.4f}) | "
                 f"GateAcc={val_acc:.2f}% | Acc0={acc_0:.1f}% | Acc1={acc_1:.1f}% | "
                 f"GatedAcc={gated_acc:.2f}% | "
                 f"GateDense={n_gate_dense}/{len(val_preds)} ({frac_dense:.1f}%)")

        log_lines.append(f"{epoch},{epoch_total:.6f},{epoch_bce:.6f},{epoch_aux:.6f},"
                         f"{val_acc:.2f},{acc_0:.2f},{acc_1:.2f},{gated_acc:.2f},{frac_dense:.1f}")

        if gated_acc > best_gated_acc:
            best_gated_acc = gated_acc
            best_state = {k: v.cpu().clone() for k, v in gate.state_dict().items()}

    gate.load_state_dict(best_state)
    log.info(f"Best gated accuracy: {best_gated_acc:.2f}%")

    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        log.info(f"Training log saved to {log_path}")

    return gate, best_gated_acc


def evaluate_full(gate, val_images, val_labels, dense_preds, sparse_preds, targets):
    gate = gate.to(DEVICE).eval()

    with torch.no_grad():
        gate_logits = gate(val_images.to(DEVICE)).squeeze(1).cpu()

    gate_preds = (gate_logits > 0).long()

    dense_correct = dense_preds == targets
    sparse_correct = sparse_preds == targets
    gate_sparse = gate_preds == 0
    gate_dense = gate_preds == 1

    gate_binary_acc = (gate_preds == val_labels).float().mean().item() * 100

    n1 = val_labels == 1
    n0 = val_labels == 0
    tp = (gate_dense & n1).sum().float()
    fp = (gate_dense & n0).sum().float()
    fn = (gate_sparse & n1).sum().float()
    precision_1 = (tp / (tp + fp)).item() * 100 if (tp + fp) > 0 else 0.0
    recall_1 = (tp / (tp + fn)).item() * 100 if (tp + fn) > 0 else 0.0

    categories = [
        ("predict 1:32, 1:32 correct,   dense correct",   gate_sparse & sparse_correct & dense_correct),
        ("predict 1:32, 1:32 correct,   dense incorrect",  gate_sparse & sparse_correct & ~dense_correct),
        ("predict 1:32, 1:32 incorrect, dense correct",   gate_sparse & ~sparse_correct & dense_correct),
        ("predict 1:32, 1:32 incorrect, dense incorrect", gate_sparse & ~sparse_correct & ~dense_correct),
        ("predict dense, 1:32 correct,   dense correct",  gate_dense & sparse_correct & dense_correct),
        ("predict dense, 1:32 correct,   dense incorrect", gate_dense & sparse_correct & ~dense_correct),
        ("predict dense, 1:32 incorrect, dense correct",   gate_dense & ~sparse_correct & dense_correct),
        ("predict dense, 1:32 incorrect, dense incorrect", gate_dense & ~sparse_correct & ~dense_correct),
    ]

    gated_preds = torch.where(gate_sparse, sparse_preds, dense_preds)
    gated_acc = (gated_preds == targets).float().mean().item() * 100

    dense_only_acc = dense_correct.float().mean().item() * 100
    sparse_only_acc = sparse_correct.float().mean().item() * 100

    n_gate_sparse = gate_sparse.sum().item()
    n_gate_dense = gate_dense.sum().item()
    total = len(gate_preds)

    lines = []
    lines.append("=" * 65)
    lines.append("  GATE RESULTS")
    lines.append("=" * 65)
    lines.append("")
    lines.append("Gate Binary Classification:")
    lines.append(f"  Accuracy:     {gate_binary_acc:.2f}%")
    lines.append(f"  Precision (needs-dense): {precision_1:.2f}%")
    lines.append(f"  Recall (needs-dense):    {recall_1:.2f}%")
    lines.append(f"  Label distribution: {n0.sum()} sparse-OK, {n1.sum()} needs-dense")
    lines.append("")
    lines.append("Full 2x2x2 Breakdown (gate prediction x sparse x dense):")
    for name, mask in categories:
        lines.append(f"  {name}: {mask.sum().item()}")
    lines.append("")
    lines.append("System Accuracy:")
    lines.append(f"  Dense only:  {dense_only_acc:.2f}%")
    lines.append(f"  Sparse only: {sparse_only_acc:.2f}%")
    lines.append(f"  Gated:       {gated_acc:.2f}%")
    lines.append(f"  Improvement over sparse: {gated_acc - sparse_only_acc:+.2f}%")
    lines.append(f"  Gap to dense:            {dense_only_acc - gated_acc:+.2f}%")
    lines.append("")
    lines.append("Gate Mode Selection Distribution:")
    lines.append(f"  1:32:  {n_gate_sparse} ({100*n_gate_sparse/total:.1f}%)")
    lines.append(f"  Dense: {n_gate_dense} ({100*n_gate_dense/total:.1f}%)")
    lines.append("")
    lines.append("=" * 65)

    result_text = "\n".join(lines)
    print(result_text)

    return {
        "gate_acc": gate_binary_acc,
        "gated_acc": gated_acc,
        "dense_only_acc": dense_only_acc,
        "sparse_only_acc": sparse_only_acc,
        "precision": precision_1,
        "recall": recall_1,
        "categories": {name: mask.sum().item() for name, mask in categories},
        "mode_distribution": {"1:32": n_gate_sparse, "dense": n_gate_dense},
    }


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    print("=" * 65)
    print("  GATE TRAINING: Predict sparse-OK vs needs-dense")
    print("=" * 65)
    print(f"  Device: {DEVICE}")
    print(f"  Gate: CNN 3->{'->'.join(str(c) for c in GATE_CHANNELS)} -> GAP -> Linear({GATE_CHANNELS[-1]},1)")
    print(f"  Input: raw 3x32x32 images with augmentation")
    print(f"  Optimizer: AdamW (lr={GATE_LR}, wd={GATE_WEIGHT_DECAY})")
    print(f"  Scheduler: {GATE_WARMUP_EPOCHS}ep warmup + cosine to {GATE_EPOCHS}ep")
    print(f"  Label: 1 = dense correct AND sparse incorrect")
    print(f"          0 = everything else")
    print()

    dense_model = load_dense_model()
    sparse_model = load_sparse_model()

    log.info("Creating data loaders (no augmentation for label generation)...")
    train_loader, val_loader = create_eval_loaders("./data", 128, 4)

    log.info("Generating labels for training set...")
    train_images, train_dense_preds, train_sparse_preds, train_targets = \
        collect_predictions(dense_model, sparse_model, train_loader)
    train_labels = generate_labels(train_dense_preds, train_sparse_preds, train_targets)

    log.info("Generating labels for validation set...")
    val_images, val_dense_preds, val_sparse_preds, val_targets = \
        collect_predictions(dense_model, sparse_model, val_loader)
    val_labels = generate_labels(val_dense_preds, val_sparse_preds, val_targets)

    log.info("Training gate model (with augmentation on raw images)...")
    gate = GateModel()
    gate, best_gated_acc = train_gate(
        gate, train_images, train_labels,
        val_images, val_labels,
        val_dense_preds, val_sparse_preds, val_targets,
    )

    ckpt_path = f"{BASE_DIR}/gate_checkpoint.pt"
    torch.save({
        "gate_state_dict": gate.state_dict(),
        "best_gated_acc": best_gated_acc,
    }, ckpt_path)
    log.info(f"Gate checkpoint saved to {ckpt_path}")

    results = evaluate_full(gate, val_images, val_labels, val_dense_preds, val_sparse_preds, val_targets)

    results_path = f"{BASE_DIR}/gate_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Gate Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Gate Binary Accuracy: {results['gate_acc']:.2f}%\n")
        f.write(f"Precision (needs-dense): {results['precision']:.2f}%\n")
        f.write(f"Recall (needs-dense): {results['recall']:.2f}%\n\n")
        f.write(f"Gated System Accuracy: {results['gated_acc']:.2f}%\n")
        f.write(f"Dense Only Accuracy: {results['dense_only_acc']:.2f}%\n")
        f.write(f"Sparse Only Accuracy: {results['sparse_only_acc']:.2f}%\n")
        f.write(f"Improvement over sparse: {results['gated_acc'] - results['sparse_only_acc']:+.2f}%\n")
        f.write(f"Gap to dense: {results['dense_only_acc'] - results['gated_acc']:+.2f}%\n\n")
        f.write(f"2x2x2 Breakdown:\n")
        for name, count in results['categories'].items():
            f.write(f"  {name}: {count}\n")
        f.write(f"\nMode Distribution:\n")
        f.write(f"  1:32: {results['mode_distribution']['1:32']}\n")
        f.write(f"  Dense: {results['mode_distribution']['dense']}\n")
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
