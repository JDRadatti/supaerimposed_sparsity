"""
gate.py - Train a binary gate to predict whether an input should run
in sparse (1:32) or dense mode.

Uses baseline dense + sparse models to generate labels:
  Label 1 (needs dense): dense correct AND sparse incorrect
  Label 0 (sparse OK):  everything else

Gate input: 2048-dim feature vector from dense model (pre-avgpool, 512x2x2 flattened).
Gate: Linear(2048,256) -> ReLU -> Linear(256,1) trained with BCEWithLogitsLoss.
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

from vgg import build_model, VGG16
from baseline import prune_nm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs_gate"
DENSE_CKPT = "./baseline/dense.pt"
SPARSE_CKPT = "./baseline/sparse_1_32.pt"
GATE_INPUT_DIM = 2048
GATE_HIDDEN_DIM = 256
GATE_EPOCHS = 20
GATE_LR = 0.01
GATE_BATCH_SIZE = 512

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class FeatureExtractor(nn.Module):
    _POOL4_IDX = 33  # 4th MaxPool2d in VGG16 features

    def __init__(self, model: VGG16):
        super().__init__()
        self.backbone = nn.Sequential(*list(model.features.children())[:self._POOL4_IDX + 1])

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x


class GateModel(nn.Module):
    def __init__(self, input_dim=GATE_INPUT_DIM, hidden_dim=GATE_HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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
def extract_features_and_predictions(dense_model, sparse_model, loader):
    extractor = FeatureExtractor(dense_model).to(DEVICE)
    extractor.eval()
    dense_model.eval()
    sparse_model.eval()

    all_features = []
    all_dense_preds = []
    all_sparse_preds = []
    all_targets = []

    for images, targets_batch in loader:
        images = images.to(DEVICE)

        feats = extractor(images)
        all_features.append(feats.cpu())

        dense_out = dense_model(images)
        all_dense_preds.append(dense_out.argmax(1).cpu())

        sparse_out = sparse_model(images)
        all_sparse_preds.append(sparse_out.argmax(1).cpu())

        all_targets.append(targets_batch)

    return (
        torch.cat(all_features, dim=0),
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


def train_gate(gate, train_features, train_labels, val_features, val_labels):
    gate = gate.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(gate.parameters(), lr=GATE_LR, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=GATE_EPOCHS)

    train_dataset = TensorDataset(train_features, train_labels.float())
    n_pos = train_labels.sum().item()
    n_neg = len(train_labels) - n_pos
    sample_weights = torch.where(train_labels == 1, n_neg / n_pos, 1.0)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=GATE_BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)

    val_features_dev = val_features.to(DEVICE)
    val_labels_bool = val_labels.bool()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, GATE_EPOCHS + 1):
        gate.train()
        epoch_loss = 0.0
        n_samples = 0
        for batch_feats, batch_labels in train_loader:
            batch_feats = batch_feats.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad()
            logits = gate(batch_feats).squeeze(1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_feats.size(0)
            n_samples += batch_feats.size(0)

        scheduler.step()
        epoch_loss /= n_samples

        gate.eval()
        with torch.no_grad():
            val_logits = gate(val_features_dev).squeeze(1).cpu()
            val_preds = val_logits > 0

        val_acc = (val_preds == val_labels_bool).float().mean().item() * 100

        label_0_mask = val_labels == 0
        label_1_mask = val_labels == 1
        acc_0 = (val_preds[label_0_mask] == 0).float().mean().item() * 100 if label_0_mask.any() else 0.0
        acc_1 = (val_preds[label_1_mask] == 1).float().mean().item() * 100 if label_1_mask.any() else 0.0

        n_gate_dense = val_preds.sum().item()
        log.info(f"Epoch {epoch:2d}/{GATE_EPOCHS}: Loss={epoch_loss:.4f} | "
                 f"ValAcc={val_acc:.2f}% | Acc0={acc_0:.1f}% | Acc1={acc_1:.1f}% | "
                 f"GateDense={n_gate_dense}/{len(val_preds)}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in gate.state_dict().items()}

    gate.load_state_dict(best_state)
    log.info(f"Best val accuracy: {best_val_acc:.2f}%")
    return gate, best_val_acc


def evaluate_full(gate, val_features, val_labels, dense_preds, sparse_preds, targets):
    gate = gate.to(DEVICE).eval()

    with torch.no_grad():
        gate_logits = gate(val_features.to(DEVICE)).squeeze(1).cpu()

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
    print(f"  Gate: {GATE_INPUT_DIM} -> {GATE_HIDDEN_DIM} -> 1 (BCEWithLogitsLoss)")
    print(f"  Features: pre-avgpool ({GATE_INPUT_DIM}-dim)")
    print(f"  Sampling: class-balanced (WeightedRandomSampler)")
    print(f"  Epochs: {GATE_EPOCHS}, LR: {GATE_LR}")
    print(f"  Label: 1 = dense correct AND sparse incorrect")
    print(f"          0 = everything else")
    print()

    dense_model = load_dense_model()
    sparse_model = load_sparse_model()

    log.info("Creating data loaders (no augmentation for feature extraction)...")
    train_loader, val_loader = create_eval_loaders("./data", 128, 4)

    log.info("Generating training dataset...")
    train_features, train_dense_preds, train_sparse_preds, train_targets = \
        extract_features_and_predictions(dense_model, sparse_model, train_loader)
    train_labels = generate_labels(train_dense_preds, train_sparse_preds, train_targets)

    log.info("Generating validation dataset...")
    val_features, val_dense_preds, val_sparse_preds, val_targets = \
        extract_features_and_predictions(dense_model, sparse_model, val_loader)
    val_labels = generate_labels(val_dense_preds, val_sparse_preds, val_targets)

    log.info("Training gate model...")
    gate = GateModel()
    gate, best_val_acc = train_gate(gate, train_features, train_labels, val_features, val_labels)

    ckpt_path = f"{BASE_DIR}/gate_checkpoint.pt"
    torch.save({
        "gate_state_dict": gate.state_dict(),
        "best_val_acc": best_val_acc,
        "input_dim": GATE_INPUT_DIM,
        "hidden_dim": GATE_HIDDEN_DIM,
    }, ckpt_path)
    log.info(f"Gate checkpoint saved to {ckpt_path}")

    results = evaluate_full(gate, val_features, val_labels, val_dense_preds, val_sparse_preds, val_targets)

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