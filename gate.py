import contextlib
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sparsity_modes.masks import set_active_sparsity
from sparsity_modes.bn_buffers import restore_bn_stats


def _quiet_restore_bn(model, key):
    with contextlib.redirect_stdout(io.StringIO()):
        restore_bn_stats(model, key)


class GateModel(nn.Module):
    def __init__(self, hidden_dim=0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        input_dim = 64 * 16 * 16  # 16384
        if hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        if x.dim() == 4 and x.size(-1) >= 32:
            x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


@torch.no_grad()
def generate_gate_data(model, loader, device):
    """Generate (features, binary_label) dataset for gate training.

    Features are extracted from the output of the first conv+bn+relu
    (never masked), stored as [B, 64, 32, 32]. The gate's own
    MaxPool2d and flatten handle the rest.

    Labels are 1 if 1:32 top-1 matches ground truth, 0 otherwise.
    """
    model.eval()
    set_active_sparsity(model, "1:32")
    _quiet_restore_bn(model, "1:32")

    all_features, all_labels = [], []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        feat = model.features[:3](images)

        outputs = model(images)
        labels = (outputs.argmax(1) == targets).float()

        all_features.append(feat.cpu())
        all_labels.append(labels.cpu())

    return TensorDataset(torch.cat(all_features), torch.cat(all_labels))


def train_gate(gate, train_loader, val_loader, device, epochs=20, lr=1e-3):
    gate = gate.to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        gate.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = gate(features).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
        train_loss /= len(train_loader.dataset)

        gate.eval()
        val_loss, val_acc = _eval_gate(gate, val_loader, device, criterion)
        print(f"  Epoch {epoch:2d}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = gate.state_dict().copy()

    if best_state is not None:
        gate.load_state_dict(best_state)
    return gate


def _eval_gate(gate, loader, device, criterion=None):
    gate.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = gate(features).squeeze(-1)
            if criterion is not None:
                total_loss += criterion(logits, labels).item() * features.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total if criterion is not None else 0.0
    return avg_loss, 100.0 * correct / total


@torch.no_grad()
def evaluate_gated(model, gate, loader, device, threshold=0.5):
    """Evaluate the gated inference system.

    For each batch: extract features, run gate to decide 1:32 vs dense,
    then evaluate each sub-batch under its assigned mode with correct BN stats.

    Returns:
        mode_counts:   dict {mode: num_samples_routed}
        mode_correct:  dict {mode: num_correct_predictions}
        gate_stats:    dict with gate's own accuracy on this evaluation set
    """
    model.eval()
    gate.eval()

    mode_counts = {"1:32": 0, "dense": 0}
    mode_correct = {"1:32": 0, "dense": 0}
    gate_correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        feat = model.features[:3](images)

        logits = gate(feat).squeeze(-1)
        probs = torch.sigmoid(logits)
        use_1_32 = probs >= threshold

        # 1:32 sub-batch
        if use_1_32.any():
            idx = use_1_32.nonzero().squeeze(1)
            set_active_sparsity(model, "1:32")
            _quiet_restore_bn(model, "1:32")
            out = model(images[idx])
            correct = (out.argmax(1) == targets[idx])
            mode_correct["1:32"] += correct.sum().item()
            mode_counts["1:32"] += idx.size(0)
            gate_correct += correct.sum().item()

        # dense sub-batch
        dense_mask = ~use_1_32
        if dense_mask.any():
            idx = dense_mask.nonzero().squeeze(1)
            set_active_sparsity(model, "dense")
            _quiet_restore_bn(model, "dense")
            out = model(images[idx])
            correct = (out.argmax(1) == targets[idx])
            mode_correct["dense"] += correct.sum().item()
            mode_counts["dense"] += idx.size(0)
            gate_correct += correct.sum().item()

        total += images.size(0)

    gate_stats = {
        "accuracy": 100.0 * gate_correct / total,
    }
    return mode_counts, mode_correct, gate_stats
