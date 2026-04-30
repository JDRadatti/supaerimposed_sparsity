import time
import random
from pathlib import Path
from typing import Optional, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



# ---------------------------------------------------------------------------
# N:M activation sparsity utilities
# ---------------------------------------------------------------------------

def _validate_nm(N: Optional[int], M: Optional[int]):
    if N is None or M is None:
        return
    if M <= 0 or N < 0:
        raise ValueError("N and M must be positive with 0 <= N <= M.")
    if N > M:
        raise ValueError("N must be <= M.")


def nm_mask_along_dim(x: torch.Tensor, dim: int, N: Optional[int], M: Optional[int]) -> torch.Tensor:
    """
    Create a 0/1 mask (same shape as x) that keeps N largest-by-magnitude
    elements within each contiguous group of size M along `dim`.
    If N==M or N is None or M is None, returns an all-ones mask (no-op).

    For tails where group_len < M, keeps min(N, group_len).

    NOTE: This is generic: `dim` can be batch (0), channel, feature, etc.
    """
    _validate_nm(N, M)
    if N is None or M is None or N == M:
        return torch.ones_like(x, dtype=x.dtype)

    D = x.size(dim)
    if D == 0 or N == 0:
        return torch.zeros_like(x, dtype=x.dtype)

    # Move target dim to the end for easy grouping
    perm = list(range(x.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_t = x.permute(*perm)  # shape: [..., D]

    # Pad last dim to multiple of M (with zeros) so grouping is easy
    pad_elems = (M - (D % M)) % M
    if pad_elems > 0:
        pad_shape = list(x_t.shape)
        pad_shape[-1] = pad_elems
        x_t = torch.cat([x_t, x_t.new_zeros(pad_shape)], dim=-1)

    # Group to (..., G, M)
    D_padded = x_t.size(-1)
    G = D_padded // M
    x_g = x_t.reshape(*x_t.shape[:-1], G, M)

    # Top-k (by magnitude) along the group axis (last dim)
    k = min(N, M)
    _, topk_idx = torch.topk(x_g, k=k, dim=-1, largest=True, sorted=False)

    # Build a mask of zeros then scatter 1s at topk positions
    mask_g = torch.zeros_like(x_g)
    mask_g.scatter_(-1, topk_idx, 1.0)

    # Reshape back and unpad
    mask_t = mask_g.reshape(*mask_g.shape[:-2], D_padded)
    if pad_elems > 0:
        mask_t = mask_t[..., :D]

    # Inverse permute to original layout
    inv_perm = list(range(x.dim()))
    inv_perm[dim], inv_perm[-1] = inv_perm[-1], inv_perm[dim]
    mask = mask_t.permute(*inv_perm)

    return mask


class ActSparseLinear(nn.Module):
    """
    Wrapper around nn.Linear that enforces N:M sparsity on *activations*
    along a configurable dimension (default: feature dim = last dim)
    before forwarding to the underlying Linear layer.

    - If mask_dim == -1 (default): per-sample feature sparsity.
    - If mask_dim == 0: cross-batch sparsity at each feature position.
    - If N == M (or N/M is None): masking is a no-op.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        N: Optional[int] = None,
        M: Optional[int] = None,
        mask_dim: int = -1,
    ):
        super().__init__()
        _validate_nm(N, M)
        self.N = N
        self.M = M
        self.mask_dim = mask_dim
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x shape: (B, ..., in_features)
        if self.N is None or self.M is None or self.N == self.M:
            x_masked = x
        else:
            mask = nm_mask_along_dim(x, dim=self.mask_dim, N=self.N, M=self.M)
            x_masked = x * mask
      

        return self.linear(x_masked)


class ActSparseConv2d(nn.Module):
    """
    Wrapper around nn.Conv2d that enforces N:M sparsity on *activations*
    along a configurable dimension before forwarding to the underlying Conv2d.

    Typical options:
      - mask_dim = 1  (default): per-sample, per-(H,W) channel-wise sparsity
        (original behavior).
      - mask_dim = 0: cross-batch sparsity per (C,H,W); for N=1,M=2, batch=2
        this chooses between the two samples at each (C,H,W).

    If N == M (or N/M is None), masking is a no-op.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        N: Optional[int] = None,
        M: Optional[int] = None,
        mask_dim: int = 1,
    ):
        super().__init__()
        _validate_nm(N, M)
        self.N = N
        self.M = M
        self.mask_dim = mask_dim
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode,
            device=device, dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        if self.N is None or self.M is None or self.N == self.M:
            x_masked = x
        else:
            mask = nm_mask_along_dim(x, dim=self.mask_dim, N=self.N, M=self.M)
            x_masked = x * mask
        return self.conv(x_masked)


# ---------------------------------------------------------------------------
# VGG-16 with activation sparsity for CIFAR-10
# ---------------------------------------------------------------------------

# VGG-16 cfg for CIFAR-10 (same as ImageNet VGG16 but input is 32x32)
_CFG_VGG16: List[Union[int, str]] = [
    64, 64, 'M',
    128, 128, 'M',
    256, 256, 256, 'M',
    512, 512, 512, 'M',
    512, 512, 512, 'M'
]


def _make_layers_vgg16(
    cfg: List[Union[int, str]],
    N: Optional[int],
    M: Optional[int],
    batch_norm: bool = True,
    conv_mask_dim: int = 1,
) -> nn.Sequential:
    """
    Build the convolutional feature extractor.

    - First conv is plain nn.Conv2d (no activation sparsity).
    - All subsequent convs are ActSparseConv2d with shared (N, M)
      and configurable `conv_mask_dim`.
    """
    layers: List[nn.Module] = []
    in_channels = 3
    first_conv_done = False

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if not first_conv_done:
                # First layer: NO activation sparsity
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=not batch_norm)
                first_conv_done = True
            else:
                # Sparse-activation conv
                conv2d = ActSparseConv2d(
                    in_channels, v, kernel_size=3, padding=1,
                    bias=not batch_norm, N=N, M=M, mask_dim=conv_mask_dim,
                )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


class VGG16Act(nn.Module):
    """
    VGG-16 for CIFAR-10 with activation N:M sparsity applied to:
      - ALL conv layers EXCEPT the very first one
      - classifier hidden layer(s) but NOT the final output Linear

    New: you can choose the dimension where sparsity is applied:

      - `conv_mask_dim`:
          1 (default): per-sample, channel-wise sparsity (original behavior).
          0          : cross-batch sparsity per (C,H,W).

      - `linear_mask_dim`:
          -1 (default): per-sample feature sparsity (original behavior).
           0         : cross-batch sparsity per feature.

    If N==M or N/M is None, ActSparse layers effectively no-op.
    """
    def __init__(
        self,
        num_classes: int = 10,
        N: Optional[int] = None,
        M: Optional[int] = None,
        batch_norm: bool = True,
        classifier_hidden: int = 512,
        dropout_p: float = 0.5,
        conv_mask_dim: int = 1,
        linear_mask_dim: int = -1,
    ):
        super().__init__()
        self.N = N
        self.M = M

        # Convolutional trunk
        self.features = _make_layers_vgg16(
            _CFG_VGG16, N=N, M=M, batch_norm=batch_norm,
            conv_mask_dim=conv_mask_dim,
        )

        # Force down to 1x1 after 5 pools on 32x32; using AdaptiveAvgPool for safety
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier:
        #   Flatten -> (optional sparse-activation Linear) -> ReLU/Dropout
        #   -> final Linear (no sparsity)
        cls_layers: List[nn.Module] = []
        if classifier_hidden and classifier_hidden > 0:
            # Hidden layer WITH activation sparsity
            cls_layers += [
                ActSparseLinear(
                    512, classifier_hidden, bias=True,
                    N=N, M=M, mask_dim=linear_mask_dim,
                ),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
                # Final layer WITHOUT activation sparsity (user requirement)
                nn.Linear(classifier_hidden, num_classes, bias=True),
            ]
        else:
            # Direct 512 -> num_classes WITHOUT activation sparsity
            cls_layers += [nn.Linear(512, num_classes, bias=True)]
        self.classifier = nn.Sequential(*cls_layers)

        self._init_weights()

    def _init_weights(self):
        # Kaiming init for convs; normal small std for linears, BN to 1/0
        for m in self.modules():
            if isinstance(m, ActSparseConv2d):
                nn.init.kaiming_normal_(m.conv.weight, mode='fan_out', nonlinearity='relu')
                if m.conv.bias is not None:
                    nn.init.zeros_(m.conv.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, ActSparseLinear):
                nn.init.normal_(m.linear.weight, 0, 0.01)
                if m.linear.bias is not None:
                    nn.init.zeros_(m.linear.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (B, 512, 1, 1)
        x = self.avgpool(x)           # (B, 512, 1, 1)
        x = torch.flatten(x, 1)       # (B, 512)
        x = self.classifier(x)        # (B, num_classes)
        return x


def vgg16_act_cifar10(
    N: Optional[int] = None,
    M: Optional[int] = None,
    batch_norm: bool = True,
    classifier_hidden: int = 512,
    dropout_p: float = 0.5,
    num_classes: int = 10,
    conv_mask_dim: int = 1,
    linear_mask_dim: int = -1,
) -> VGG16Act:
    """
    Convenience constructor. Example:
        # Standard per-sample sparsity:
        model = vgg16_act_cifar10(N=4, M=16)

        # Cross-batch sparsity in convs (e.g. N=1,M=2 with batch_size=2):
        model = vgg16_act_cifar10(N=1, M=2, conv_mask_dim=0)
    """
    return VGG16Act(
        num_classes=num_classes,
        N=N, M=M,
        batch_norm=batch_norm,
        classifier_hidden=classifier_hidden,
        dropout_p=dropout_p,
        conv_mask_dim=conv_mask_dim,
        linear_mask_dim=linear_mask_dim,
    )


# ---------------------------------------------------------------------------
# Training / eval utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += images.size(0)
    avg_loss = running_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


def create_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    # CIFAR-10 stats
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scaler=None, strict: bool = True):
    if not path.exists():
        print(f"No checkpoint found at {path}, starting from scratch.")
        return 0, 0.0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt and scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_acc = ckpt.get("best_acc", 0.0)
    return start_epoch, best_acc


def build_model(
    N: Optional[int],
    M: Optional[int],
    num_classes: int,
    batch_norm: bool,
    classifier_hidden: int,
    dropout_p: float,
    conv_mask_dim: int = 1,
    linear_mask_dim: int = -1,
) -> nn.Module:
    return vgg16_act_cifar10(
        N=N, M=M,
        batch_norm=batch_norm,
        classifier_hidden=classifier_hidden,
        dropout_p=dropout_p,
        num_classes=num_classes,
        conv_mask_dim=conv_mask_dim,
        linear_mask_dim=linear_mask_dim,
    )


def get_schedulers(optimizer, epochs: int, warmup_epochs: int):
    # Warmup (linear LR scale up) -> Cosine annealing
    if warmup_epochs > 0:
        def warmup_lambda(epoch):
            return max(1e-8, float(epoch + 1) / max(1, warmup_epochs))
        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return scheduler


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, print_freq=100):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    start = time.time()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += images.size(0)

        if (i + 1) % print_freq == 0:
            elapsed = time.time() - start
            avg_loss = running_loss / max(total, 1)
            acc = 100.0 * correct / max(total, 1)
            print(f"Epoch {epoch} | Iter {i+1}/{len(loader)} | "
                  f"Loss {avg_loss:.4f} | Acc {acc:.2f}% | {elapsed:.1f}s")
            start = time.time()

    avg_loss = running_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def get_validation_accuracy(model: nn.Module) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_dataloaders("./data", batch_size=128, num_workers=4)
    model.eval()
    total = 0
    correct = 0
    # NOTE: assumes model is already on the right device in the caller
    for images, targets in val_loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(images)
        pred = outputs.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += images.size(0)
    return 100.0 * correct / max(total, 1)


def train_and_save(
    model: nn.Module,
    epochs: int,
    save_path: Path,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    use_amp: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_dataloaders("./data", batch_size=128, num_workers=4)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum,
        weight_decay=weight_decay, nesterov=True
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and use_amp))

    # Multi-step scheduler: decays LR at 20%, 40%, 60%, 80% of total epochs
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=6,   # every 5 epochs
        gamma=0.5      # multiply LR by 0.5
    )

    # ---- Pre-training check ----
    val_loss, curr_val_acc = evaluate(model, val_loader, device)
    print(f"[Pre-train check] Current val: {val_loss:.4f}/{curr_val_acc:.2f}%")
    best_acc = curr_val_acc
    save_checkpoint(
        {
            "epoch": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_acc": best_acc,
        },
        Path(save_path),
    )

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and use_amp)):
                outputs = model(images)
                loss = criterion(outputs, targets)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += images.size(0)

        # Step the scheduler once per epoch
        scheduler.step()

        train_acc = 100.0 * correct / total
        train_loss = running_loss / total
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f}/{train_acc:.2f}% "
              f"| Val: {val_loss:.4f}/{val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.5f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "best_acc": best_acc,
                },
                Path(save_path),
            )
            print(f" Saved new best to {save_path} (acc={best_acc:.2f}%)")

    print(f"Done. Best val acc: {best_acc:.2f}%")
    return best_acc

