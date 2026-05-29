"""
train_gate.py
-------------
Cascade training of sparsity mode gating for VGG16.

The gate learns to predict confidence at each sparsity level:
- Start at most sparse mode (1:32)
- If gate confidence > threshold, accept and classify
- Otherwise cascade to next less sparse mode

Training: For each mode level, train gate to predict correctness
  - At 1:32 level: target = (1:32 prediction == target)
  - At 1:16 level: target = (1:16 prediction == target)
  - etc.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from vgg import build_model, create_dataloaders, evaluate
import sparsity_modes as su
from gate import GatedVGG16, SparsityGate, create_gate_from_model, compute_gate_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs/gate"
os.makedirs(BASE_DIR, exist_ok=True)


def get_registered_modes(model: nn.Module) -> list[str]:
    """Get all registered sparsity modes from the model."""
    modes = set()
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'nm_masks'):
            modes.update(module.nm_masks.keys())
            break
    
    # Sort by sparsity (most sparse first: largest N:M ratio -> smallest)
    def _sparsity_key(key: str) -> tuple[int, int]:
        if key == 'dense':
            return (0, 0)  # least sparse
        parts = key.split(':')
        return (int(parts[1]), int(parts[0]))  # sort by mask factor asc, then N asc
    
    modes = sorted(modes, key=_sparsity_key, reverse=True)
    if 'dense' in modes:
        modes.remove('dense')
        modes.append('dense')  # Dense is least sparse, goes last
    
    return modes


def evaluate_cascade(gated_model: GatedVGG16, loader: DataLoader, 
                     mode_order: list[str], device: torch.device) -> tuple[float, float, dict, dict]:
    """
    Evaluate the gated model with cascade inference.
    
    Returns:
        (loss, accuracy, mode_selection_counts, stats)
    """
    gated_model.eval()
    gated_model.reset_stats()
    
    correct = 0
    total = 0
    loss_sum = 0
    mode_counts = {key: 0 for key in mode_order}
    confidence_sum = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            
            logits, selected_modes, confidences, stats = gated_model.cascade_forward(
                images, mode_order, return_stats=True
            )
            loss = criterion(logits, targets)
            
            loss_sum += loss.item() * images.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            total += images.size(0)
            
            for mode in selected_modes:
                mode_counts[mode] += 1
            confidence_sum += stats['avg_confidence'] * images.size(0)
    
    stats = {
        'avg_confidence': confidence_sum / total if total > 0 else 0,
        'cascade_counts': stats['cascade_counts'],
    }
    
    return loss_sum / total, 100.0 * correct / total, mode_counts, stats


def train_cascade_gate(
    base_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mode_order: list[str],
    epochs: int = 30,
    gate_lr: float = 0.01,
    checkpoint_path: str = None,
    confidence_threshold: float = 0.5,
    gate_position: int = 3,
):
    """
    Train cascade gate with frozen base model.
    
    For each epoch, we train the gate at ALL cascade levels:
    1. For each mode (from most sparse to least sparse):
       - Set that mode active
       - Compute features
       - Train gate to predict correctness (binary classification)
    
    Args:
        base_model: VGG16 model with sparsity masks registered (will be frozen)
        train_loader: Training data loader
        val_loader: Validation data loader
        mode_order: Modes from most sparse to least sparse (e.g., ['1:32', '1:16', '1:8', '1:4', 'dense'])
        epochs: Total training epochs
        gate_lr: Learning rate for gate
        checkpoint_path: Path to save checkpoint
        confidence_threshold: Threshold for cascade acceptance
        gate_position: Which maxpool to place gate after (1, 2, or 3)
    """
    # Wrap model with gate
    gated_model = GatedVGG16(
        base_model, 
        num_modes=len(mode_order),
        confidence_threshold=confidence_threshold,
        gate_position=gate_position,
    )
    gated_model.set_mode_registry(mode_order)
    gated_model = gated_model.to(DEVICE)
    
    # Freeze all base model parameters
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Optimize gate parameters
    gate = gated_model.gate
    gate_params = list(gate.parameters())
    gate_optimizer = optim.SGD(gate_params, lr=gate_lr, momentum=0.9, weight_decay=5e-4)
    gate_scheduler = lr_scheduler.CosineAnnealingLR(gate_optimizer, T_max=epochs)
    
    print(f"\n{'=' * 60}")
    print(f"  CASCADE GATE TRAINING (model frozen)")
    print(f"{'=' * 60}")
    print(f"  Modes (cascade order): {mode_order}")
    print(f"  Gate position: after maxpool {gate_position}")
    print(f"  Epochs: {epochs}")
    print(f"  Gate LR: {gate_lr}")
    print(f"  Confidence threshold: {confidence_threshold}")
    print(f"  Frozen params: {sum(p.numel() for p in base_model.parameters())}")
    print(f"  Trainable params: {sum(p.numel() for p in gate.parameters())}")
    print(f"{'=' * 60}\n")
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        gated_model.train()
        
        total_loss_sum = 0
        total_samples = 0
        train_modes = mode_order[:-1]  # exclude dense from training
        level_losses = {mode: 0.0 for mode in train_modes}
        level_counts = {mode: 0 for mode in train_modes}
        
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            batch_size = images.size(0)
            
            # Train gate at each cascade level (excluding dense — no cascade decision there)
            epoch_loss = 0.0
            
            for mode_key in mode_order[:-1]:
                # Set current sparsity mode
                su.set_active_sparsity(base_model, mode_key)
                su.restore_bn_stats(base_model, mode_key)
                
                # Compute features up to gate
                with torch.no_grad():
                    x_pre = gated_model.features_pre_gate(images)
                
                # Get gate output
                gate_logits = gate(x_pre)
                
                # Compute correctness target for this mode
                with torch.no_grad():
                    # Full forward with current mode
                    x_post = gated_model.features_post_gate(x_pre)
                    x_post = gated_model.avgpool(x_post)
                    x_post = torch.flatten(x_post, 1)
                    logits = gated_model.classifier(x_post)
                    correctness = (logits.argmax(1) == targets)
                
                # Compute loss
                loss = compute_gate_loss(gate_logits, correctness)
                epoch_loss += loss
                
                level_losses[mode_key] += loss.item() * batch_size
                level_counts[mode_key] += batch_size
            
            # Backprop (sum of losses from all levels)
            gate_optimizer.zero_grad()
            epoch_loss.backward()
            gate_optimizer.step()
            
            total_loss_sum += epoch_loss.item() * batch_size
            total_samples += batch_size
        
        gate_scheduler.step()
        
        # Validation
        val_loss, val_acc, mode_counts, eval_stats = evaluate_cascade(
            gated_model, val_loader, mode_order, DEVICE
        )
        
        # Print epoch stats
        avg_loss = total_loss_sum / total_samples
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Avg Loss {avg_loss:.4f} | "
              f"Val Acc {val_acc:.2f}%")
        print(f"  Level losses: " + " | ".join(
            f"{m}: {level_losses[m]/level_counts[m]:.4f}" for m in train_modes
        ))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if checkpoint_path:
                torch.save({
                    'gate_state_dict': gate.state_dict(),
                    'gated_model_state_dict': gated_model.state_dict(),
                    'mode_order': mode_order,
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'confidence_threshold': confidence_threshold,
                    'gate_position': gate_position,
                }, checkpoint_path)
                print(f"  [SAVE] Best checkpoint saved to {checkpoint_path}")
    
    print(f"\n  Best Val Acc: {best_val_acc:.2f}%")
    
    return gated_model, gate


def main():
    """Demo: Train cascade gate on pre-trained model with multiple sparsity modes."""
    
    # Load pre-trained model with all sparsity modes
    print("\n[1] Loading pre-trained model with sparsity modes...")
    
    # Check if progressive sparse checkpoint exists
    checkpoint_path = "./runs/progressive_sparse_dense_final.pt"
    if not os.path.exists(checkpoint_path):
        print(f"  [WARN] {checkpoint_path} not found.")
        print("  Please run progressive_sparse.py first to create a model with all sparsity modes.")
        return
    
    model = build_model().to(DEVICE)
    su.load_checkpoint(model, checkpoint_path)
    
    # Get registered modes (most sparse first, dense last)
    mode_order = get_registered_modes(model)
    print(f"  Registered modes: {mode_order}")
    
    # Create data loaders
    print("\n[2] Creating data loaders...")
    train_loader, val_loader = create_dataloaders("./data", batch_size=128, num_workers=4)
    
    # Train cascade gate
    print("\n[3] Training cascade gate...")
    gated_model, trained_gate = train_cascade_gate(
        base_model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode_order=mode_order,
        epochs=20,
        gate_lr=0.01,
        checkpoint_path=f"{BASE_DIR}/gate_checkpoint.pt",
        confidence_threshold=0.5,
        gate_position=3,
    )
    
    # Final evaluation
    print("\n[4] Final evaluation...")
    final_loss, final_acc, mode_counts, eval_stats = evaluate_cascade(
        gated_model, val_loader, mode_order, DEVICE
    )
    
    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Val Loss: {final_loss:.4f}")
    print(f"  Val Accuracy: {final_acc:.2f}%")
    print(f"  Avg confidence: {eval_stats['avg_confidence']:.3f}")
    print(f"  Mode selection distribution:")
    total = sum(mode_counts.values())
    for mode in mode_order:
        count = mode_counts.get(mode, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"    {mode}: {count} ({pct:.1f}%)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
