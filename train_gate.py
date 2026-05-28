"""
train_gate.py
-------------
Joint training of VGG16 with sparsity mode gating.

The gate learns to predict the optimal sparsity mode for each input,
prioritizing accuracy while maximizing sparsity.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from vgg import build_model, create_dataloaders, evaluate
import nm_sparsity_utils as su
from gate import GatedVGG16, create_gate_from_model, compute_gate_loss, select_best_sparse_mode

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
    
    # Sort by sparsity (most sparse first)
    modes = sorted(list(modes))
    if 'dense' in modes:
        modes.remove('dense')
        modes.insert(0, 'dense')
    
    return modes


def evaluate_with_gate(gated_model: GatedVGG16, loader: DataLoader, 
                       mode_priority: list[str], device: torch.device) -> tuple[float, float, dict]:
    """
    Evaluate the gated model.
    
    Returns:
        (loss, accuracy, mode_selection_counts)
    """
    gated_model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    mode_counts = {key: 0 for key in mode_priority}
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            
            logits, selected_mode, _ = gated_model.gated_forward(images, mode_priority)
            loss = criterion(logits, targets)
            
            loss_sum += loss.item() * images.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            total += images.size(0)
            mode_counts[selected_mode] += images.size(0)
    
    return loss_sum / total, 100.0 * correct / total, mode_counts


def train_gate(
    base_model: nn.Module,
    gate: SparsityGate,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    gate_lr: float = 0.01,
    model_lr: float = 0.01,
    gate_loss_weight: float = 1.0,
    mode_priority: list[str] | None = None,
    warmup_epochs: int = 5,
    checkpoint_path: str = None,
):
    """
    Joint training of model and gate.
    
    Args:
        base_model: VGG16 model with sparsity masks registered
        gate: SparsityGate module
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Total training epochs
        gate_lr: Learning rate for gate
        model_lr: Learning rate for model (classifier + features after gate)
        gate_loss_weight: Weight for gate loss (λ in total_loss = ce_loss + λ * gate_loss)
        mode_priority: Modes in order of decreasing sparsity (most sparse first)
        warmup_epochs: Epochs to train classifier only before activating gate
        checkpoint_path: Path to save checkpoint
    """
    if mode_priority is None:
        mode_priority = get_registered_modes(base_model)[1:]  # Exclude 'dense'
    
    # Wrap model with gate
    gated_model = GatedVGG16(base_model, num_modes=len(mode_priority) + 1)
    gated_model.set_mode_registry(['dense'] + mode_priority)
    gated_model = gated_model.to(DEVICE)
    
    # Separate optimizers
    gate_params = list(gate.parameters())
    model_params = list(gated_model.features_post_gate.parameters()) + \
                   list(gated_model.classifier.parameters())
    
    gate_optimizer = optim.SGD(gate_params, lr=gate_lr, momentum=0.9, weight_decay=5e-4)
    model_optimizer = optim.SGD(model_params, lr=model_lr, momentum=0.9, weight_decay=5e-4)
    
    gate_scheduler = lr_scheduler.CosineAnnealingLR(gate_optimizer, T_max=epochs)
    model_scheduler = lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=epochs)
    
    print(f"\n{'=' * 60}")
    print(f"  GATE TRAINING")
    print(f"{'=' * 60}")
    print(f"  Modes: {['dense'] + mode_priority}")
    print(f"  Epochs: {epochs} (warmup: {warmup_epochs})")
    print(f"  Gate LR: {gate_lr}, Model LR: {model_lr}")
    print(f"  Gate loss weight (λ): {gate_loss_weight}")
    print(f"{'=' * 60}\n")
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        gated_model.train()
        
        total_loss = 0
        ce_loss_sum = 0
        gate_loss_sum = 0
        correct = 0
        total = 0
        
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            # Get features up to gate
            x_pre = gated_model.features_pre_gate(images)
            
            # Run gate
            gate_logits = gate(x_pre)
            
            # For training signal, we need to know which modes would be correct
            # Run forward pass with each mode to get correctness signal
            correctness = {}
            logits_by_mode = {}
            
            for mode_key in ['dense'] + mode_priority:
                su.set_active_sparsity(base_model, mode_key)
                su.restore_bn_stats(base_model, mode_key)
                
                # Continue forward from gate point
                x = gated_model.features_post_gate(x_pre)
                x = gated_model.avgpool(x)
                x = torch.flatten(x, 1)
                logits = gated_model.classifier(x)
                
                logits_by_mode[mode_key] = logits
                correctness[mode_key] = (logits.argmax(1) == targets)
            
            # Use dense mode for main classification loss
            dense_logits = logits_by_mode['dense']
            ce_loss = nn.CrossEntropyLoss()(dense_logits, targets)
            
            # Compute gate loss
            if epoch >= warmup_epochs:
                gate_loss = compute_gate_loss(gate_logits, correctness, ['dense'] + mode_priority)
                total_loss = ce_loss + gate_loss_weight * gate_loss
                gate_loss_sum += gate_loss.item() * images.size(0)
            else:
                total_loss = ce_loss
            
            # Backprop
            gate_optimizer.zero_grad()
            model_optimizer.zero_grad()
            total_loss.backward()
            gate_optimizer.step()
            model_optimizer.step()
            
            # Stats
            total_loss_item = total_loss.item()
            total_loss_sum += total_loss_item * images.size(0)
            ce_loss_sum += ce_loss.item() * images.size(0)
            correct += (dense_logits.argmax(1) == targets).sum().item()
            total += images.size(0)
        
        gate_scheduler.step()
        model_scheduler.step()
        
        # Validation
        val_loss, val_acc, mode_counts = evaluate_with_gate(
            gated_model, val_loader, mode_priority, DEVICE
        )
        
        # Print epoch stats
        gate_loss_avg = gate_loss_sum / total if epoch >= warmup_epochs else 0
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss {total_loss_sum/total:.4f} "
              f"(CE: {ce_loss_sum/total:.4f}, Gate: {gate_loss_avg:.4f}) | "
              f"Train Acc {100*correct/total:.2f}% | "
              f"Val Acc {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if checkpoint_path:
                torch.save({
                    'gate_state_dict': gate.state_dict(),
                    'model_state_dict': gated_model.state_dict(),
                    'mode_priority': mode_priority,
                    'epoch': epoch,
                    'val_acc': val_acc,
                }, checkpoint_path)
                print(f"  [SAVE] Best checkpoint saved to {checkpoint_path}")
    
    print(f"\n  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Final mode distribution: {mode_counts}")
    
    return gated_model, gate


def main():
    """Demo: Train gate on pre-trained model with multiple sparsity modes."""
    
    # Load pre-trained model with all sparsity modes
    print("\n[1] Loading pre-trained model with sparsity modes...")
    
    # Check if progressive sparse checkpoint exists
    checkpoint_path = "./runs/progressive_sparse_1_4.pt"
    if not os.path.exists(checkpoint_path):
        print(f"  [WARN] {checkpoint_path} not found.")
        print("  Please run progressive_sparse.py first to create a model with all sparsity modes.")
        return
    
    model = build_model().to(DEVICE)
    su.load_checkpoint(model, checkpoint_path)
    
    # Get registered modes
    modes = get_registered_modes(model)
    print(f"  Registered modes: {modes}")
    
    # Create gate
    gate = create_gate_from_model(model).to(DEVICE)
    print(f"  Gate created: {gate}")
    
    # Create data loaders
    print("\n[2] Creating data loaders...")
    train_loader, val_loader = create_dataloaders("./data", batch_size=128, num_workers=4)
    
    # Mode priority (most sparse first, excluding 'dense' which is fallback)
    mode_priority = [m for m in modes if m != 'dense']
    
    # Train gate
    print("\n[3] Training gate...")
    gated_model, trained_gate = train_gate(
        base_model=model,
        gate=gate,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        gate_lr=0.01,
        model_lr=0.001,
        gate_loss_weight=0.5,
        mode_priority=mode_priority,
        warmup_epochs=3,
        checkpoint_path=f"{BASE_DIR}/gate_checkpoint.pt",
    )
    
    # Final evaluation
    print("\n[4] Final evaluation...")
    final_loss, final_acc, mode_counts = evaluate_with_gate(
        gated_model, val_loader, mode_priority, DEVICE
    )
    
    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Val Loss: {final_loss:.4f}")
    print(f"  Val Accuracy: {final_acc:.2f}%")
    print(f"  Mode selection distribution:")
    total = sum(mode_counts.values())
    for mode, count in mode_counts.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"    {mode}: {count} ({pct:.1f}%)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
