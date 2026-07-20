"""
gate.py
-------
Sparsity mode gating for VGG16 with N:M sparsity modes.

The gate predicts the optimal sparsity mode for each input based on
early features, prioritizing accuracy while maximizing sparsity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsity_modes import set_active_sparsity, restore_bn_stats

class SparsityGate(nn.Module):
    """
    Gate that predicts confidence score for current sparsity mode.
    
    Architecture:
        AdaptiveAvgPool2d(1) -> Flatten -> Linear(channels, 128) -> ReLU -> Linear(128, 1)
    
    Output: confidence score in [0, 1] after sigmoid
    High confidence = "trust current mode's features"
    Low confidence = "cascade to less sparse mode"
    
    Placed after a maxpool layer to see features from current sparsity mode.
    """
    
    def __init__(self, in_channels: int = 256, num_modes: int = 2):
        """
        Args:
            in_channels: Number of feature channels at gate position
            num_modes: Number of sparsity modes (for registry, not output)
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_modes = num_modes
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, 128)
        self.fc2 = nn.Linear(128, 1)  # Single confidence output
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [batch, in_channels, H, W]
        
        Returns:
            Confidence scores [batch, 1] (logits, before sigmoid)
        """
        x = self.pool(x)           # [batch, in_channels, 1, 1]
        x = x.flatten(1)           # [batch, in_channels]
        x = F.relu(self.fc1(x))    # [batch, 128]
        return self.fc2(x)         # [batch, 1]
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict whether to accept current mode (1) or cascade (0).
        
        Returns:
            Binary decisions [batch]: 1=accept, 0=cascade
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).long()
    
    def confidence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get confidence scores (after sigmoid).
        
        Returns:
            Confidence [batch, 1] in range [0, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)


class GatedVGG16(nn.Module):
    """
    VGG16 with cascading sparsity gate.
    
    The gate predicts confidence at each sparsity level:
    - Start at most sparse mode (1:32)
    - If gate confidence > threshold, accept and classify
    - Otherwise cascade to next less sparse mode (1:16, 1:8, 1:4, dense)
    
    Gate sees features computed with CURRENT active sparsity mode,
    so it can detect feature quality degradation.
    """
    
    def __init__(self, base_model: nn.Module, num_modes: int = 2, 
                 confidence_threshold: float = 0.5, gate_position: int = 3):
        """
        Args:
            base_model: VGG16 model with sparsity masks
            num_modes: Number of sparsity modes
            confidence_threshold: Threshold for accepting current mode
            gate_position: Which maxpool to place gate after (1, 2, or 3)
        """
        super().__init__()
        self.base_model = base_model
        self.num_modes = num_modes
        self.confidence_threshold = confidence_threshold
        self.gate_position = gate_position
        
        # VGG16 maxpool positions: 6, 13, 23, 33, 43
        # gate_position=1 -> after index 6 (64 channels)
        # gate_position=2 -> after index 13 (128 channels)
        # gate_position=3 -> after index 23 (256 channels)
        gate_indices = {1: 7, 2: 14, 3: 24}
        gate_channels = {1: 64, 2: 128, 3: 256}
        
        split_idx = gate_indices.get(gate_position, 24)
        gate_ch = gate_channels.get(gate_position, 256)
        
        self.features_pre_gate = base_model.features[:split_idx]
        self.features_post_gate = base_model.features[split_idx:]
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
        
        # Create the gate (single gate reused at all sparsity levels)
        self.gate = SparsityGate(in_channels=gate_ch, num_modes=num_modes)
        
        # Mode registry: ordered from most sparse to least sparse
        self.mode_registry = []
        
        # Stats tracking
        self.cascade_stats = {'total': 0, 'accepts_at_level': {}}
        
        # Mode registry: maps mode index -> sparsity key
        self.mode_registry = []
        
        # Stats tracking
        self.fallback_count = 0
        self.total_samples = 0
        
    def register_mode(self, sparsity_key: str):
        """Register a sparsity mode. Call for each mode in order of decreasing sparsity."""
        self.mode_registry.append(sparsity_key)
        
    def set_mode_registry(self, keys: list[str]):
        """Set the complete mode registry."""
        self.mode_registry = list(keys)
        
    def forward_with_mode(self, x: torch.Tensor, mode_key: str) -> torch.Tensor:
        """
        Forward pass with a specific sparsity mode active.
        
        Args:
            x: Input tensor
            mode_key: Sparsity key (e.g., 'dense', '1:4')
        
        Returns:
            Logits [batch, num_classes]
        """
        # Set the active sparsity mode
        set_active_sparsity(self.base_model, mode_key)
        restore_bn_stats(self.base_model, mode_key)
        
        # Full forward pass
        x = self.base_model.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    
    def forward(self, x: torch.Tensor, return_gate_output: bool = False):
        """
        Standard forward pass (uses currently active sparsity mode).
        
        Args:
            x: Input tensor
            return_gate_output: If True, also return gate logits
        
        Returns:
            If return_gate_output: (logits, gate_features)
            Else: logits
        """
        # Get features up to gate
        x_pre = self.features_pre_gate(x)
        
        # Run gate (but don't use output for routing in standard forward)
        gate_output = self.gate(x_pre)
        
        # Continue with rest of model
        x = self.features_post_gate(x_pre)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        
        if return_gate_output:
            return logits, gate_output
        return logits
    
    def cascade_forward(self, x: torch.Tensor, mode_order: list[str] | None = None,
                        return_stats: bool = False):
        """
        Cascade inference through sparsity modes from most to least sparse.
        
        For each sample:
          Start at most sparse mode (e.g., 1:32)
          Compute features with current mode's masks/BN
          Run gate to get confidence
          If confidence > threshold: accept and classify
          Else: cascade to next less sparse mode
        
        Args:
            x: Input tensor [batch, 3, 32, 32]
            mode_order: Modes from most sparse to least sparse
                       (default: uses mode_registry)
            return_stats: If True, return cascade statistics
        
        Returns:
            If return_stats: (logits, selected_mode, all_confidences, stats)
            Else: (logits, selected_mode, all_confidences)
        """    
        
        if mode_order is None:
            mode_order = self.mode_registry
        
        batch_size = x.size(0)
        device = x.device
        
        # Track which samples are still cascading (not yet accepted)
        remaining = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Track selected mode and confidence for each sample
        selected_mode = [''] * batch_size
        all_confidences = torch.zeros((batch_size, len(mode_order)), device=device)
        
        # Final logits (initialized to zeros, will be filled when sample accepts)
        final_logits = None
        
        cascade_counts = {mode: 0 for mode in mode_order}
        
        for level_idx, mode_key in enumerate(mode_order):
            if not remaining.any():
                break  # All samples accepted at earlier level
            
            # Set current sparsity mode
            set_active_sparsity(self.base_model, mode_key)
            restore_bn_stats(self.base_model, mode_key)
            
            # Compute features up to gate with current mode
            x_pre = self.features_pre_gate(x)
            
            # Get gate confidence for remaining samples
            gate_logits = self.gate(x_pre[remaining])
            confidence = torch.sigmoid(gate_logits).squeeze(-1)  # [num_remaining]
            
            # Store confidence for all samples (only meaningful for remaining)
            all_confidences[remaining, level_idx] = confidence
            
            # Determine which samples accept at this level
            accept = confidence > self.confidence_threshold
            
            if accept.any():
                # Get indices of accepting samples in original batch
                remaining_indices = remaining.nonzero(as_tuple=True)[0]
                accept_indices = remaining_indices[accept]
                
                # Compute features for accepting samples
                x_post = self.features_post_gate(x_pre[accept_indices])
                x_post = self.avgpool(x_post)
                x_post = torch.flatten(x_post, 1)
                logits = self.classifier(x_post)
                
                # Store logits for accepting samples
                if final_logits is None:
                    final_logits = torch.zeros((batch_size, logits.shape[1]), device=device)
                final_logits[accept_indices] = logits
                
                # Update tracking
                for idx in accept_indices:
                    selected_mode[idx.item()] = mode_key
                    cascade_counts[mode_key] += 1
                
                # Remove accepted samples from remaining
                remaining[accept_indices] = False
        
        # Handle any remaining samples (shouldn't happen if dense is last)
        if remaining.any():
            # Force use dense for remaining
            remaining_indices = remaining.nonzero(as_tuple=True)[0]
            set_active_sparsity(self.base_model, 'dense')
            restore_bn_stats(self.base_model, 'dense')
            x_pre = self.features_pre_gate(x[remaining_indices])
            x_post = self.features_post_gate(x_pre)
            x_post = self.avgpool(x_post)
            x_post = torch.flatten(x_post, 1)
            logits = self.classifier(x_post)
            if final_logits is None:
                final_logits = torch.zeros((batch_size, logits.shape[1]), device=device)
            final_logits[remaining_indices] = logits
            for idx in remaining_indices:
                selected_mode[idx.item()] = 'dense'
                cascade_counts['dense'] += 1
        
        # Update stats
        self.cascade_stats['total'] += batch_size
        for mode, count in cascade_counts.items():
            if mode not in self.cascade_stats['accepts_at_level']:
                self.cascade_stats['accepts_at_level'][mode] = 0
            self.cascade_stats['accepts_at_level'][mode] += count
        
        stats = {
            'cascade_counts': cascade_counts,
            'avg_confidence': all_confidences.mean().item(),
            'batch_size': batch_size,
        }
        
        if return_stats:
            return final_logits, selected_mode, all_confidences, stats
        return final_logits, selected_mode, all_confidences
    
    def reset_stats(self):
        """Reset cascade statistics."""
        self.cascade_stats = {'total': 0, 'accepts_at_level': {}}
    
    def get_mode_distribution(self) -> dict[str, float]:
        """Get the fraction of samples accepted at each level."""
        if self.cascade_stats['total'] == 0:
            return {}
        return {
            mode: count / self.cascade_stats['total']
            for mode, count in self.cascade_stats['accepts_at_level'].items()
        }


def create_gate_from_model(model: nn.Module, gate_position: int = 3) -> SparsityGate:
    """
    Create a gate module based on the model's registered sparsity modes.
    
    Args:
        model: VGG16 model with nm_masks registered
        gate_position: Which maxpool to place gate after (1, 2, or 3)
    
    Returns:
        SparsityGate with correct in_channels
    """
    gate_channels = {1: 64, 2: 128, 3: 256}
    
    # Count registered sparsity modes
    modes = set()
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'nm_masks'):
            modes.update(module.nm_masks.keys())
            break
    
    num_modes = len(modes) if modes else 2
    in_channels = gate_channels.get(gate_position, 256)
    return SparsityGate(in_channels=in_channels, num_modes=num_modes)


def compute_gate_loss(gate_logits: torch.Tensor, correctness: torch.Tensor,
                      balanced: bool = True) -> torch.Tensor:
    """
    Compute binary cross-entropy loss for cascade gate training.
    
    The gate learns to predict whether the current sparsity mode will
    produce a correct prediction.
    
    When balanced=True, the loss weights incorrect predictions more heavily
    to counter the class imbalance (most modes are correct >87% of the time).
    Without balancing, the gate optimally always predicts high confidence and
    never learns to cascade.
    
    Args:
        gate_logits: Gate output [batch, 1] (before sigmoid)
        correctness: Boolean tensor [batch] indicating if current mode is correct
        balanced: If True, use class-balanced weighting
    
    Returns:
        Scalar BCE loss tensor
    """
    targets = correctness.float()
    
    if balanced:
        n_correct = correctness.sum().float()
        n_incorrect = correctness.numel() - n_correct
        if n_incorrect > 0 and n_correct > 0:
            pos_weight = (n_incorrect / n_correct).unsqueeze(0)
            return F.binary_cross_entropy_with_logits(
                gate_logits.squeeze(-1), targets, pos_weight=pos_weight
            )
    
    return F.binary_cross_entropy_with_logits(gate_logits.squeeze(-1), targets)
