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


class SparsityGate(nn.Module):
    """
    Ultra-minimal gate that predicts sparsity mode from early features.
    
    Architecture:
        AdaptiveAvgPool2d(1) -> Flatten -> Linear(64, num_modes)
    
    Placed after first maxpool in VGG16 (64 channels, 32x32 -> 16x16).
    """
    
    def __init__(self, in_channels: int = 64, num_modes: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.num_modes = num_modes
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_modes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [batch, in_channels, H, W]
        
        Returns:
            Logits for each sparsity mode [batch, num_modes]
        """
        x = self.pool(x)           # [batch, in_channels, 1, 1]
        x = x.flatten(1)           # [batch, in_channels]
        return self.fc(x)          # [batch, num_modes]
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the optimal sparsity mode index.
        
        Returns:
            Mode indices [batch]
        """
        logits = self.forward(x)
        return logits.argmax(dim=1)
    
    def probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get softmax probabilities over modes.
        
        Returns:
            Probabilities [batch, num_modes]
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class GatedVGG16(nn.Module):
    """
    VGG16 wrapper with integrated sparsity gate.
    
    The gate is placed after the first maxpool layer and predicts
    which sparsity mode to use for the rest of the forward pass.
    """
    
    def __init__(self, base_model: nn.Module, num_modes: int = 2):
        super().__init__()
        self.base_model = base_model
        self.num_modes = num_modes
        
        # Extract features up to and after first maxpool
        # VGG16 structure: Conv(64) -> BN -> ReLU -> Conv(64) -> BN -> ReLU -> MaxPool
        # We want gate after the MaxPool (index 6 in the features Sequential)
        self.features_pre_gate = base_model.features[:7]   # Up to and including first MaxPool
        self.features_post_gate = base_model.features[7:]  # Rest of features
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
        
        # Create the gate
        self.gate = SparsityGate(in_channels=64, num_modes=num_modes)
        
        # Mode registry: maps mode index -> sparsity key
        self.mode_registry = []
        
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
        from nm_sparsity_utils import set_active_sparsity, restore_bn_stats
        
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
    
    def gated_forward(self, x: torch.Tensor, mode_priority: list[str] | None = None):
        """
        Forward pass with automatic mode selection via the gate.
        
        Args:
            x: Input tensor
            mode_priority: Optional list of mode keys in order of preference
                          (default: uses mode_registry, most sparse first)
        
        Returns:
            (logits, selected_mode_key, gate_probs)
        """
        from nm_sparsity_utils import set_active_sparsity, restore_bn_stats
        
        # Get features up to gate
        x_pre = self.features_pre_gate(x)
        
        # Get gate prediction
        gate_logits = self.gate(x_pre)
        gate_probs = F.softmax(gate_logits, dim=1)
        predicted_idx = gate_logits.argmax(dim=1)
        
        # Map predicted index to mode key
        if mode_priority is None:
            mode_priority = self.mode_registry
        
        # For now, use the predicted mode directly
        # (more sophisticated routing can be added later)
        selected_idx = predicted_idx.item() if predicted_idx.numel() == 1 else predicted_idx[0].item()
        selected_key = mode_priority[selected_idx % len(mode_priority)]
        
        # Set the selected mode
        set_active_sparsity(self.base_model, selected_key)
        restore_bn_stats(self.base_model, selected_key)
        
        # Continue forward pass with selected mode
        x = self.features_post_gate(x_pre)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        
        return logits, selected_key, gate_probs


def create_gate_from_model(model: nn.Module) -> SparsityGate:
    """
    Create a gate module based on the model's registered sparsity modes.
    
    Args:
        model: VGG16 model with nm_masks registered
    
    Returns:
        SparsityGate with correct num_modes
    """
    # Count registered sparsity modes
    modes = set()
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'nm_masks'):
            modes.update(module.nm_masks.keys())
            break
    
    num_modes = len(modes) if modes else 2  # Default to 2 if no masks found
    return SparsityGate(in_channels=64, num_modes=num_modes)


def select_best_sparse_mode(correctness: dict[str, torch.Tensor], 
                            mode_priority: list[str]) -> torch.Tensor:
    """
    Select the best (most sparse) mode that predicts correctly for each sample.
    
    Args:
        correctness: Dict mapping sparsity_key -> boolean tensor [batch] 
                    indicating correct prediction
        mode_priority: List of mode keys in order of decreasing sparsity
                      (most sparse first, e.g., ['1:32', '1:16', '1:8', '1:4', 'dense'])
    
    Returns:
        Target mode indices [batch] for gate training
    """
    batch_size = next(iter(correctness.values())).shape[0]
    device = next(iter(correctness.values())).device
    
    # Initialize with last (least sparse) mode as fallback
    target_modes = torch.full((batch_size,), len(mode_priority) - 1, 
                              dtype=torch.long, device=device)
    
    # Iterate from least sparse to most sparse
    # (later assignments override with more sparse modes that work)
    for i, key in enumerate(reversed(mode_priority)):
        if key in correctness:
            target_modes[correctness[key]] = len(mode_priority) - 1 - i
    
    return target_modes


def compute_gate_loss(gate_logits: torch.Tensor, 
                      correctness: dict[str, torch.Tensor],
                      mode_priority: list[str]) -> torch.Tensor:
    """
    Compute cross-entropy loss for gate training.
    
    Args:
        gate_logits: Gate output [batch, num_modes]
        correctness: Dict mapping sparsity_key -> boolean tensor [batch]
        mode_priority: List of mode keys in order of decreasing sparsity
    
    Returns:
        Scalar loss tensor
    """
    target_indices = select_best_sparse_mode(correctness, mode_priority)
    return F.cross_entropy(gate_logits, target_indices)
