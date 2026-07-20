"""
sparsity_modes - N:M Sparsity with Switchable Modes

A framework for training models with multiple sparsity modes that can be
switched at inference time. Each mode maintains its own BatchNorm statistics.
"""

# Mask injection and activation
from .masks import (
    inject_switchable_mask,
    set_active_sparsity,
    get_active_sparsity,
)

# Freezing utilities
from .freeze import (
    # Parameter-level freeze
    register_frozen_param,
    unregister_frozen_param,
    is_frozen,
    zero_frozen_gradients,
    all_trainable_parameters,
    frozen_parameter_names,
    # Element-wise freeze
    register_element_freeze,
    clear_element_freeze,
    # User-facing helpers
    freeze_biases,
    freeze_mask,
    freeze_inverse_mask,
    freeze_batchnorm,
    freeze_non_masked_layers,
    freeze_all,
    unfreeze_all,
)

# BN buffer management
from .bn_buffers import (
    save_active_bn_stats,
    restore_bn_stats,
)

# Training
from .training import train, train_joint

# Evaluation
from .evaluation import evaluate_all_sparsity_modes

# Validation
from .validation import (
    snapshot_parameters,
    parameter_snapshot,
    verify_after_training,
    compare_snapshots,
)

# Checkpoints
from .checkpoints import (
    save_checkpoint,
    load_checkpoint,
)

# Output
from .output import OutputLogger

# Visualization
from .visualize import visualize, visualize_diff

# Logging helpers
from .logger import build_overview, build_phase_summary


__all__ = [
    # Masks
    'inject_switchable_mask',
    'set_active_sparsity',
    'get_active_sparsity',
    
    # Freeze - Parameter level
    'register_frozen_param',
    'unregister_frozen_param',
    'is_frozen',
    'zero_frozen_gradients',
    'all_trainable_parameters',
    'frozen_parameter_names',
    
    # Freeze - Element-wise
    'register_element_freeze',
    'clear_element_freeze',
    
    # Freeze - Helpers
    'freeze_biases',
    'freeze_mask',
    'freeze_inverse_mask',
    'freeze_batchnorm',
    'freeze_non_masked_layers',
    'freeze_all',
    'unfreeze_all',
    
    # BN Buffers
    'save_active_bn_stats',
    'restore_bn_stats',
    
    # Training
    'train',
    'train_joint',
    
    # Evaluation
    'evaluate_all_sparsity_modes',
    
    # Validation
    'snapshot_parameters',
    'parameter_snapshot',
    'verify_after_training',
    'compare_snapshots',
    
    # Checkpoints
    'save_checkpoint',
    'load_checkpoint',
    
    # Output
    'OutputLogger',
    
    # Visualization
    'visualize',
    'visualize_diff',
    
    # Logging helpers
    'build_overview',
    'build_phase_summary',
]
