"""
sparsity_modes - N:M Sparsity with Switchable Mode Profiles

A framework for training models with multiple sparsity modes that can be
switched at inference time. Each mode maintains its own BatchNorm statistics.

Layer‑granularity masks, freeze profiles, and training profiles are all
specified declaratively and validated at construction time.

Quick start::

    from sparsity_modes import SparsityModel, ModeProfile, LayerSpec
    from sparsity_modes import FreezeProfile, TrainingProfile, OutputOptions
    from sparsity_modes import visualize, visualize_diff
    from sparsity_modes import build_truly_pruned, compare_models

    sparse = SparsityModel(
        model,
        mode_profiles=[...],
        freeze_profiles=[...],
        training_profiles=[...],
        output_options=OutputOptions(output_dir="./runs"),
    )
    sparse.train(mode="mixed", freeze="freeze_kept", training="default")
"""

# ===================================================================
# New API — Profile-driven wrapper
# ===================================================================

# Data objects
from .profiles import (
    LayerSpec,
    ModeProfile,
    FreezeProfile,
    TrainingProfile,
    OutputOptions,
)

# SparsityModel wrapper
from .model import SparsityModel

# Visualization
from .visualize import visualize, visualize_diff

# Output logging
from .output import OutputLogger

# Testing utilities
from .testing import build_truly_pruned, compare_models, compare_single_input

# ===================================================================
# Legacy API — Functional (backward compatible)
# ===================================================================

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
from .training import train

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


__all__ = [
    # === New API ===
    # Data objects
    "LayerSpec",
    "ModeProfile",
    "FreezeProfile",
    "TrainingProfile",
    "OutputOptions",
    # Wrapper
    "SparsityModel",
    # Visualization
    "visualize",
    "visualize_diff",
    # Output
    "OutputLogger",
    # Testing
    "build_truly_pruned",
    "compare_models",
    "compare_single_input",

    # === Legacy API ===
    # Masks
    "inject_switchable_mask",
    "set_active_sparsity",
    "get_active_sparsity",

    # Freeze - Parameter level
    "register_frozen_param",
    "unregister_frozen_param",
    "is_frozen",
    "zero_frozen_gradients",
    "all_trainable_parameters",
    "frozen_parameter_names",

    # Freeze - Element-wise
    "register_element_freeze",
    "clear_element_freeze",

    # Freeze - Helpers
    "freeze_biases",
    "freeze_mask",
    "freeze_inverse_mask",
    "freeze_batchnorm",
    "freeze_non_masked_layers",
    "freeze_all",
    "unfreeze_all",

    # BN Buffers
    "save_active_bn_stats",
    "restore_bn_stats",

    # Training
    "train",

    # Evaluation
    "evaluate_all_sparsity_modes",

    # Validation
    "snapshot_parameters",
    "parameter_snapshot",
    "verify_after_training",
    "compare_snapshots",

    # Checkpoints
    "save_checkpoint",
    "load_checkpoint",
]
