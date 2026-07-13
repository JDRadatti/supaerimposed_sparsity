"""
`sparsity_modes` — N:M Sparsity with Switchable Mode Profiles.

A framework for training models with multiple sparsity modes that can be
switched at inference time.  Each mode maintains its own BatchNorm statistics.
Layer‑granularity masks, freeze profiles, and training profiles are all
specified declaratively and validated at construction time.

Quick start::

    from v3 import SparsityModel, ModeProfile, LayerSpec, FreezeProfile, TrainingProfile, OutputOptions
    from v3 import visualize, visualize_diff, build_truly_pruned, compare_models
"""

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
from .testing import build_truly_pruned, compare_models

__all__ = [
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
]
