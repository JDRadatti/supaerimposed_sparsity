"""
Global logging utilities for the sparsity training pipeline.
"""
import logging
import os
import sys

import torch
import torch.nn as nn

_logger: logging.Logger | None = None


def setup_logger(verbose: bool = False, log_file: str | None = None, name: str = "sparsity") -> logging.Logger:
    """Configure the global logger.

    Console handler: INFO level (DEBUG if verbose).
    File handler (if log_file given): DEBUG level with timestamps.
    """
    global _logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    if log_file:
        abs_path = os.path.abspath(log_file)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)

    _logger = logger
    return logger


def info(msg: str, *args, **kwargs):
    if _logger:
        _logger.info(msg, *args, **kwargs)


def debug(msg: str, *args, **kwargs):
    if _logger:
        _logger.debug(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    if _logger:
        _logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    if _logger:
        _logger.error(msg, *args, **kwargs)


def banner(title: str):
    """Print a dashed section header."""
    if _logger:
        _logger.info("-" * 58)
        if title:
            _logger.info(title)


# ---------------------------------------------------------------------------
# Frozen / sparsity summaries
# ---------------------------------------------------------------------------

def frozen_summary(model) -> list[str]:
    """Inspect frozen state and return a list of categorized summary lines."""
    from .freeze import _element_freeze_masks, frozen_parameter_names

    parts: list[str] = []
    fnames = frozen_parameter_names(model)

    # Categorize by module type, not name pattern
    bn = []
    bias = []
    weights = []
    name_to_param = dict(model.named_parameters())
    name_to_module = dict(model.named_modules())

    for n in fnames:
        # Find the module that owns this parameter
        mod_name = n.rsplit(".", 1)[0] if "." in n else ""
        module = name_to_module.get(mod_name)
        if module is not None and isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn.append(n)
        elif n.endswith(".bias"):
            bias.append(n)
        elif n.endswith(".weight"):
            weights.append(n)
        else:
            bias.append(n)  # fallback

    if bn:
        parts.append(f"Parameter-level frozen: batchnorm: {len(bn)} params")
    if bias:
        parts.append(f"Parameter-level frozen: biases: {len(bias)} params")
    if weights:
        layers = set(n.rsplit(".weight", 1)[0] for n in weights)
        parts.append(f"Parameter-level frozen: full weights: {', '.join(sorted(layers))}")

    if _element_freeze_masks:
        n_modules = len(_element_freeze_masks)
        total_frozen = sum(int(mask.sum().item()) for mask in _element_freeze_masks.values())
        parts.append(f"Element-wise frozen: {n_modules} modules ({total_frozen:,} elements)")

        # Per-layer detail
        for pname, mask in sorted(_element_freeze_masks.items(), key=lambda x: x[0].numel(), reverse=True):
            for n, p in name_to_param.items():
                if p is pname:
                    layer_name = n.replace(".weight", "")
                    total = mask.numel()
                    frozen = int(mask.sum().item())
                    parts.append(f"  {layer_name}: {frozen:,} / {total:,} elements frozen")
                    break

    if not parts:
        return ["None"]

    return parts


def sparsity_summary(model) -> dict[str, str]:
    """For each registered sparsity key, compute percentage of weights zeroed."""
    results: dict[str, dict[str, int]] = {}
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "nm_masks"):
            for key, mask in module.nm_masks.items():
                if mask is None:
                    continue
                total = mask.numel()
                zeros = int((mask == 0).sum().item())
                if key not in results:
                    results[key] = {"total": 0, "zeros": 0}
                results[key]["total"] += total
                results[key]["zeros"] += zeros

    output = {}
    for key, counts in results.items():
        if counts["total"] > 0:
            output[key] = f"{100.0 * counts['zeros'] / counts['total']:.1f}%"
        else:
            output[key] = "N/A"

    return output


# ---------------------------------------------------------------------------
# Layer-level helpers
# ---------------------------------------------------------------------------

def _get_conv_linear_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return ``[(name, module), ...]`` for all Conv2d/Linear with masks."""
    return [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "nm_masks")
    ]


def _layer_sparsity_map(model: nn.Module) -> dict[str, dict[str, str]]:
    """Return ``{key: {module_name: sparsity_str}}`` for every registered key."""
    result: dict[str, dict[str, str]] = {}
    for name, module in _get_conv_linear_modules(model):
        if not module.nm_masks:
            continue
        for key in module.nm_masks:
            if key == "dense":
                continue
            if key not in result:
                result[key] = {}
            result[key][name] = key

    # Add dense entry: layers that don't have a specific key
    dense_map: dict[str, str] = {}
    for name, module in _get_conv_linear_modules(model):
        if not module.nm_masks:
            continue
        # If the module only has 'dense' key or no specific sparse key, mark as dense
        sparse_keys = [k for k in module.nm_masks if k != "dense"]
        if not sparse_keys:
            dense_map[name] = "dense"
    if dense_map:
        result["dense"] = dense_map

    return result


def _layer_shapes(model: nn.Module) -> list[tuple[str, str, int]]:
    """Return ``[(name, description, n_params), ...]`` for Conv2d/Linear layers."""
    shapes = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight
            shapes.append((
                name,
                f"Conv2d({w.shape[1]}, {w.shape[0]}, {w.shape[2]}x{w.shape[3]})",
                w.numel(),
            ))
        elif isinstance(module, nn.Linear):
            w = module.weight
            shapes.append((
                name,
                f"Linear({w.shape[1]}, {w.shape[0]})",
                w.numel(),
            ))
    return shapes


# ---------------------------------------------------------------------------
# Overview builder
# ---------------------------------------------------------------------------

def _categorize_frozen_params(model: nn.Module) -> tuple[list[str], list[str], list[str]]:
    """Categorize frozen param names into (batchnorm, bias, weights) by module type."""
    from .freeze import frozen_parameter_names

    fnames = frozen_parameter_names(model)
    name_to_module = dict(model.named_modules())

    bn, bias, weights = [], [], []
    for n in fnames:
        mod_name = n.rsplit(".", 1)[0] if "." in n else ""
        module = name_to_module.get(mod_name)
        if module is not None and isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn.append(n)
        elif n.endswith(".bias"):
            bias.append(n)
        elif n.endswith(".weight"):
            weights.append(n)
        else:
            bias.append(n)
    return bn, bias, weights


def build_overview(model: nn.Module, run_dir: str, timestamp: str,
                   commit: str, branch: str) -> str:
    """Build the full run overview text.

    Returns a string suitable for printing and writing to ``overview.txt``.
    """
    total_params = sum(p.numel() for p in model.parameters())
    model_class = type(model).__name__

    lines: list[str] = []
    w = lines.append

    w("=" * 60)
    w("  OVERVIEW")
    w(f"  Run: {run_dir}")
    w(f"  Timestamp: {timestamp}")
    w(f"  Git: {commit} on branch {branch}")
    w(f"  Model: {model_class}")
    w(f"  Total parameters: {total_params:,}")
    w("=" * 60)
    w("")

    # --- Registered sparsity modes ---
    sparsity_map = _layer_sparsity_map(model)
    if sparsity_map:
        w("  Registered Sparsity Modes:")
        for key in sorted(sparsity_map.keys()):
            assignments = sparsity_map[key]
            parts = [f"{n}={s}" for n, s in assignments.items()]
            w(f"    {key}: {', '.join(parts)}")
        w("")

    # --- Layer shapes ---
    shapes = _layer_shapes(model)
    if shapes:
        w("  Layer Shapes:")
        for name, desc, n_params in shapes:
            w(f"    {name:<14s} {desc:<24s} {n_params:,} params")
        w("")

    # --- Current freeze state ---
    from .freeze import _element_freeze_masks, _frozen_params

    bn, bias, weights = _categorize_frozen_params(model)

    w("  Current Freeze State:")
    has_freeze = bn or bias or weights or _element_freeze_masks
    if not has_freeze:
        w("    (none)")
    else:
        if bn or bias or weights:
            w("    Parameter-level frozen:")
            if bn:
                w(f"      batchnorm: {len(bn)} params")
            if bias:
                w(f"      biases: {len(bias)} params")
            if weights:
                layers = set(n.rsplit(".weight", 1)[0] for n in weights)
                w(f"      full weights: {', '.join(sorted(layers))}")

        if _element_freeze_masks:
            total_frozen = sum(int(m.sum().item()) for m in _element_freeze_masks.values())
            w(f"    Element-wise frozen:")
            name_to_param = dict(model.named_parameters())
            for pname, mask in sorted(_element_freeze_masks.items(), key=lambda x: x[0].numel(), reverse=True):
                for n, p in name_to_param.items():
                    if p is pname:
                        layer_name = n.replace(".weight", "")
                        total = mask.numel()
                        frozen = int(mask.sum().item())
                        w(f"      {layer_name}: {frozen:,} / {total:,} elements")
                        break
            w(f"      Total: {total_frozen:,} elements across {len(_element_freeze_masks)} modules")
    w("")

    # --- Current sparsity ---
    sparsity_pct = sparsity_summary(model)
    if sparsity_pct:
        w("  Current Sparsity:")
        for key in sorted(sparsity_pct.keys()):
            if key == "dense":
                continue
            w(f"    {key}: {sparsity_pct[key]} pruned")
        w("")

    w("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase summary builder
# ---------------------------------------------------------------------------

def build_phase_summary(
    name: str,
    model: nn.Module,
    epochs: int,
    N: int,
    M: int,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    label_smoothing: float = 0.0,
    cutout: int = 0,
    scheduler_desc: str = "StepLR(step_size=6, gamma=0.5)",
) -> str:
    """Build a brief phase summary string.

    Returns a string suitable for printing before a training phase begins.
    """
    sparsity_key = f"{N}:{M}" if N != 0 and M != 0 else "dense"

    lines: list[str] = []
    w = lines.append

    w("-" * 60)
    w(f"  {name}")
    w("-" * 60)
    w(f"  Mode:       {sparsity_key} (N={N}, M={M})")
    w(f"  Epochs:     {epochs}")
    w(f"  Optimizer:  SGD(lr={lr}, momentum={momentum}, weight_decay={weight_decay})")
    w(f"  Scheduler:  {scheduler_desc}")
    w(f"  Label smoothing: {label_smoothing}")
    w(f"  Cutout:     {cutout}")
    w("")

    # Freeze state
    from .freeze import _element_freeze_masks, _frozen_params

    bn, bias, weights = _categorize_frozen_params(model)

    has_freeze = bn or bias or weights or _element_freeze_masks
    if has_freeze:
        w("  Frozen before training:")
        if bn:
            w(f"    Parameter-level: batchnorm: {len(bn)} params")
        if bias:
            w(f"    Parameter-level: biases: {len(bias)} params")
        if weights:
            layers = set(n.rsplit(".weight", 1)[0] for n in weights)
            w(f"    Parameter-level: full weights: {', '.join(sorted(layers))}")
        if _element_freeze_masks:
            total_frozen = sum(int(m.sum().item()) for m in _element_freeze_masks.values())
            w(f"    Element-wise: {len(_element_freeze_masks)} modules ({total_frozen:,} elements)")
            name_to_param = dict(model.named_parameters())
            for pname, mask in sorted(_element_freeze_masks.items(), key=lambda x: x[0].numel(), reverse=True):
                for n, p in name_to_param.items():
                    if p is pname:
                        layer_name = n.replace(".weight", "")
                        total = mask.numel()
                        frozen = int(mask.sum().item())
                        w(f"      {layer_name}: {frozen:,} / {total:,} elements")
                        break
        w("")

    # Layers with active mask
    if sparsity_key != "dense":
        layers_with_mask = []
        for name, module in _get_conv_linear_modules(model):
            if hasattr(module, "nm_masks") and sparsity_key in module.nm_masks:
                layers_with_mask.append(name)
        if layers_with_mask:
            w(f"  Layers with active mask ({len(layers_with_mask)}):")
            # Print in rows of 4
            for i in range(0, len(layers_with_mask), 4):
                chunk = layers_with_mask[i:i + 4]
                w(f"    {', '.join(chunk)},")
            w("")

    w("-" * 60)
    return "\n".join(lines)
