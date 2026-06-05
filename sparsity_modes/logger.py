"""
Global logging utilities for the sparsity training pipeline.
"""
import logging
import os
import sys

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


def frozen_summary(model) -> list[str]:
    """Inspect frozen state and return a list of categorized summary lines."""
    from .freeze import _element_freeze_masks, frozen_parameter_names

    parts: list[str] = []
    fnames = frozen_parameter_names(model)
    bn = [n for n in fnames if "norm" in n.lower() or "bn" in n.lower()]
    bias = [n for n in fnames if n.endswith(".bias") and n not in bn]
    weights = [n for n in fnames if n.endswith(".weight") and n not in bn]

    if bn:
        parts.append(f"batchnorm: {len(bn)} params")
    if bias:
        parts.append(f"biases: {len(bias)} params")
    if weights:
        layers = set(n.rsplit(".weight", 1)[0] for n in weights)
        parts.append(f"full weights: {', '.join(sorted(layers))}")

    if _element_freeze_masks:
        n_modules = len(_element_freeze_masks)
        total_frozen = sum(int(mask.sum().item()) for mask in _element_freeze_masks.values())
        parts.append(f"element masks: {n_modules} modules ({total_frozen:,} elements)")

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
