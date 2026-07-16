"""
runner.py
=========
Shared utilities for experiment pipelines (main.py, main_extended.py, etc.)
"""

import contextlib
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import nm_sparsity_utils as su


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs"
os.makedirs(BASE_DIR, exist_ok=True)


def read_eval(checkpoint_path: str) -> dict | None:
    if not os.path.exists(checkpoint_path):
        return None
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    return ckpt.get("metrics", {}).get("eval", None)


def format_eval_line(key: str, loss: float, acc: float) -> str:
    return f"    {key:<10}  loss={loss:.4f}  acc={acc:.2f}%"


def overview_section(
    title: str,
    checkpoint: str,
    frozen: str,
    mask: str,
    eval_results: dict | None = None,
    note: str = "",
) -> list[str]:
    lines = [f"\n{'─' * 55}", f"  {title}", f"{'─' * 55}"]
    lines.append(f"  Checkpoint: {checkpoint}")
    if note:
        lines.append(f"  Note: {note}")
    lines.append(f"  Mask applied: {mask}")
    lines.append(f"  Frozen: {frozen}")
    if eval_results:
        lines.append("  Results:")
        for key, (loss, acc) in sorted(eval_results.items()):
            lines.append(format_eval_line(key, loss, acc))
    else:
        lines.append("  Results: (checkpoint not found)")
    return lines


def run_phase(
    name: str,
    model: nn.Module,
    val_loader: DataLoader,
    train_loader: DataLoader,
    checkpoint_path: str,
    comparison_path: str,
    snapshot_before: dict | None,
    epochs: int,
    N: int,
    M: int,
    freeze_params: list | None = None,
    sparsity_key: str | None = None,
    train_kwargs: dict | None = None,
    skip_if_exists: bool = True,
) -> dict:
    """Run a training phase with snapshot + comparison + eval.

    If *skip_if_exists* is ``True`` and *checkpoint_path* already exists,
    the checkpoint is loaded and training is skipped.

    Returns the post-training (or post-load) snapshot.
    """
    header = f"  Phase: {name}  |  epochs={epochs}  |  N={N}  M={M}"
    print(f"\n{'=' * 60}")
    print(f"  {header}")
    if sparsity_key:
        print(f"  mask: {sparsity_key}")
    print(f"{'=' * 60}")

    if skip_if_exists and os.path.exists(checkpoint_path):
        print(f"  [CACHED] Checkpoint exists, loading and skipping training.")
        su.load_checkpoint(model, checkpoint_path)
        return su.parameter_snapshot(model)

    su.train(
        model=model,
        epochs=epochs,
        N=N,
        M=M,
        freeze_params=freeze_params,
        **(train_kwargs or {}),
    )
    eval_results = su.evaluate_all_sparsity_modes(model, val_loader, DEVICE,
                                                   recalibrate_loader=train_loader)
    su.save_checkpoint(model, checkpoint_path, metrics={"eval": eval_results})

    snap_after = su.parameter_snapshot(model)
    if snapshot_before is not None:
        with open(comparison_path, "w") as f:
            if eval_results:
                f.write("Eval results:\n")
                for key, (loss, acc) in sorted(eval_results.items()):
                    f.write(f"  {key:<10}  loss={loss:.4f}  acc={acc:.2f}%\n")
                f.write("\n")
            with contextlib.redirect_stdout(f):
                su.compare_snapshots(
                    model,
                    snapshot_before,
                    snap_after,
                    sparsity_key=sparsity_key,
                    line_cap=50,
                )
        print(f"  [COMPARISON] saved to {comparison_path}")
    return snap_after
