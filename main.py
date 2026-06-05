"""
main.py - Progressive Sparse Training with Multiple Sparsity Modes

Trains a model progressively with nested sparsity modes (1:32, 1:16, 1:8, 1:4),
freezing previously-trained modes at each stage.
"""
import argparse
import contextlib
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from vgg import (
    build_model,
    create_dataloaders,
    evaluate,
)
import sparsity_modes as su
from sparsity_modes.logger import (
    setup_logger,
    info,
    debug,
    banner,
    frozen_summary,
    sparsity_summary,
)

# Silence the NumPy VisibleDeprecationWarning that CIFAR triggers
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torchvision\.datasets\.cifar",
)

EPOCHS   = 200
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs"
os.makedirs(BASE_DIR, exist_ok=True)


def _run_phase(
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
    the checkpoint is loaded and training is skipped (the phase is assumed
    to have been completed in a previous run).

    Returns the post-training (or post-load) snapshot.
    """
    banner(f"  {name}")
    info(f"  Epochs:   {epochs}")
    info(f"  Mode:     {sparsity_key or 'dense'}")
    if sparsity_key:
        summary = sparsity_summary(model)
        if sparsity_key in summary:
            info(f"  Sparsity: {summary[sparsity_key]}")
    info("  Frozen:")
    for line in frozen_summary(model):
        info(f"    {line}")

    if skip_if_exists and os.path.exists(checkpoint_path):
        debug(f"  [CACHED] Checkpoint exists, loading and skipping training.")
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
    eval_results = su.evaluate_all_sparsity_modes(model, val_loader, DEVICE)
    su.save_checkpoint(model, checkpoint_path, metrics={"eval": eval_results})

    info("Eval Results:")
    for key, (loss, acc) in sorted(eval_results.items()):
        info(f"    {key:<10}  loss={loss:.4f}  acc={acc:.2f}%")

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
        debug(f"  [COMPARISON] saved to {comparison_path}")

    info("-" * 58)
    return snap_after


def main() -> None:
    parser = argparse.ArgumentParser(description="Progressive sparse training")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("--log-file", default=None, help="log file path")
    args = parser.parse_args()

    log_file = args.log_file or f"{BASE_DIR}/main_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logger(verbose=args.verbose, log_file=log_file)

    #
    # 0. Init
    #
    model = build_model().to(DEVICE)
    train_loader, val_loader = create_dataloaders("./data", batch_size=128, num_workers=4)

    #
    # 1. Initial Training
    #
    snap = su.parameter_snapshot(model)
    snap = _run_phase(
        name="Dense (1:1)",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{BASE_DIR}/phase1_dense.pt",
        comparison_path=f"{BASE_DIR}/phase1_dense_comparison.txt",
        snapshot_before=snap,
        epochs=EPOCHS,
        N=0, M=0,
        train_kwargs=dict(
            lr=0.1,
            scheduler_fn=lambda opt: lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS),
            label_smoothing=0.1,
            cutout=16,
        ),
    )

    #
    # 2. Inject all masks
    #
    su.inject_switchable_mask(
        model=model,
        sparsity_key="1:32",
        N=1, M=32,
        conv_axis="out",
        linear_orientation="row",
    )

    su.inject_switchable_mask(
        model=model,
        sparsity_key="1:16",
        N=1, M=16,
        conv_axis="out",
        linear_orientation="row",
    )
    su.inject_switchable_mask(
        model=model,
        sparsity_key="1:8",
        N=1, M=8,
        conv_axis="out",
        linear_orientation="row",
    )
    su.inject_switchable_mask(
        model=model,
        sparsity_key="1:4",
        N=1, M=4,
        conv_axis="out",
        linear_orientation="row",
    )
    #
    # 2. 1:32 sparse training
    #
    su.freeze_inverse_mask(model, "1:32")
    #su.freeze_non_masked_layers(model, "1:32")
    snap = su.parameter_snapshot(model)
    snap = _run_phase(
        name="Sparse 1:32 (pruned positions frozen)",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{BASE_DIR}/progressive_sparse_1_32.pt",
        comparison_path=f"{BASE_DIR}/progressive_sparse_1_32_comparison.txt",
        snapshot_before=snap,
        epochs=30,
        N=1, M=32,
        sparsity_key="1:32",
    )

    #
    # Freeze everything except weights
    #

    #
    # 1:16 sparse training
    #
    su.freeze_mask(model, "1:32")
    su.freeze_non_masked_layers(model, "1:16")
    su.freeze_biases(model)
    su.freeze_batchnorm(model)
    snap = su.parameter_snapshot(model)
    snap = _run_phase(
            name="freeze 1:32 mask + everything else, train 1:16",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{BASE_DIR}/progressive_sparse_1_16.pt",
        comparison_path=f"{BASE_DIR}/progressive_sparse_1_16_comparison.txt",
        snapshot_before=snap,
        epochs=30,
        N=1, M=16,
        sparsity_key="1:16",
    )

    #
    # 1:8 sparse training
    #
    su.freeze_mask(model, "1:16")
    su.freeze_non_masked_layers(model, "1:8")
    su.freeze_biases(model)
    su.freeze_batchnorm(model)
    snap = su.parameter_snapshot(model)
    snap = _run_phase(
            name="freeze 1:32 mask + everything else, train 1:8",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{BASE_DIR}/progressive_sparse_1_8.pt",
        comparison_path=f"{BASE_DIR}/progressive_sparse_1_8_comparison.txt",
        snapshot_before=snap,
        epochs=30,
        N=1, M=8,
        sparsity_key="1:8",
    )

    #
    # 1:4 sparse training
    #
    su.freeze_mask(model, "1:8")
    su.freeze_non_masked_layers(model, "1:4")
    su.freeze_biases(model)
    su.freeze_batchnorm(model)
    snap = su.parameter_snapshot(model)
    snap = _run_phase(
            name="freeze 1:32 mask + everything else, train 1:4",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{BASE_DIR}/progressive_sparse_1_4.pt",
        comparison_path=f"{BASE_DIR}/progressive_sparse_1_4_comparison.txt",
        snapshot_before=snap,
        epochs=30,
        N=1, M=4,
        sparsity_key="1:4",
    )

    #
    # Dense final training - train only pruned positions
    #
    su.load_checkpoint(model, f"{BASE_DIR}/progressive_sparse_1_4.pt")
    su.freeze_mask(model, "1:4")
    su.freeze_biases(model)
    su.freeze_batchnorm(model)
    su.freeze_non_masked_layers(model, "dense")
    snap = su.parameter_snapshot(model)
    snap = _run_phase(
        name="Phase 3c — freeze all masks + everything else, train dense (pruned positions)",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{BASE_DIR}/progressive_sparse_dense_final.pt",
        comparison_path=f"{BASE_DIR}/progressive_sparse_dense_final_comparison.txt",
        snapshot_before=snap,
        epochs=30,
        N=0, M=0,
        sparsity_key="dense",
    )


if __name__ == "__main__":
    main()
