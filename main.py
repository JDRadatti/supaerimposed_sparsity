"""
main.py - Progressive Sparse Training with Multiple Sparsity Modes

Trains a model progressively with nested sparsity modes (1:32, 1:16, 1:8, 1:4),
freezing previously-trained modes at each stage.
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from vgg import (
    build_model,
    create_dataloaders,
)
import sparsity_modes as su
from sparsity_modes.logger import (
    setup_logger,
    info,
    debug,
    build_overview,
    build_phase_summary,
)
from sparsity_modes.output import OutputLogger
from sparsity_modes.visualize import visualize, visualize_diff

# Silence the NumPy VisibleDeprecationWarning that CIFAR triggers
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torchvision\.datasets\.cifar",
)

EPOCHS   = 200
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_phase(
    name: str,
    model: nn.Module,
    val_loader: DataLoader,
    train_loader: DataLoader,
    checkpoint_path: str,
    snapshot_before: dict | None,
    epochs: int,
    N: int,
    M: int,
    freeze_params: list | None = None,
    sparsity_key: str | None = None,
    train_kwargs: dict | None = None,
    skip_if_exists: bool = True,
    logger: OutputLogger | None = None,
) -> dict:
    """Run a training phase with snapshot + eval.

    If *skip_if_exists* is ``True`` and *checkpoint_path* already exists,
    the checkpoint is loaded and training is skipped (the phase is assumed
    to have been completed in a previous run).

    Returns the post-training (or post-load) snapshot.
    """
    if skip_if_exists and os.path.exists(checkpoint_path):
        debug(f"  [CACHED] Checkpoint exists, loading and skipping training.")
        su.load_checkpoint(model, checkpoint_path)
        return su.parameter_snapshot(model)

    # Build on_epoch callback if logger is available
    on_epoch = None
    if logger is not None:
        def on_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr):
            logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr)

    su.train(
        model=model,
        epochs=epochs,
        N=N,
        M=M,
        freeze_params=freeze_params,
        on_epoch=on_epoch,
        **(train_kwargs or {}),
    )
    eval_results = su.evaluate_all_sparsity_modes(model, val_loader, DEVICE)
    su.save_checkpoint(model, checkpoint_path, metrics={"eval": eval_results})

    if logger is not None:
        logger.log_eval_summary(eval_results)

    info("  Eval Results:")
    for key, (loss, acc) in sorted(eval_results.items()):
        info(f"    {key:<10}  loss={loss:.4f}  acc={acc:.2f}%")

    snap_after = su.parameter_snapshot(model)
    info("-" * 60)
    return snap_after


def main() -> None:
    parser = argparse.ArgumentParser(description="Progressive sparse training")
    parser.add_argument("dir_name", help="Run directory name (under runs/)")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_args()

    # Set up output directory
    run_dir = f"runs/{args.dir_name}"
    os.makedirs(run_dir, exist_ok=True)

    # Logger setup — log file goes into the run directory
    log_file = f"{run_dir}/main.log"
    setup_logger(verbose=args.verbose, log_file=log_file)

    # Output logger for CSV artifacts
    out = OutputLogger(output_dir=run_dir, run_name=args.dir_name)

    #
    # 0. Init
    #
    model = build_model().to(DEVICE)
    train_loader, val_loader = create_dataloaders("./data", batch_size=128, num_workers=4)

    out.start_run()

    # Print and save overview
    overview = build_overview(model, out.run_dir, out.timestamp, out.commit, out.branch)
    info(overview)
    out.write_overview(overview)

    checkpoint_dir = out.run_dir

    #
    # 1. Initial Training
    #
    info(build_phase_summary(
        name="Phase 1: Dense Training",
        model=model,
        epochs=EPOCHS,
        N=0, M=0,
        lr=0.1,
        label_smoothing=0.1,
        cutout=16,
        scheduler_desc=f"CosineAnnealingLR(T_max={EPOCHS})",
    ))
    snap = su.parameter_snapshot(model)
    snap = _run_phase(
        name="Phase 1: Dense Training",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{checkpoint_dir}/phase1_dense.pt",
        snapshot_before=snap,
        epochs=EPOCHS,
        N=0, M=0,
        train_kwargs=dict(
            lr=0.1,
            scheduler_fn=lambda opt: lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS),
            label_smoothing=0.1,
            cutout=16,
        ),
        logger=out,
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
    info(build_phase_summary(
        name="Phase 2: Sparse 1:32 Training",
        model=model,
        epochs=30,
        N=1, M=32,
        lr=0.1,
    ))
    snap_before = su.parameter_snapshot(model)
    snap_after = _run_phase(
        name="Phase 2: Sparse 1:32 Training",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{checkpoint_dir}/progressive_sparse_1_32.pt",
        snapshot_before=snap_before,
        epochs=30,
        N=1, M=32,
        sparsity_key="1:32",
        logger=out,
    )
    visualize_diff(snap_before, snap_after, out, "1:32", model=model)

    #
    # 1:16 sparse training
    #
    su.freeze_mask(model, "1:32")
    su.freeze_non_masked_layers(model, "1:16")
    su.freeze_biases(model)
    su.freeze_batchnorm(model)
    info(build_phase_summary(
        name="Phase 3: Sparse 1:16 Training",
        model=model,
        epochs=30,
        N=1, M=16,
        lr=0.1,
    ))
    snap_before = su.parameter_snapshot(model)
    snap_after = _run_phase(
        name="Phase 3: Sparse 1:16 Training",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{checkpoint_dir}/progressive_sparse_1_16.pt",
        snapshot_before=snap_before,
        epochs=30,
        N=1, M=16,
        sparsity_key="1:16",
        logger=out,
    )
    visualize_diff(snap_before, snap_after, out, "1:16", model=model)

    #
    # 1:8 sparse training
    #
    su.freeze_mask(model, "1:16")
    su.freeze_non_masked_layers(model, "1:8")
    su.freeze_biases(model)
    su.freeze_batchnorm(model)
    info(build_phase_summary(
        name="Phase 4: Sparse 1:8 Training",
        model=model,
        epochs=30,
        N=1, M=8,
        lr=0.1,
    ))
    snap_before = su.parameter_snapshot(model)
    snap_after = _run_phase(
        name="Phase 4: Sparse 1:8 Training",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{checkpoint_dir}/progressive_sparse_1_8.pt",
        snapshot_before=snap_before,
        epochs=30,
        N=1, M=8,
        sparsity_key="1:8",
        logger=out,
    )
    visualize_diff(snap_before, snap_after, out, "1:8", model=model)

    #
    # 1:4 sparse training
    #
    su.freeze_mask(model, "1:8")
    su.freeze_non_masked_layers(model, "1:4")
    su.freeze_biases(model)
    su.freeze_batchnorm(model)
    info(build_phase_summary(
        name="Phase 5: Sparse 1:4 Training",
        model=model,
        epochs=30,
        N=1, M=4,
        lr=0.1,
    ))
    snap_before = su.parameter_snapshot(model)
    snap_after = _run_phase(
        name="Phase 5: Sparse 1:4 Training",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{checkpoint_dir}/progressive_sparse_1_4.pt",
        snapshot_before=snap_before,
        epochs=30,
        N=1, M=4,
        sparsity_key="1:4",
        logger=out,
    )
    visualize_diff(snap_before, snap_after, out, "1:4", model=model)

    #
    # Dense final training - train only pruned positions
    #
    su.load_checkpoint(model, f"{checkpoint_dir}/progressive_sparse_1_4.pt")
    su.freeze_mask(model, "1:4")
    su.freeze_biases(model)
    su.freeze_batchnorm(model)
    su.freeze_non_masked_layers(model, "dense")
    info(build_phase_summary(
        name="Phase 6: Dense Final Training",
        model=model,
        epochs=30,
        N=0, M=0,
        lr=0.1,
    ))
    snap_before = su.parameter_snapshot(model)
    snap_after = _run_phase(
        name="Phase 6: Dense Final Training",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{checkpoint_dir}/progressive_sparse_dense_final.pt",
        snapshot_before=snap_before,
        epochs=30,
        N=0, M=0,
        sparsity_key="dense",
        logger=out,
    )
    visualize_diff(snap_before, snap_after, out, "dense", model=model)

    #
    # Final weight visualization
    #
    visualize(model, out)

    out.finish_run()
    info(f"All outputs saved to: {out.run_dir}")


if __name__ == "__main__":
    main()
