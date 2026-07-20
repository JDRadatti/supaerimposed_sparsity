"""
main_joint.py - Joint Training with Switchable BN (Slimmable Neural Networks)

Trains a single VGG16 model across all sparsity modes simultaneously using the
joint training approach from "Slimmable Neural Networks" (Yu et al., 2019).

Unlike the progressive pipeline in main.py, this trains all modes together:
each batch computes losses for all sparsity modes, accumulates gradients, and
performs a single shared optimizer step.  No parameters are frozen.
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
    evaluate,
)
import sparsity_modes as su
from sparsity_modes.logger import (
    setup_logger,
    info,
    build_overview,
)
from sparsity_modes.output import OutputLogger
from sparsity_modes.visualize import visualize

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torchvision\.datasets\.cifar",
)

EPOCHS   = 200
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sparsity modes to train jointly (most sparse → dense)
SPARSITY_KEYS = ["1:32", "1:16", "1:8", "1:4", "dense"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint sparse training (Slimmable Neural Networks approach)"
    )
    parser.add_argument("dir_name", help="Run directory name (under runs/)")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--cutout", type=int, default=16)
    args = parser.parse_args()

    # Set up output directory
    run_dir = f"runs/{args.dir_name}"
    os.makedirs(run_dir, exist_ok=True)

    # Logger setup
    log_file = f"{run_dir}/main.log"
    setup_logger(verbose=args.verbose, log_file=log_file)

    # Output logger for CSV artifacts
    out = OutputLogger(output_dir=run_dir, run_name=args.dir_name)

    #
    # 0. Init model and data
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
    # 1. Inject all sparse masks upfront
    #
    for key in SPARSITY_KEYS:
        if key == "dense":
            continue
        parts = key.split(":")
        N, M = int(parts[0]), int(parts[1])
        su.inject_switchable_mask(
            model=model,
            sparsity_key=key,
            N=N, M=M,
            conv_axis="out",
            linear_orientation="row",
        )
    info(f"[SETUP] Injected masks for {len(SPARSITY_KEYS) - 1} sparse modes.")

    #
    # 2. Joint training
    #
    info(f"[TRAIN] Starting joint training for {args.epochs} epochs "
         f"across modes: {SPARSITY_KEYS}")

    def on_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr):
        out.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr)

    su.train_joint(
        model=model,
        sparsity_keys=SPARSITY_KEYS,
        epochs=args.epochs,
        lr=args.lr,
        label_smoothing=args.label_smoothing,
        cutout=args.cutout,
        scheduler_fn=lambda opt: lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs),
        on_epoch=on_epoch,
    )

    #
    # 3. Final evaluation across all modes
    #
    eval_results = su.evaluate_all_sparsity_modes(model, val_loader, DEVICE)
    su.save_checkpoint(model, f"{checkpoint_dir}/joint_trained.pt", metrics={"eval": eval_results})
    out.log_eval_summary(eval_results)

    info("Final Evaluation Results:")
    for key, (loss, acc) in sorted(eval_results.items()):
        info(f"  {key:<10}  loss={loss:.4f}  acc={acc:.2f}%")

    #
    # 4. Visualization
    #
    visualize(model, out)

    out.finish_run()
    info(f"All outputs saved to: {out.run_dir}")


if __name__ == "__main__":
    main()
