"""
main_extended.py
================
Train dense, then train with 1:4 sparse mask, inject both 1:4 and 1:8 masks
on the same model so all sparsity modes can be compared.

Pipeline:
  1. Phase 1: Dense training (90 epochs)
  2. Phase 2: Inject 1:4 & 1:8 masks, train with 1:4 active (pruned frozen)
             Then evaluate all modes: dense, 1:4, 1:8
  3. Phase 3: Load Phase 2 checkpoint, train with 1:8 active (pruned frozen)
             Then evaluate all modes: dense, 1:4, 1:8
  4. Phase 4: Load Phase 2 checkpoint, train dense with 1:4 kept frozen
"""

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
import nm_sparsity_utils as su
from runner import DEVICE, BASE_DIR, read_eval, overview_section, run_phase

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torchvision\.datasets\.cifar",
)

EPOCHS   = 90


def main() -> None:

    #
    # 0. Init
    #
    model = build_model().to(DEVICE)
    train_loader, val_loader = create_dataloaders("./data", batch_size=128, num_workers=4)

    #
    # 1. Initial Dense Training
    #
    snap = su.parameter_snapshot(model)
    snap = run_phase(
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
    # 2. Sparse 1:4
    #
    su.inject_switchable_mask(
        model=model,
        sparsity_key="1:4",
        N=1, M=4,
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
    su.set_active_sparsity(model, "1:4")
    su.freeze_inverse_mask(model, "1:4")
    snap = su.parameter_snapshot(model)
    snap = run_phase(
        name="Sparse 1:4 (pruned frozen), 1:8 registered",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{BASE_DIR}/phase2_sparse_1_4.pt",
        comparison_path=f"{BASE_DIR}/phase2_sparse_1_4_comparison.txt",
        snapshot_before=snap,
        epochs=30,
        N=1, M=4,
        sparsity_key="1:4",
    )
    su.clear_all_frozen()

    #
    # 3. Sparse 1:8
    #
    su.set_active_sparsity(model, "1:8")
    su.freeze_inverse_mask(model, "1:8")
    snap = su.parameter_snapshot(model)
    snap = run_phase(
        name="Sparse 1:8 (pruned frozen, from 1:4 checkpoint)",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{BASE_DIR}/phase3_sparse_1_8.pt",
        comparison_path=f"{BASE_DIR}/phase3_sparse_1_8_comparison.txt",
        snapshot_before=snap,
        epochs=30,
        N=1, M=8,
        sparsity_key="1:8",
    )
    su.clear_all_frozen()

    # ------------------------------------------------------------------
    # Write overview.txt
    # ------------------------------------------------------------------
    overview = []
    overview.append("=" * 55)
    overview.append("  OVERVIEW — VGG16 on CIFAR-10 (1:4 + 1:8)")
    overview.append(f"  Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")
    overview.append("=" * 55)

    overview.extend(overview_section(
        "Phase 1 — Initial Dense Training",
        checkpoint=f"{BASE_DIR}/phase1_dense.pt",
        mask="None (dense)",
        frozen="None",
        eval_results=read_eval(f"{BASE_DIR}/phase1_dense.pt"),
    ))

    overview.extend(overview_section(
        "Phase 2 — Sparse 1:4 (pruned frozen, 1:8 also registered)",
        checkpoint=f"{BASE_DIR}/phase2_sparse_1_4.pt",
        mask="1:4 (active during training), 1:8 registered",
        frozen="Inverse 1:4 mask (pruned / mask==0 positions)",
        note="Both 1:4 and 1:8 masks registered. Training was with 1:4 active.",
        eval_results=read_eval(f"{BASE_DIR}/phase2_sparse_1_4.pt"),
    ))

    overview.extend(overview_section(
        "Phase 3 — Sparse 1:8 (pruned frozen, from 1:4 checkpoint)",
        checkpoint=f"{BASE_DIR}/phase3_sparse_1_8.pt",
        mask="1:8 (active during training), 1:4 also present",
        frozen="Inverse 1:8 mask (pruned / mask==0 positions)",
        note="Loaded from Phase 2, then trained with 1:8 active.",
        eval_results=read_eval(f"{BASE_DIR}/phase3_sparse_1_8.pt"),
    ))

    overview_path = f"{BASE_DIR}/overview_extended.txt"
    with open(overview_path, "w") as f:
        f.write("\n".join(overview))
        f.write("\n")
    print(f"\n[OVERVIEW] saved to {overview_path}")
    for line in overview:
        print(line)


if __name__ == "__main__":
    main()
