"""
main.py
-------
Runs a number of experiments on superimposed weights evaluated with different
sparsities.
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

# Silence the NumPy VisibleDeprecationWarning that CIFAR triggers
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
    # 1. Initial Training
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
    # 2. Sparse Training
    #
    su.inject_switchable_mask(
        model=model,
        sparsity_key="1:4",
        N=1, M=4,
        conv_axis="out",
        linear_orientation="row",
    )
    su.freeze_inverse_mask(model, "1:4")
    snap = su.parameter_snapshot(model)
    snap = run_phase(
        name="Sparse 1:4 (pruned positions frozen)",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{BASE_DIR}/phase2_sparse.pt",
        comparison_path=f"{BASE_DIR}/phase2_sparse_comparison.txt",
        snapshot_before=snap,
        epochs=30,
        N=1, M=4,
        sparsity_key="1:4",
    )

    # ------------------------------------------------------------------
    # Phase 3 experiments — each starts fresh from Phase 2 checkpoint
    # ------------------------------------------------------------------

    #
    # 3a. Freeze only 1:4 mask values, train inverse mask
    #
    su.load_checkpoint(model, f"{BASE_DIR}/phase2_sparse.pt")
    su.freeze_mask(model, "1:4")
    snap = su.parameter_snapshot(model)
    snap = run_phase(
        name="Phase 3a — freeze 1:4 mask, train pruned (dense)",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{BASE_DIR}/phase3a_freeze_mask.pt",
        comparison_path=f"{BASE_DIR}/phase3a_freeze_mask_comparison.txt",
        snapshot_before=snap,
        epochs=30,
        N=0, M=0,
        sparsity_key="1:4",
    )

    #
    # 3b. Freeze mask + biases → train pruned (dense)
    #
    su.load_checkpoint(model, f"{BASE_DIR}/phase2_sparse.pt")
    su.freeze_mask(model, "1:4")
    su.freeze_biases(model)
    snap = su.parameter_snapshot(model)
    snap = run_phase(
        name="Phase 3b — freeze mask + biases, train pruned (dense)",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{BASE_DIR}/phase3b_freeze_mask_biases.pt",
        comparison_path=f"{BASE_DIR}/phase3b_freeze_mask_biases_comparison.txt",
        snapshot_before=snap,
        epochs=30,
        N=0, M=0,
        sparsity_key="1:4",
    )

    #
    # 3c. Freeze mask + everything else (including batchnorm), train pruned (dense)
    #
    su.load_checkpoint(model, f"{BASE_DIR}/phase2_sparse.pt")
    su.freeze_mask(model, "1:4")
    su.freeze_biases(model)
    su.freeze_batchnorm(model)
    su.freeze_non_masked_layers(model, "1:4")
    snap = su.parameter_snapshot(model)
    snap = run_phase(
        name="Phase 3c — freeze mask + everything else, train pruned (dense)",
        model=model,
        val_loader=val_loader,
        train_loader=train_loader,
        checkpoint_path=f"{BASE_DIR}/phase3c_freeze_everything_else.pt",
        comparison_path=f"{BASE_DIR}/phase3c_freeze_everything_else_comparison.txt",
        snapshot_before=snap,
        epochs=30,
        N=0, M=0,
        sparsity_key="1:4",
    )

    # ------------------------------------------------------------------
    # Write overview.txt
    # ------------------------------------------------------------------
    overview = []
    overview.append("=" * 55)
    overview.append("  OVERVIEW — VGG16 on CIFAR-10")
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
        "Phase 2 — Sparse 1:4 Training (pruned positions frozen)",
        checkpoint=f"{BASE_DIR}/phase2_sparse.pt",
        mask="1:4 (active during training)",
        frozen="Inverse mask (pruned / mask==0 positions)",
        eval_results=read_eval(f"{BASE_DIR}/phase2_sparse.pt"),
    ))

    overview.extend(overview_section(
        "Phase 3a — Freeze mask, train pruned (dense)",
        checkpoint=f"{BASE_DIR}/phase3a_freeze_mask.pt",
        mask="1:4 (eval only; trained in dense mode)",
        frozen="Mask kept positions (mask==1)",
        eval_results=read_eval(f"{BASE_DIR}/phase3a_freeze_mask.pt"),
    ))

    overview.extend(overview_section(
        "Phase 3b — Freeze mask + biases, train pruned (dense)",
        checkpoint=f"{BASE_DIR}/phase3b_freeze_mask_biases.pt",
        mask="1:4 (eval only; trained in dense mode)",
        frozen="Mask kept positions (mask==1) + all biases",
        eval_results=read_eval(f"{BASE_DIR}/phase3b_freeze_mask_biases.pt"),
    ))

    overview.extend(overview_section(
        "Phase 3c — Freeze mask + everything else, train pruned (dense)",
        checkpoint=f"{BASE_DIR}/phase3c_freeze_everything_else.pt",
        mask="1:4 (eval only; trained in dense mode)",
        frozen="Mask kept positions (mask==1) + biases + BN params + non-masked layers",
        eval_results=read_eval(f"{BASE_DIR}/phase3c_freeze_everything_else.pt"),
    ))

    overview_path = f"{BASE_DIR}/overview.txt"
    with open(overview_path, "w") as f:
        f.write("\n".join(overview))
        f.write("\n")
    print(f"\n[OVERVIEW] saved to {overview_path}")
    for line in overview:
        print(line)


if __name__ == "__main__":
    main()
