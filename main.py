"""
<<<<<<< HEAD
main.py - Progressive Sparse Training with Multiple Sparsity Modes

Trains a model progressively with nested sparsity modes (1:32, 1:16, 1:8, 1:4),
freezing previously-trained modes at each stage.

Uses the new SparsityModel API with declarative profiles.
=======
main.py
-------
Runs a number of experiments on superimposed weights evaluated with different
sparsities.
>>>>>>> b2ba4e1b8e3f02280bff41b00b9ba4949826d609
"""
import os

import torch
from torch.optim import lr_scheduler

from vgg import build_model, create_dataloaders
from sparsity_modes import (
    SparsityModel,
    ModeProfile,
    LayerSpec,
    FreezeProfile,
    TrainingProfile,
    OutputOptions,
)
<<<<<<< HEAD
=======
import nm_sparsity_utils as su
from runner import DEVICE, BASE_DIR, read_eval, overview_section, run_phase
>>>>>>> b2ba4e1b8e3f02280bff41b00b9ba4949826d609

# Silence the NumPy VisibleDeprecationWarning that CIFAR triggers
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torchvision\.datasets\.cifar",
)

<<<<<<< HEAD
EPOCHS = 90
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 
# PROFILES
# 

# Mode profiles (per-layer sparsity assignments)

MODE_DENSE = ModeProfile("dense", [
    LayerSpec(0, None, "dense"),
])

MODE_1_32 = ModeProfile("sparse_1_32", [
    LayerSpec(0, 1, "dense"),
    LayerSpec(1, -1, (1, 32)),
    LayerSpec(-1, None, "dense"),
])

MODE_1_16 = ModeProfile("sparse_1_16", [
    LayerSpec(0, 1, "dense"),
    LayerSpec(1, -1, (1, 16)),
    LayerSpec(-1, None, "dense"),
])

MODE_1_8 = ModeProfile("sparse_1_8", [
    LayerSpec(0, 1, "dense"),
    LayerSpec(1, -1, (1, 8)),
    LayerSpec(-1, None, "dense"),
])

MODE_1_4 = ModeProfile("sparse_1_4", [
    LayerSpec(0, 1, "dense"),
    LayerSpec(1, -1, (1, 4)),
    LayerSpec(-1, None, "dense"),
])

ALL_MODES = [MODE_DENSE, MODE_1_32, MODE_1_16, MODE_1_8, MODE_1_4]

# Freeze profiles

FREEZE_BIASES = FreezeProfile("freeze_biases", "biases")
FREEZE_BN = FreezeProfile("freeze_bn", "batchnorm")

FREEZE_INVERSE_1_32 = FreezeProfile("freeze_inverse_1_32", "inverse_mask", mode_profile="sparse_1_32")

FREEZE_KEPT_1_32 = FreezeProfile("freeze_kept_1_32", "mask", mode_profile="sparse_1_32")
FREEZE_KEPT_1_16 = FreezeProfile("freeze_kept_1_16", "mask", mode_profile="sparse_1_16")
FREEZE_KEPT_1_8 = FreezeProfile("freeze_kept_1_8", "mask", mode_profile="sparse_1_8")
FREEZE_KEPT_1_4 = FreezeProfile("freeze_kept_1_4", "mask", mode_profile="sparse_1_4")

FREEZE_NON_MASKED_1_16 = FreezeProfile("freeze_non_masked_1_16", "non_masked", mode_profile="sparse_1_16")
FREEZE_NON_MASKED_1_8 = FreezeProfile("freeze_non_masked_1_8", "non_masked", mode_profile="sparse_1_8")
FREEZE_NON_MASKED_1_4 = FreezeProfile("freeze_non_masked_1_4", "non_masked", mode_profile="sparse_1_4")
FREEZE_NON_MASKED_DENSE = FreezeProfile("freeze_non_masked_dense", "non_masked", mode_profile="dense")

ALL_FREEZE = [
    FREEZE_BIASES, FREEZE_BN,
    FREEZE_INVERSE_1_32,
    FREEZE_KEPT_1_32, FREEZE_KEPT_1_16, FREEZE_KEPT_1_8, FREEZE_KEPT_1_4,
    FREEZE_NON_MASKED_1_16, FREEZE_NON_MASKED_1_8, FREEZE_NON_MASKED_1_4, FREEZE_NON_MASKED_DENSE,
]

# Training profiles

TRAIN_DENSE = TrainingProfile(
    name="dense_train",
    epochs=EPOCHS,
    optimizer_fn=lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4),
    scheduler_fn=lambda opt: lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS),
    label_smoothing=0.1,
    cutout=16,
)

TRAIN_SPARSE = TrainingProfile(
    name="sparse_train",
    epochs=30,
)

ALL_TRAINING = [TRAIN_DENSE, TRAIN_SPARSE]


#
# MAIN
#
=======
EPOCHS   = 90

>>>>>>> b2ba4e1b8e3f02280bff41b00b9ba4949826d609

def main() -> None:

    # 0. Build model + data
    model = build_model().to(DEVICE)
    train_loader, val_loader = create_dataloaders("./data", batch_size=128, num_workers=4)

    # 1. Create SparsityModel with all profiles declared upfront
    sparse = SparsityModel(
        model,
        mode_profiles=ALL_MODES,
        freeze_profiles=ALL_FREEZE,
        training_profiles=ALL_TRAINING,
        output_options=OutputOptions(output_dir="./runs", filename_prefix="progressive"),
    )

    print(sparse)
    print()

    #
    # Phase 1: Dense training
    #
<<<<<<< HEAD
    print(f"\n{'=' * 60}")
    print("  Phase 1: Dense Training")
    print(f"{'=' * 60}")

    sparse.train(
        mode="dense",
        freeze=None,
        training="dense_train",
        train_loader=train_loader,
=======
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
>>>>>>> b2ba4e1b8e3f02280bff41b00b9ba4949826d609
        val_loader=val_loader,
    )

    #
    # Phase 2: 1:32 sparse training (pruned positions frozen)
    #
    print(f"\n{'=' * 60}")
    print("  Phase 2: Sparse 1:32 Training")
    print(f"{'=' * 60}")

<<<<<<< HEAD
    sparse.train(
        mode="sparse_1_32",
        freeze=[FREEZE_INVERSE_1_32.name],
        training="sparse_train",
        train_loader=train_loader,
=======
    #
    # 3a. Freeze only 1:4 mask values, train inverse mask
    #
    su.load_checkpoint(model, f"{BASE_DIR}/phase2_sparse.pt")
    su.freeze_mask(model, "1:4")
    snap = su.parameter_snapshot(model)
    snap = run_phase(
        name="Phase 3a — freeze 1:4 mask, train pruned (dense)",
        model=model,
>>>>>>> b2ba4e1b8e3f02280bff41b00b9ba4949826d609
        val_loader=val_loader,
    )

    #
    # Phase 3: 1:16 training (freeze 1:32 kept + biases + BN + non-masked)
    #
<<<<<<< HEAD
    print(f"\n{'=' * 60}")
    print("  Phase 3: Sparse 1:16 Training")
    print(f"{'=' * 60}")

    sparse.train(
        mode="sparse_1_16",
        freeze=[
            FREEZE_KEPT_1_32.name,
            FREEZE_NON_MASKED_1_16.name,
            FREEZE_BIASES.name,
            FREEZE_BN.name,
        ],
        training="sparse_train",
        train_loader=train_loader,
=======
    su.load_checkpoint(model, f"{BASE_DIR}/phase2_sparse.pt")
    su.freeze_mask(model, "1:4")
    su.freeze_biases(model)
    snap = su.parameter_snapshot(model)
    snap = run_phase(
        name="Phase 3b — freeze mask + biases, train pruned (dense)",
        model=model,
>>>>>>> b2ba4e1b8e3f02280bff41b00b9ba4949826d609
        val_loader=val_loader,
    )

    #
    # Phase 4: 1:8 training (freeze 1:16 kept + biases + BN + non-masked)
    #
<<<<<<< HEAD
    print(f"\n{'=' * 60}")
    print("  Phase 4: Sparse 1:8 Training")
    print(f"{'=' * 60}")

    sparse.train(
        mode="sparse_1_8",
        freeze=[
            FREEZE_KEPT_1_16.name,
            FREEZE_NON_MASKED_1_8.name,
            FREEZE_BIASES.name,
            FREEZE_BN.name,
        ],
        training="sparse_train",
        train_loader=train_loader,
=======
    su.load_checkpoint(model, f"{BASE_DIR}/phase2_sparse.pt")
    su.freeze_mask(model, "1:4")
    su.freeze_biases(model)
    su.freeze_batchnorm(model)
    su.freeze_non_masked_layers(model, "1:4")
    snap = su.parameter_snapshot(model)
    snap = run_phase(
        name="Phase 3c — freeze mask + everything else, train pruned (dense)",
        model=model,
>>>>>>> b2ba4e1b8e3f02280bff41b00b9ba4949826d609
        val_loader=val_loader,
    )

    #
    # Phase 5: 1:4 training (freeze 1:8 kept + biases + BN + non-masked)
    #
    print(f"\n{'=' * 60}")
    print("  Phase 5: Sparse 1:4 Training")
    print(f"{'=' * 60}")

<<<<<<< HEAD
    sparse.train(
        mode="sparse_1_4",
        freeze=[
            FREEZE_KEPT_1_8.name,
            FREEZE_NON_MASKED_1_4.name,
            FREEZE_BIASES.name,
            FREEZE_BN.name,
        ],
        training="sparse_train",
        train_loader=train_loader,
        val_loader=val_loader,
    )

    #
    # Phase 6: Dense final - train pruned positions only
    #
    print(f"\n{'=' * 60}")
    print("  Phase 6: Dense Final (pruned positions only)")
    print(f"{'=' * 60}")

    sparse.train(
        mode="sparse_1_4",
        freeze=[
            FREEZE_KEPT_1_4.name,
            FREEZE_NON_MASKED_DENSE.name,
            FREEZE_BIASES.name,
            FREEZE_BN.name,
        ],
        training="sparse_train",
        train_loader=train_loader,
        val_loader=val_loader,
    )

    #
    # Final evaluation across all modes
    #
    print(f"\n{'=' * 60}")
    print("  Final Evaluation")
    print(f"{'=' * 60}")

    results = sparse.evaluate(val_loader=val_loader)
=======
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
>>>>>>> b2ba4e1b8e3f02280bff41b00b9ba4949826d609

    #
    # Visualization
    #
    sparse.visualize(first_n=50)

    print(f"\nDone. Outputs in: {sparse._logger.run_dir}")


if __name__ == "__main__":
    main()
