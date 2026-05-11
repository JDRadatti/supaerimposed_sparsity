"""
main.py
-------
Runs a number of experiments on superimposed weights evaluated with different
sparsities. 
"""
import contextlib
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vgg import (
    build_model,
    create_dataloaders,
    evaluate,
)
import nm_sparsity_utils as su

# Silence the NumPy VisibleDeprecationWarning that CIFAR triggers
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torchvision\.datasets\.cifar",
)

EPOCHS   = 30
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs"
os.makedirs(BASE_DIR, exist_ok=True)


def _read_eval(checkpoint_path: str) -> dict | None:
    if not os.path.exists(checkpoint_path):
        return None
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    return ckpt.get("metrics", {}).get("eval", None)


def _format_eval_line(key: str, loss: float, acc: float) -> str:
    return f"    {key:<10}  loss={loss:.4f}  acc={acc:.2f}%"


def _overview_section(
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
            lines.append(_format_eval_line(key, loss, acc))
    else:
        lines.append("  Results: (checkpoint not found)")
    return lines


def _run_phase(
    name: str,
    model: nn.Module,
    val_loader: DataLoader,
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
    eval_results = su.evaluate_all_sparsity_modes(model, val_loader, DEVICE)
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


def main() -> None:

    #
    # 0. Init
    #
    model = build_model().to(DEVICE)
    _, val_loader = create_dataloaders("./data", batch_size=128, num_workers=4)

    #
    # 1. Initial Training
    #
    snap = su.parameter_snapshot(model)
    snap = _run_phase(
        name="Dense (1:1)",
        model=model,
        val_loader=val_loader,
        checkpoint_path=f"{BASE_DIR}/phase1_dense.pt",
        comparison_path=f"{BASE_DIR}/phase1_dense_comparison.txt",
        snapshot_before=snap,
        epochs=EPOCHS,
        N=0, M=0,
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
    snap = _run_phase(
        name="Sparse 1:4 (pruned positions frozen)",
        model=model,
        val_loader=val_loader,
        checkpoint_path=f"{BASE_DIR}/phase2_sparse.pt",
        comparison_path=f"{BASE_DIR}/phase2_sparse_comparison.txt",
        snapshot_before=snap,
        epochs=EPOCHS,
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
    snap = _run_phase(
        name="Phase 3a — freeze 1:4 mask, train pruned (dense)",
        model=model,
        val_loader=val_loader,
        checkpoint_path=f"{BASE_DIR}/phase3a_freeze_mask.pt",
        comparison_path=f"{BASE_DIR}/phase3a_freeze_mask_comparison.txt",
        snapshot_before=snap,
        epochs=10,
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
    snap = _run_phase(
        name="Phase 3b — freeze mask + biases, train pruned (dense)",
        model=model,
        val_loader=val_loader,
        checkpoint_path=f"{BASE_DIR}/phase3b_freeze_mask_biases.pt",
        comparison_path=f"{BASE_DIR}/phase3b_freeze_mask_biases_comparison.txt",
        snapshot_before=snap,
        epochs=10,
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
    snap = _run_phase(
        name="Phase 3c — freeze mask + everything else, train pruned (dense)",
        model=model,
        val_loader=val_loader,
        checkpoint_path=f"{BASE_DIR}/phase3c_freeze_everything_else.pt",
        comparison_path=f"{BASE_DIR}/phase3c_freeze_everything_else_comparison.txt",
        snapshot_before=snap,
        epochs=10,
        N=0, M=0,
        sparsity_key="1:4",
    )

    # ------------------------------------------------------------------
    # Phase 3d: weight transplant experiment
    #
    # 1. Load Phase 3a checkpoint & evaluate densely
    # 2. Load Phase 2 params
    # 3. Transplant Phase 3a's .weight tensors into the Phase 2 copy
    # 4. Evaluate the hybrid model sparsely
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Phase 3d — transplant Phase 3a weights into Phase 2 skeleton")
    print("=" * 60)

    # 1. Load Phase 3a checkpoint & evaluate densely
    su.load_checkpoint(model, f"{BASE_DIR}/phase3a_freeze_mask.pt")
    loss_dense, acc_dense = evaluate(model, val_loader, DEVICE)
    print(f"\n  Phase 3a (dense eval): loss={loss_dense:.4f}  acc={acc_dense:.2f}%")

    # 2. Load Phase 2 params into a fresh model
    model_hybrid = build_model().to(DEVICE)
    su.inject_switchable_mask(model_hybrid, "1:4", N=1, M=4,
                              conv_axis="out", linear_orientation="row")
    su.load_checkpoint(model_hybrid, f"{BASE_DIR}/phase2_sparse.pt")
    state_phase2 = model_hybrid.state_dict()

    # 3. Transplant Phase 3a's Conv2d/Linear .weight tensors only
    #    (skip BatchNorm gamma — it was trained for dense activations
    #     in Phase 3a and would corrupt sparse evaluation)
    state_3a = model.state_dict()
    for key in state_phase2:
        if key.endswith(".weight"):
            mod_name = key.rsplit(".weight", 1)[0]
            if not isinstance(model_hybrid.get_submodule(mod_name), (nn.Conv2d, nn.Linear)):
                continue
            state_phase2[key].copy_(state_3a[key])

    # 4. Evaluate hybrid sparsely
    su.set_active_sparsity(model_hybrid, "1:4")
    loss_sparse, acc_sparse = evaluate(model_hybrid, val_loader, DEVICE)
    print(f"  Hybrid   (sparse eval): loss={loss_sparse:.4f}  acc={acc_sparse:.2f}%")

    print(f"\n  {'=' * 50}")
    print(f"  Phase 3d — weight transplant")
    print(f"  {'=' * 50}")
    print(f"  {'Config':<40} {'Loss':<10} {'Acc':<10}")
    print(f"  {'-' * 60}")
    print(f"  {'Phase 3a model (dense eval)':<40} {loss_dense:<10.4f} {acc_dense:<10.2f}")
    print(f"  {'Phase 2 skeleton + 3a weights (sparse eval)':<40} {loss_sparse:<10.4f} {acc_sparse:<10.2f}")
    print(f"  {'=' * 50}")
    with open(f"{BASE_DIR}/phase3d_transplant.txt", "w") as f:
        f.write(f"Phase 3d — weight transplant\n")
        f.write(f"{'Config':<40} {'Loss':<10} {'Acc':<10}\n")
        f.write(f"{'Phase 3a model (dense eval)':<40} {loss_dense:<10.4f} {acc_dense:<10.2f}\n")
        f.write(f"{'Phase 2 skeleton + 3a weights (sparse eval)':<40} {loss_sparse:<10.4f} {acc_sparse:<10.2f}\n")
    print(f"  [RESULT] saved to {BASE_DIR}/phase3d_transplant.txt")

    # ------------------------------------------------------------------
    # Write overview.txt
    # ------------------------------------------------------------------
    overview = []
    overview.append("=" * 55)
    overview.append("  OVERVIEW — VGG16 on CIFAR-10")
    overview.append(f"  Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")
    overview.append("=" * 55)

    overview.extend(_overview_section(
        "Phase 1 — Initial Dense Training",
        checkpoint=f"{BASE_DIR}/phase1_dense.pt",
        mask="None (dense)",
        frozen="None",
        eval_results=_read_eval(f"{BASE_DIR}/phase1_dense.pt"),
    ))

    overview.extend(_overview_section(
        "Phase 2 — Sparse 1:4 Training (pruned positions frozen)",
        checkpoint=f"{BASE_DIR}/phase2_sparse.pt",
        mask="1:4 (active during training)",
        frozen="Inverse mask (pruned / mask==0 positions)",
        eval_results=_read_eval(f"{BASE_DIR}/phase2_sparse.pt"),
    ))

    overview.extend(_overview_section(
        "Phase 3a — Freeze mask, train pruned (dense)",
        checkpoint=f"{BASE_DIR}/phase3a_freeze_mask.pt",
        mask="1:4 (eval only; trained in dense mode)",
        frozen="Mask kept positions (mask==1)",
        note="BN statistics were updated for dense activations → sparse eval is poor.",
        eval_results=_read_eval(f"{BASE_DIR}/phase3a_freeze_mask.pt"),
    ))

    overview.extend(_overview_section(
        "Phase 3b — Freeze mask + biases, train pruned (dense)",
        checkpoint=f"{BASE_DIR}/phase3b_freeze_mask_biases.pt",
        mask="1:4 (eval only; trained in dense mode)",
        frozen="Mask kept positions (mask==1) + all biases",
        eval_results=_read_eval(f"{BASE_DIR}/phase3b_freeze_mask_biases.pt"),
    ))

    overview.extend(_overview_section(
        "Phase 3c — Freeze mask + everything else, train pruned (dense)",
        checkpoint=f"{BASE_DIR}/phase3c_freeze_everything_else.pt",
        mask="1:4 (eval only; trained in dense mode)",
        frozen="Mask kept positions (mask==1) + biases + BN params + non-masked layers",
        eval_results=_read_eval(f"{BASE_DIR}/phase3c_freeze_everything_else.pt"),
    ))

    overview.extend(_overview_section(
        "Phase 3d — Weight transplant (Phase 3a weights → Phase 2 skeleton)",
        checkpoint=f"{BASE_DIR}/phase3d_transplant.txt",
        mask="1:4 (evaluated sparsely with Phase 2 BN stats)",
        frozen="N/A (eval only — no training)",
        note="Transplanted Conv2d/Linear .weight only; kept Phase 2 BN gamma/beta & running stats.",
    ))
    overview.append(f"    {'Phase 3a model (dense eval)':<40} {loss_dense:<10.4f} {acc_dense:<10.2f}")
    overview.append(f"    {'Phase 2 skeleton + 3a weights (sparse eval)':<40} {loss_sparse:<10.4f} {acc_sparse:<10.2f}")

    overview_path = f"{BASE_DIR}/overview.txt"
    with open(overview_path, "w") as f:
        f.write("\n".join(overview))
        f.write("\n")
    print(f"\n[OVERVIEW] saved to {overview_path}")
    for line in overview:
        print(line)


if __name__ == "__main__":
    main()
