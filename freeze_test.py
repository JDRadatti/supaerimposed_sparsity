"""
freeze_test.py
==============
Freeze all params after sparse training, train 10 epochs, then verify
that only BN buffers changed and restoring them recovers baseline accuracy.
"""

import torch
import torch.nn as nn

from vgg import build_model, create_dataloaders, evaluate as vgg_evaluate
import nm_sparsity_utils as su

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs"


def _bn_snapshot(model):
    return {
        name: {
            "running_mean": m.running_mean.detach().clone(),
            "running_var":  m.running_var.detach().clone(),
            "num_batches_tracked": m.num_batches_tracked.detach().clone(),
        }
        for name, m in model.named_modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    }


def _bn_restore(model, snapshot):
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            s = snapshot[name]
            m.running_mean.copy_(s["running_mean"])
            m.running_var.copy_(s["running_var"])
            m.num_batches_tracked.copy_(s["num_batches_tracked"])


if __name__ == "__main__":
    print("=" * 60)
    print("  FREEZE TEST")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    _, test_loader = create_dataloaders("./data", batch_size=128, num_workers=4)

    # Load Phase 2 — sparse-trained model
    print("\n" + "─" * 55)
    print("  1. Load Phase 2 (sparse) checkpoint")
    print("─" * 55)
    model = build_model().to(DEVICE)
    su.load_checkpoint(model, f"{BASE_DIR}/phase2_sparse.pt", device=DEVICE)
    model.eval()

    # --- Baseline: vgg.evaluate with sparse-trained BN buffers ---
    print("\n" + "─" * 55)
    print("  2. BASELINE — vgg.evaluate (no BN recalibration)")
    print("─" * 55)
    bl_loss, bl_acc = vgg_evaluate(model, test_loader, DEVICE)
    print(f"  dense: loss={bl_loss:.4f}  acc={bl_acc:.2f}%")

    # --- Save BN buffers from sparse training ---
    sparse_buffers_before = _bn_snapshot(model)

    # --- Freeze everything + train 10 epochs ---
    print("\n" + "─" * 55)
    print("  3. Freeze all params + train 10 epochs")
    print("─" * 55)
    su.freeze_all(model)
    su.train(
        model=model,
        epochs=10,
        N=0, M=0,
        freeze_params=list(model.parameters()),
    )
    model.eval()

    # --- Eval with drifted BN buffers ---
    print("\n" + "─" * 55)
    print("  4. DRIFTED BN — vgg.evaluate (no recalibration)")
    print("─" * 55)
    dr_loss, dr_acc = vgg_evaluate(model, test_loader, DEVICE)
    print(f"  dense: loss={dr_loss:.4f}  acc={dr_acc:.2f}%")

    # --- Restore original BN buffers ---
    _bn_restore(model, sparse_buffers_before)

    # --- Eval with restored BN buffers ---
    print("\n" + "─" * 55)
    print("  5. RESTORED BN — vgg.evaluate (no recalibration)")
    print("─" * 55)
    re_loss, re_acc = vgg_evaluate(model, test_loader, DEVICE)
    print(f"  dense: loss={re_loss:.4f}  acc={re_acc:.2f}%")

    # --- Results ---
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  {'':>15} {'acc':>8} {'loss':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8}")
    print(f"  {'baseline':>15} {bl_acc:>7.2f}% {bl_loss:>7.4f}")
    print(f"  {'drifted BN':>15} {dr_acc:>7.2f}% {dr_loss:>7.4f}")
    print(f"  {'restored BN':>15} {re_acc:>7.2f}% {re_loss:>7.4f}")
    print()

    match = abs(re_acc - bl_acc) < 0.01
    print(f"  Restored matches baseline?  {'YES' if match else 'NO'}  "
          f"(\u0394={abs(re_acc - bl_acc):.4f}%)")
    print(f"  BN drift impact:  {bl_acc:.2f}% \u2192 {dr_acc:.2f}%  "
          f"(\u0394={dr_acc - bl_acc:+.2f}%)")
    print()
    if match:
        print("  \u2713 Freeze mechanism verified: only BN buffers changed, weights intact.")
    else:
        print("  \u2717 Restored BN differs from baseline — something else moved!")
