"""
baseline_switchable.py
======================
Train a dense model, clone it, train sparse copies using the switchable mask
(nm_sparsity_utils) approach instead of permanent prune_nm.

Pipeline:
  1. Train dense (60 epochs)
  2. Clone -> inject 1:4  mask -> freeze_inverse_mask -> su.train 30 epochs
  3. Clone -> inject 1:8  mask -> freeze_inverse_mask -> su.train 30 epochs
  4. Clone -> inject 1:16 mask -> freeze_inverse_mask -> su.train 30 epochs
  5. Clone -> inject 1:32 mask -> freeze_inverse_mask -> su.train 30 epochs
  6. Summary

Each sparse clone is evaluated via evaluate_all_sparsity_modes which
recalibrates BN for both its sparse and dense modes, showing how the
weights evolved during sparse training (dense column reveals weight drift).

============================================================
  RESULTS
============================================================
                            Dense        1:4        1:8       1:16       1:32
  -------------------- ---------- ---------- ---------- ---------- ----------
        Dense baseline     92.25%          —          —          —          —
       After 1:4 train     89.91%     90.91%          —          —          —
       After 1:8 train     88.62%          —     90.34%          —          —
      After 1:16 train     84.67%          —          —     88.94%          —
      After 1:32 train     72.48%          —          —          —     87.35%

  (Dense column evaluates the sparse clone without its mask — reveals weight drift.)

"""

import copy
import warnings

import torch
from vgg import build_model, create_dataloaders, evaluate as vgg_evaluate, train as vgg_train
import nm_sparsity_utils as su

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.datasets\.cifar")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DENSE_EPOCHS = 60
SPARSE_EPOCHS = 30

SPARSITYS_CFG = [
    ("1:4",  1,  4),
    ("1:8",  1,  8),
    ("1:16", 1, 16),
    ("1:32", 1, 32),
]


def check_sparsity_switchable(model, tag=""):
    print(f"\n  [SPARSITY CHECK{(' ' + tag) if tag else ''}]")
    found = False
    for name, m in model.named_modules():
        if hasattr(m, 'nm_masks') and m.nm_masks:
            for key, mask in m.nm_masks.items():
                if mask is None:
                    continue
                found = True
                total = mask.numel()
                zeros = (mask == 0).sum().item()
                pct = 100.0 * zeros / total
                print(f"    {name} [{key}]: {zeros}/{total} zero ({pct:.1f}%)")
    if not found:
        print(f"    (no masked layers)")


if __name__ == "__main__":
    print("=" * 60)
    print("  BASELINE — switchable mask (inject + freeze_inverse + su.train)")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    train_loader, val_loader = create_dataloaders("./data", batch_size=128, num_workers=4)

    # ------------------------------------------------------------------
    # 1. Train dense for 60 epochs
    # ------------------------------------------------------------------
    print("\n" + "-" * 55)
    print(f"  1. Train dense ({DENSE_EPOCHS} epochs)")
    print("-" * 55)
    dense_model = build_model().to(DEVICE)
    vgg_train(dense_model, epochs=DENSE_EPOCHS)
    dense_model.eval()

    state_dict = copy.deepcopy(dense_model.state_dict())

    print(f"\n  --- dense ---")
    check_sparsity_switchable(dense_model, "dense")
    loss, acc = vgg_evaluate(dense_model, val_loader, DEVICE)
    print(f"  dense:  loss={loss:.4f}  acc={acc:.2f}%")
    dense_result = (loss, acc)

    # ------------------------------------------------------------------
    # 2-5. Clone -> inject -> freeze_inverse -> su.train -> eval all
    # ------------------------------------------------------------------
    results = {}
    for name, N, M in SPARSITY_CFG:
        print("\n" + "-" * 55)
        print(f"  Clone -> inject {name} -> freeze_inverse_mask -> su.train")
        print("-" * 55)
        model = build_model().to(DEVICE)
        model.load_state_dict(state_dict)

        su.inject_switchable_mask(
            model, name, N=N, M=M,
            conv_axis="out", linear_orientation="row",
        )
        su.set_active_sparsity(model, name)
        su.freeze_inverse_mask(model, name)
        su.train(model, epochs=SPARSE_EPOCHS, N=N, M=M)

        print(f"\n  --- {name} ---")
        check_sparsity_switchable(model, name)
        results[name] = su.evaluate_all_sparsity_modes(
            model, val_loader, DEVICE,
            recalibrate_loader=train_loader,
        )

        su.clear_all_frozen()

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'':>20} {'Dense':>10} {'1:4':>10} {'1:8':>10} {'1:16':>10} {'1:32':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    r4   = results["1:4"]
    r8   = results["1:8"]
    r16  = results["1:16"]
    r32  = results["1:32"]

    print(f"  {'Dense baseline':>20} {dense_result[1]:>9.2f}% "
          f"{'—':>10} {'—':>10} {'—':>10} {'—':>10}")
    print(f"  {'After 1:4 train':>20} {r4['dense'][1]:>9.2f}% {r4['1:4'][1]:>9.2f}% "
          f"{'—':>10} {'—':>10} {'—':>10}")
    print(f"  {'After 1:8 train':>20} {r8['dense'][1]:>9.2f}% {'—':>10} {r8['1:8'][1]:>9.2f}% "
          f"{'—':>10} {'—':>10}")
    print(f"  {'After 1:16 train':>20} {r16['dense'][1]:>9.2f}% {'—':>10} {'—':>10} "
          f"{r16['1:16'][1]:>9.2f}% {'—':>10}")
    print(f"  {'After 1:32 train':>20} {r32['dense'][1]:>9.2f}% {'—':>10} {'—':>10} "
          f"{'—':>10} {r32['1:32'][1]:>9.2f}%")

    print(f"\n  (Dense column evaluates the sparse clone without its "
          f"mask — reveals weight drift.)")
    print(f"\n  Done.")
