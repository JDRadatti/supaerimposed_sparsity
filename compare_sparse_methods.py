"""
compare_sparse_methods.py
=========================
Compare two approaches to 1:4 sparse training from the same dense initialization:

  A. Switchable mask (inject_switchable_mask + freeze_inverse_mask + su.train)
  B. Permanent pruning (prune_nm + vgg.train)

Both start from the Phase 1 dense checkpoint and train for the same number
of epochs with the same hyperparameters. The 1:4 sparse accuracy should
be very similar.


============================================================
 RESULTS
============================================================

-------------------------------------------------------
  1. Load Phase 1 (dense) checkpoint
-------------------------------------------------------
[FREEZE] All frozen flags cleared.
[LOAD] Checkpoint loaded from ./runs/phase1_dense.pt
[EVAL] Sparsity 'dense': Avg Loss = 0.2913, Accuracy = 92.59%

-------------------------------------------------------
  2A. Switchable mask (inject + freeze_inverse + su.train)
-------------------------------------------------------
[FREEZE] 13 modules inverse mask '1:4' frozen.
  [13/15 Conv/Linear layers have nm_masks]
[EVAL] Sparsity '1:4': Avg Loss = 0.2762, Accuracy = 92.50%
  [13/15 Conv/Linear layers have nm_masks]
[EVAL] Sparsity 'dense': Avg Loss = 0.3191, Accuracy = 92.18%

-------------------------------------------------------
  2B. Permanent pruning (prune_nm + vgg.train)
-------------------------------------------------------

============================================================
  SUMMARY
============================================================
                            Dense        1:4
  -------------------- ---------- ----------
              Baseline     92.59%          —
       Switchable mask     92.18%     92.50%
     Permanent pruning          —     91.85%

  1:4 accuracy gap (switchable vs permanent): 0.65%
"""

import copy
import warnings

import torch
from vgg import build_model, create_dataloaders, evaluate as vgg_evaluate, train as vgg_train
import nm_sparsity_utils as su
import sparse

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.datasets\.cifar")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "./runs"
EPOCHS = 30


if __name__ == "__main__":
    print("=" * 60)
    print("  COMPARE SPARSE METHODS: switchable mask vs permanent pruning")
    print(f"  Device: {DEVICE}   Epochs: {EPOCHS}")
    print("=" * 60)

    train_loader, val_loader = create_dataloaders("./data", batch_size=128, num_workers=4)

    # ------------------------------------------------------------------
    # 1. Load dense baseline & save clean state_dict
    # ------------------------------------------------------------------
    print("\n" + "-" * 55)
    print("  1. Load Phase 1 (dense) checkpoint")
    print("-" * 55)
    dense_model = build_model().to(DEVICE)
    su.load_checkpoint(dense_model, f"{BASE_DIR}/phase1_dense.pt", device=DEVICE)
    dense_model.eval()

    state_dict = copy.deepcopy(dense_model.state_dict())

    results_bl = su.evaluate_all_sparsity_modes(
        dense_model, val_loader, DEVICE,
        recalibrate_loader=train_loader,
    )

    # ------------------------------------------------------------------
    # 2A. Switchable mask approach
    # ------------------------------------------------------------------
    print("\n" + "-" * 55)
    print("  2A. Switchable mask (inject + freeze_inverse + su.train)")
    print("-" * 55)

    model_a = build_model().to(DEVICE)
    model_a.load_state_dict(state_dict)

    su.inject_switchable_mask(
        model_a, "1:4", N=1, M=4,
        conv_axis="out", linear_orientation="row",
    )
    su.set_active_sparsity(model_a, "1:4")
    su.freeze_inverse_mask(model_a, "1:4")
    su.train(model_a, epochs=EPOCHS, N=1, M=4)

    results_a = su.evaluate_all_sparsity_modes(
        model_a, val_loader, DEVICE,
        recalibrate_loader=train_loader,
    )

    # ------------------------------------------------------------------
    # 2B. Permanent pruning approach
    # ------------------------------------------------------------------
    print("\n" + "-" * 55)
    print("  2B. Permanent pruning (prune_nm + vgg.train)")
    print("-" * 55)

    model_b = build_model().to(DEVICE)
    model_b.load_state_dict(state_dict)

    masks = sparse.prune_nm(
        model_b, N=1, M=4,
        conv_axis="out", linear_orientation="row",
    )
    vgg_train(model_b, epochs=EPOCHS)

    pr_loss, pr_acc = vgg_evaluate(model_b, val_loader, DEVICE)
    print(f"  Permanent sparse (1:4):  loss={pr_loss:.4f}  acc={pr_acc:.2f}%")

    # ------------------------------------------------------------------
    # 3. Summary comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    bl_dense = results_bl.get("dense")
    a_dense = results_a.get("dense")
    a_sparse = results_a.get("1:4")

    print(f"  {'':>20} {'Dense':>10} {'1:4':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    print(f"  {'Baseline':>20} {bl_dense[1]:>9.2f}% {'—':>10}")
    if a_dense:
        print(f"  {'Switchable mask':>20} {a_dense[1]:>9.2f}% {a_sparse[1]:>9.2f}%")
    else:
        print(f"  {'Switchable mask':>20} {'—':>10} {a_sparse[1]:>9.2f}%")
    print(f"  {'Permanent pruning':>20} {'—':>10} {pr_acc:>9.2f}%")

    # Compare 1:4 accuracies
    print()
    gap = abs(a_sparse[1] - pr_acc)
    print(f"  1:4 accuracy gap (switchable vs permanent): {gap:.2f}%")
    if gap < 1.0:
        print(u"  \u2713 Methods agree within 1% — the sparse training mechanism doesn\u2019t matter,")
        print(u"    only the mask and trainable positions.")
    else:
        print(u"  \u2717 Gap exceeds 1% — something differs between the two approaches.")
