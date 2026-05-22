"""
baseline.py
============
Train a dense model for 60 epochs, clone it, prune copies with 1:4 and 1:8,
train each for 30 more epochs, then evaluate all three.

Pipeline:
  1. Train dense (60 epochs)
  2. Clone -> prune_nm 1:4 -> train 30 epochs
  3. Clone -> prune_nm 1:8 -> train 30 epochs
  4. Evaluate all three with sparsity validation

============================================================
  RESULTS
============================================================
      Mode     loss      acc
  -------- -------- --------
     dense  0.3632   92.10%
       1:4  0.3279   90.98%
      1:16  0.3866   87.50%
      1:32  0.4404   85.32%
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
DENSE_EPOCHS = 60
SPARSE_EPOCHS = 30


def check_sparsity(model, tag=""):
    """Report mask sparsity for every pruned layer."""
    print(f"\n  [SPARSITY CHECK{(' ' + tag) if tag else ''}]")
    any_pruned = False
    for name, m in model.named_modules():
        if hasattr(m, 'weight_mask'):
            any_pruned = True
            mask = m.weight_mask
            total = mask.numel()
            zeros = (mask == 0).sum().item()
            pct = 100.0 * zeros / total
            orig = m.weight_orig
            eff = m.weight
            mismatch = (eff != orig * mask).sum().item()
            print(f"    {name}: {zeros}/{total} zero ({pct:.1f}%)  "
                  f"eff==orig*mask? {'YES' if mismatch == 0 else f'{mismatch} MISMATCH'}")
    if not any_pruned:
        print(f"    (no pruned layers)")


if __name__ == "__main__":
    print("=" * 60)
    print("  BASELINE — dense 60ep, prune, sparse 30ep")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    train_loader, val_loader = create_dataloaders("./data", batch_size=128, num_workers=4)

    # ------------------------------------------------------------------
    # 1. Train dense for 60 epochs
    # ------------------------------------------------------------------
    print("\n" + "-" * 55)
    print("  1. Train dense (60 epochs)")
    print("-" * 55)
    dense_model = build_model().to(DEVICE)
    vgg_train(dense_model, epochs=DENSE_EPOCHS)
    dense_model.eval()

    # Save clean state dict for cloning
    state_dict = copy.deepcopy(dense_model.state_dict())

    # Evaluate dense
    print(f"\n  --- dense ---")
    check_sparsity(dense_model, "dense")
    loss, acc = vgg_evaluate(dense_model, val_loader, DEVICE)
    print(f"  {'dense':>8} {loss:>7.4f}  {acc:>6.2f}%")
    dense_result = (loss, acc)

    # ------------------------------------------------------------------
    # 2. Clone -> 1:4 prune -> train 30 epochs
    # ------------------------------------------------------------------
    print("\n" + "-" * 55)
    print("  2. Clone -> prune 1:4 -> train 30 epochs")
    print("-" * 55)
    model_1_4 = build_model().to(DEVICE)
    model_1_4.load_state_dict(state_dict)
    sparse.prune_nm(model_1_4, N=1, M=4, conv_axis="out", linear_orientation="row")
    vgg_train(model_1_4, epochs=SPARSE_EPOCHS)
    print(f"\n  --- 1:4 ---")
    check_sparsity(model_1_4, "1:4")
    loss, acc = vgg_evaluate(model_1_4, val_loader, DEVICE)
    print(f"  {'1:4':>8} {loss:>7.4f}  {acc:>6.2f}%")
    result_1_4 = (loss, acc)

    # ------------------------------------------------------------------
    # 4. Clone -> 1:8 prune -> train 30 epochs
    # ------------------------------------------------------------------
    print("\n" + "-" * 55)
    print("  3. Clone -> prune 1:8 -> train 30 epochs")
    print("-" * 55)
    model_1_8 = build_model().to(DEVICE)
    model_1_8.load_state_dict(state_dict)
    sparse.prune_nm(model_1_8, N=1, M=8, conv_axis="out", linear_orientation="row")
    vgg_train(model_1_8, epochs=SPARSE_EPOCHS)
    print(f"\n  --- 1:8 ---")
    check_sparsity(model_1_8, "1:8")
    loss, acc = vgg_evaluate(model_1_8, val_loader, DEVICE)
    print(f"  {'1:8':>8} {loss:>7.4f}  {acc:>6.2f}%")
    result_1_8 = (loss, acc)


    # ------------------------------------------------------------------
    # 3. Clone -> 1:16 prune -> train 30 epochs
    # ------------------------------------------------------------------
    print("\n" + "-" * 55)
    print("  3. Clone -> prune 1:16 -> train 30 epochs")
    print("-" * 55)
    model_1_16 = build_model().to(DEVICE)
    model_1_16.load_state_dict(state_dict)
    sparse.prune_nm(model_1_16, N=1, M=16, conv_axis="out", linear_orientation="row")
    vgg_train(model_1_16, epochs=SPARSE_EPOCHS)
    print(f"\n  --- 1:16 ---")
    check_sparsity(model_1_16, "1:8")
    loss, acc = vgg_evaluate(model_1_16, val_loader, DEVICE)
    print(f"  {'1:16':>8} {loss:>7.4f}  {acc:>6.2f}%")
    result_1_16 = (loss, acc)

    # ------------------------------------------------------------------
    # 3. Clone -> 1:32 prune -> train 30 epochs
    # ------------------------------------------------------------------
    print("\n" + "-" * 55)
    print("  3. Clone -> prune 1:32 -> train 30 epochs")
    print("-" * 55)
    model_1_32 = build_model().to(DEVICE)
    model_1_32.load_state_dict(state_dict)
    sparse.prune_nm(model_1_32, N=1, M=32, conv_axis="out", linear_orientation="row")
    vgg_train(model_1_32, epochs=SPARSE_EPOCHS)
    print(f"\n  --- 1:32 ---")
    check_sparsity(model_1_32, "1:8")
    loss, acc = vgg_evaluate(model_1_32, val_loader, DEVICE)
    print(f"  {'1:32':>8} {loss:>7.4f}  {acc:>6.2f}%")
    result_1_32 = (loss, acc)


    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Mode':>8} {'loss':>8} {'acc':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'dense':>8} {dense_result[0]:>7.4f}  {dense_result[1]:>6.2f}%")
    print(f"  {'1:4':>8} {result_1_4[0]:>7.4f}  {result_1_4[1]:>6.2f}%")
    print(f"  {'1:8':>8} {result_1_8[0]:>7.4f}  {result_1_8[1]:>6.2f}%")
    print(f"  {'1:16':>8} {result_1_16[0]:>7.4f}  {result_1_16[1]:>6.2f}%")
    print(f"  {'1:32':>8} {result_1_32[0]:>7.4f}  {result_1_32[1]:>6.2f}%")
    print(f"\n  Done.")
