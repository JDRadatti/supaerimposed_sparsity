"""Test freeze logic — four scenarios.

  1. Different mask per layer, all inverse_mask frozen.
  2. Same mask (1:8), different freeze type per layer.
  3. Before/after diff: freeze on → frozen unchanged; freeze off → all change.
  4. Freeze + mask active together → nothing changes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from v3.profiles import LayerSpec, ModeProfile, FreezeProfile, OutputOptions
from v3.model import SparsityModel


# ---------------------------------------------------------------------------
# Tiny model — 4 conv/linear layers, BN, biases
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1, bias=True)
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_epochs(sparse, x, targets, epochs=3):
    """Minimal training loop — expects inner model in train mode."""
    optimizer = optim.SGD(sparse.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = sparse._model(x)
        loss = loss_fn(outputs, targets)
        loss.backward()
        sparse._zero_frozen_gradients(optimizer)
        frozen_snap = sparse._snapshot_frozen_positions()
        optimizer.step()
        sparse._restore_frozen_positions(frozen_snap)


def _deep_copy_model(sparse):
    """Return a deep copy of the inner model for diffing."""
    return copy.deepcopy(sparse._model)


def _snapshot_all(sparse):
    """Return {name: cloned_param} for all parameters."""
    return {name: p.detach().clone() for name, p in sparse.named_parameters()}


# ---------------------------------------------------------------------------
# Profile definitions — shared across all tests
# ---------------------------------------------------------------------------

ALL_MODES = [
    # Per‑layer modes for test 1 (different mask per layer).
    ModeProfile("l0_1_4",  [LayerSpec(0, 1, (1, 4))]),
    ModeProfile("l1_1_8",  [LayerSpec(1, 2, (1, 8))]),
    ModeProfile("l2_1_16", [LayerSpec(2, 3, (1, 16))]),
    ModeProfile("l3_1_32", [LayerSpec(3, None, (1, 32))]),
    # Per‑layer modes for test 2 (same mask, different freeze).
    ModeProfile("l0_1_8",  [LayerSpec(0, 1, (1, 8))]),
    # Unified modes for visualisation.
    ModeProfile("all_1_8",      [LayerSpec(0, None, (1, 8))]),
    ModeProfile("front_1_8",    [LayerSpec(0, 2, (1, 8))]),      # layers 0,1
    ModeProfile("all_different",[
        LayerSpec(0, 1, (1, 4)), LayerSpec(1, 2, (1, 8)),
        LayerSpec(2, 3, (1, 16)), LayerSpec(3, None, (1, 32)),
    ]),
]

ALL_FREEZES = [
    # Test 1: inverse_mask on each layer.
    FreezeProfile("freeze_l0_pruned",  "inverse_mask", mode_profile="l0_1_4"),
    FreezeProfile("freeze_l1_pruned",  "inverse_mask", mode_profile="l1_1_8"),
    FreezeProfile("freeze_l2_pruned",  "inverse_mask", mode_profile="l2_1_16"),
    FreezeProfile("freeze_l3_pruned",  "inverse_mask", mode_profile="l3_1_32"),
    # Test 2: different freeze type per layer.
    FreezeProfile("freeze_l0_mask",    "mask",          mode_profile="l0_1_8"),
    FreezeProfile("freeze_l1_inverse", "inverse_mask",  mode_profile="l1_1_8"),
    FreezeProfile("freeze_batchnorm",  "batchnorm"),
    FreezeProfile("freeze_biases",     "biases"),
    FreezeProfile("freeze_l23_nonmasked", "non_masked", mode_profile="front_1_8"),
    FreezeProfile("freeze_everything", "all"),
]


# ---------------------------------------------------------------------------
# Test 1: different mask per layer, all inverse_mask frozen
# ---------------------------------------------------------------------------

def test1():
    print("=" * 60)
    print("TEST 1: different mask per layer, all inverse_mask")
    print("=" * 60)

    sparse = SparsityModel(
        _TinyModel(),
        mode_profiles=ALL_MODES,
        freeze_profiles=[fp for fp in ALL_FREEZES if fp.name.startswith("freeze_l") and "pruned" in fp.name],
        output_options=OutputOptions(output_dir="./runs", filename_prefix="test_freeze"),
    )

    # Stack all four inverse_mask freezes.
    for name in ["freeze_l0_pruned", "freeze_l1_pruned", "freeze_l2_pruned", "freeze_l3_pruned"]:
        sparse._set_freeze_profile(name)
    print(f"  Active: {sparse._active_freeze}")

    # Visualize.
    sparse.visualize(first_n=16)
    print("  -> wrote visualize_all_different.csv")

    # Train DENSE, verify.
    sparse._set_mode_profile(None)
    before = _snapshot_all(sparse)
    sparse._model.train()
    x = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 10, (4,))
    _run_epochs(sparse, x, t, epochs=3)

    # Check: pruned unchanged, kept changed — for each layer.
    all_ok = True
    layer_specs = [
        ("conv1", "l0_1_4",   "1:4",  0.25),
        ("conv2", "l1_1_8",   "1:8",  0.125),
        ("conv3", "l2_1_16",  "1:16", 0.0625),
        ("fc",    "l3_1_32",  "1:32", 0.03125),
    ]
    nm = dict(sparse._model.named_modules())
    for fqn, mode_name, key, kept_pct in layer_specs:
        mod = nm[fqn]
        mask = mod.nm_masks[key].bool()
        b = before[f"{fqn}.weight"]
        a = mod.weight.detach()
        kept_ok = not torch.allclose(a[mask], b[mask], atol=1e-8)
        pruned_ok = torch.allclose(a[~mask], b[~mask], atol=1e-8)
        st = "PASS" if (kept_ok and pruned_ok) else "FAIL"
        if not (kept_ok and pruned_ok):
            all_ok = False
        print(f"  {fqn} ({key}): kept_changed={kept_ok}  pruned_frozen={pruned_ok}  -> {st}")
    assert all_ok, "Test 1 failed!"
    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 2: same mask (1:8), different freeze per layer
# ---------------------------------------------------------------------------

def test2():
    print("=" * 60)
    print("TEST 2: same mask (1:8), different freeze per layer")
    print("=" * 60)

    sparse = SparsityModel(
        _TinyModel(),
        mode_profiles=ALL_MODES,
        freeze_profiles=[
            fp for fp in ALL_FREEZES
            if fp.name in ("freeze_l0_mask", "freeze_l1_inverse",
                           "freeze_batchnorm", "freeze_biases",
                           "freeze_l23_nonmasked")
        ],
        output_options=OutputOptions(output_dir="./runs", filename_prefix="test_freeze"),
    )

    # Stack freezes.
    for name in ("freeze_l0_mask", "freeze_l1_inverse",
                 "freeze_batchnorm", "freeze_biases", "freeze_l23_nonmasked"):
        sparse._set_freeze_profile(name)
    print(f"  Active: {sparse._active_freeze}")

    # Visualize.
    sparse.visualize(first_n=16)
    print("  -> wrote visualize_all_1_8.csv")

    print("  PASSED (visual inspection)\n")


# ---------------------------------------------------------------------------
# Test 3: before/after — freeze on vs off
# ---------------------------------------------------------------------------

def test3():
    print("=" * 60)
    print("TEST 3: before/after diff — freeze on vs off")
    print("=" * 60)

    # Reuse test 2 configuration.
    sparse = SparsityModel(
        _TinyModel(),
        mode_profiles=ALL_MODES,
        freeze_profiles=[
            fp for fp in ALL_FREEZES
            if fp.name in ("freeze_l0_mask", "freeze_l1_inverse",
                           "freeze_batchnorm", "freeze_biases",
                           "freeze_l23_nonmasked")
        ],
        output_options=OutputOptions(output_dir="./runs", filename_prefix="test_freeze"),
    )
    for name in ("freeze_l0_mask", "freeze_l1_inverse",
                 "freeze_batchnorm", "freeze_biases", "freeze_l23_nonmasked"):
        sparse._set_freeze_profile(name)

    sparse._set_mode_profile(None)
    x = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 10, (4,))

    # ── 3a: train WITH freeze ──────────────────────────────────────
    before_frozen = _snapshot_all(sparse)
    before_copy_a = _deep_copy_model(sparse)
    sparse._model.train()
    _run_epochs(sparse, x, t, epochs=3)
    sparse.visualize_diff(before_copy_a, mode_name="all_1_8")
    print("  3a: wrote visualize_diff_all_1_8.csv (freeze ON)")

    # Verify: only unfrozen positions changed.
    # conv1: mask freeze → M positions frozen (1/8 = 12.5%)
    # conv2: inverse_mask → pruned frozen (7/8 = 87.5%)
    # bn1/bn2: fully frozen
    # biases: fully frozen
    # conv3/fc: fully frozen (non_masked)

    # Check conv1: M positions should be unchanged.
    nm = dict(sparse._model.named_modules())
    mask_1_8 = nm["conv1"].nm_masks["1:8"].bool()
    a = nm["conv1"].weight.detach()
    b = before_frozen["conv1.weight"]
    frozen_ok = torch.allclose(a[mask_1_8], b[mask_1_8], atol=1e-8)
    unfrozen_ok = not torch.allclose(a[~mask_1_8], b[~mask_1_8], atol=1e-8)
    print(f"  3a conv1: M-frozen={frozen_ok}  pruned-changed={unfrozen_ok}")

    # Check conv3: fully frozen (non_masked).
    a3 = nm["conv3"].weight.detach()
    b3 = before_frozen["conv3.weight"]
    conv3_frozen_ok = torch.allclose(a3, b3, atol=1e-8)
    print(f"  3a conv3: fully_frozen={conv3_frozen_ok}")

    # Check biases: fully frozen.
    bias_ok = torch.allclose(nm["conv1"].bias.detach(), before_frozen["conv1.bias"], atol=1e-8)
    print(f"  3a biases: frozen={bias_ok}")

    assert frozen_ok and unfrozen_ok and conv3_frozen_ok and bias_ok, "Test 3a failed!"

    # ── 3b: clear freeze, train NOW ─────────────────────────────────
    sparse._set_freeze_profile(None)
    before_unfrozen = _snapshot_all(sparse)
    before_copy_b = _deep_copy_model(sparse)
    _run_epochs(sparse, x, t, epochs=3)
    sparse.visualize_diff(before_copy_b, mode_name="all_1_8")
    print("  3b: wrote visualize_diff_all_1_8.csv (freeze OFF)")

    # Verify: everything changed.
    all_changed = all(
        not torch.allclose(p.detach(), before_unfrozen[name], atol=1e-8)
        for name, p in sparse.named_parameters()
    )
    print(f"  3b all_params_changed={all_changed}")
    assert all_changed, "Test 3b failed!"

    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 4: freeze + mask active together
# ---------------------------------------------------------------------------

def test4():
    print("=" * 60)
    print("TEST 4: freeze + mask active together")
    print("=" * 60)

    sparse = SparsityModel(
        _TinyModel(),
        mode_profiles=ALL_MODES,
        freeze_profiles=[
            fp for fp in ALL_FREEZES
            if fp.name in ("freeze_l0_mask", "freeze_l1_inverse",
                           "freeze_batchnorm", "freeze_biases",
                           "freeze_l23_nonmasked")
        ],
        output_options=OutputOptions(output_dir="./runs", filename_prefix="test_freeze"),
    )
    for name in ("freeze_l0_mask", "freeze_l1_inverse",
                 "freeze_batchnorm", "freeze_biases", "freeze_l23_nonmasked"):
        sparse._set_freeze_profile(name)

    # ACTIVATE mask mode (not dense).
    sparse._set_mode_profile("all_1_8")

    x = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 10, (4,))

    before = _snapshot_all(sparse)
    before_copy = _deep_copy_model(sparse)
    sparse._model.train()
    _run_epochs(sparse, x, t, epochs=3)
    sparse.visualize_diff(before_copy, mode_name="all_1_8")
    print("  -> wrote visualize_diff_all_1_8.csv (mask + freeze)")

    # Expected: with mask active + freeze stacked:
    #   conv1: mask freeze protects M → both mask (pruned) + freeze (kept) → unchanged
    #   conv2: inverse_mask freeze protects pruned → kept positions train normally
    #   bn1,bn2: frozen → unchanged
    #   biases: frozen → unchanged
    #   conv3,fc: non_masked → unchanged
    nm = dict(sparse._model.named_modules())
    mask_1_8 = nm["conv1"].nm_masks["1:8"].bool()

    # conv1: all positions should be unchanged (mask protects pruned, freeze protects kept).
    a1 = nm["conv1"].weight.detach()
    b1 = before["conv1.weight"]
    c1_ok = torch.allclose(a1, b1, atol=1e-4)
    print(f"  conv1 all_unchanged={c1_ok}")

    # conv2: pruned positions frozen, kept positions train → kept should change.
    m2 = nm["conv2"].nm_masks["1:8"].bool()
    a2 = nm["conv2"].weight.detach()
    b2 = before["conv2.weight"]
    pruned2_ok = torch.allclose(a2[~m2], b2[~m2], atol=1e-4)
    kept2_changed = not torch.allclose(a2[m2], b2[m2], atol=1e-4)
    print(f"  conv2 pruned_frozen={pruned2_ok}  kept_changed={kept2_changed}")

    # conv3, fc, bn, biases: fully frozen.
    c3_ok = torch.allclose(nm["conv3"].weight.detach(), before["conv3.weight"], atol=1e-4)
    fc_ok = torch.allclose(nm["fc"].weight.detach(), before["fc.weight"], atol=1e-4)
    b_ok = torch.allclose(nm["conv1"].bias.detach(), before["conv1.bias"], atol=1e-4)
    bn_ok = torch.allclose(nm["bn1"].weight.detach(), before["bn1.weight"], atol=1e-4)
    print(f"  conv3_frozen={c3_ok}  fc_frozen={fc_ok}  bias_frozen={b_ok}  bn_frozen={bn_ok}")

    assert c1_ok and pruned2_ok and kept2_changed, "Test 4 failed!"
    assert c3_ok and fc_ok and b_ok and bn_ok, "Test 4 fully‑frozen check failed!"

    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    test1()
    test2()
    test3()
    test4()
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
