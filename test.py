import unittest
import copy
import tempfile
import os

import torch
import torch.nn as nn
import torch.optim as optim

from sparse import (
    _nm_mask_2d_rows,
    _nm_mask_conv2d_axis,
    _nm_mask_linear,
    inject_switchable_masks,
)
from vgg import restore_non_inverse_params


# ---------------------------------------------------------------------------
# Toy models for testing
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # first, skipped
        self.fc1 = nn.Linear(4 * 8 * 8, 16)                      # middle, masked
        self.fc2 = nn.Linear(16, 10)                             # last, skipped
        self.use_sparse = False

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def set_sparsity_mode(self, mode="dense"):
        assert mode in ["dense", "sparse", "tune_inverse", "tune_inverse_freeze_bn", "tune_inverse_freeze_all"]
        self.use_sparse = mode == "sparse"
        if hasattr(self, '_frozen_bn_modules'):
            for m in self._frozen_bn_modules:
                m.train()
                for p in m.parameters():
                    p.requires_grad = True
            self._frozen_bn_modules.clear()
        if hasattr(self, '_frozen_extra_items'):
            for param, orig_requires_grad in self._frozen_extra_items:
                param.requires_grad = orig_requires_grad
            self._frozen_extra_items.clear()
        for m in self.modules():
            if hasattr(m, "nm_mask"):
                m.use_sparse = (mode == "sparse")
                if hasattr(m, "_sparse_hook_handle"):
                    m._sparse_hook_handle.remove()
                    del m._sparse_hook_handle
                if mode == "sparse":
                    m._sparse_hook_handle = m.weight.register_hook(
                        lambda grad, mod=m: grad * mod.nm_mask
                    )
                elif mode in ("tune_inverse", "tune_inverse_freeze_bn", "tune_inverse_freeze_all"):
                    m._sparse_hook_handle = m.weight.register_hook(
                        lambda grad, mod=m: grad * (1 - mod.nm_mask)
                    )
        if mode == "tune_inverse_freeze_bn":
            self._frozen_bn_modules = []
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
                    self._frozen_bn_modules.append(m)
        if mode == "tune_inverse_freeze_all":
            self._frozen_bn_modules = []
            self._frozen_extra_items = []
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
                    self._frozen_bn_modules.append(m)
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    if not hasattr(m, 'nm_mask'):
                        for p in m.parameters():
                            self._frozen_extra_items.append((p, p.requires_grad))
                            p.requires_grad = False
                    else:
                        if m.bias is not None:
                            self._frozen_extra_items.append((m.bias, m.bias.requires_grad))
                            m.bias.requires_grad = False


class TinyModelWithBN(nn.Module):
    """Tiny model with BatchNorm2d for testing BN freeze behavior."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # first, skipped
        self.bn1 = nn.BatchNorm2d(4)                              # BN before masked layer
        self.fc1 = nn.Linear(4 * 8 * 8, 16)                       # middle, masked
        self.fc2 = nn.Linear(16, 10)                              # last, skipped
        self.use_sparse = False
        self._frozen_bn_modules = []

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    set_sparsity_mode = TinyModel.set_sparsity_mode


# ---------------------------------------------------------------------------
# 1. N:M Mask Generation Tests
# ---------------------------------------------------------------------------

class TestNMMaskGeneration(unittest.TestCase):
    """Tests for _nm_mask_2d_rows, _nm_mask_conv2d_axis, _nm_mask_linear."""

    def test_basic_1_of_2(self):
        x = torch.tensor([[5.0, 1.0, 3.0, 0.0]])
        mask = _nm_mask_2d_rows(x, N=1, M=2)
        expected = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        self.assertTrue(torch.equal(mask, expected))

    def test_1_of_4_multiple_rows(self):
        x = torch.tensor([
            [5.0, 1.0, 3.0, 0.0, 2.0, 8.0, 1.0, 1.0],
            [9.0, 2.0, 0.0, 0.5, 4.1, 4.0, 3.9, 1.0],
        ])
        mask = _nm_mask_2d_rows(x, N=1, M=4)
        # Row0: [5,1,3,0] -> keep 5(idx0), [2,8,1,1] -> keep 8(idx1)
        # Row1: [9,2,0,0.5] -> keep 9(idx0), [4.1,4.0,3.9,1.0] -> keep 4.1(idx0)
        expected = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ])
        self.assertTrue(torch.equal(mask, expected))

    def test_n_equals_m_all_ones(self):
        x = torch.tensor([[5.0, 1.0, 3.0, 0.0]])
        mask = _nm_mask_2d_rows(x, N=2, M=2)
        expected = torch.ones_like(x)
        self.assertTrue(torch.equal(mask, expected))

    def test_n_equals_zero_all_zeros(self):
        x = torch.tensor([[5.0, 1.0, 3.0, 0.0]])
        mask = _nm_mask_2d_rows(x, N=0, M=2)
        expected = torch.zeros_like(x)
        self.assertTrue(torch.equal(mask, expected))

    def test_padding_when_not_divisible(self):
        x = torch.tensor([[5.0, 1.0, 3.0]])  # 3 cols, not divisible by M=2
        mask = _nm_mask_2d_rows(x, N=1, M=2)
        expected = torch.tensor([[1.0, 0.0, 1.0]])
        self.assertTrue(torch.equal(mask, expected))

    def test_negative_values(self):
        x = torch.tensor([[-5.0, -1.0, -3.0, -10.0]])
        mask = _nm_mask_2d_rows(x, N=1, M=2)
        # |-5| > |-1| => keep -5; |-10| > |-3| => keep -10
        expected = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        self.assertTrue(torch.equal(mask, expected))

    def test_exact_ratio(self):
        """With random normal init, every group of M gets exactly N ones."""
        x = torch.randn(100, 64)
        mask = _nm_mask_2d_rows(x, N=2, M=4)
        ratio = mask.sum().item() / mask.numel()
        self.assertEqual(ratio, 0.5)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            _nm_mask_2d_rows(torch.randn(4, 4), N=-1, M=2)
        with self.assertRaises(ValueError):
            _nm_mask_2d_rows(torch.randn(4, 4), N=3, M=2)
        with self.assertRaises(ValueError):
            _nm_mask_2d_rows(torch.randn(4, 4), N=1, M=0)

    def test_conv2d_mask_in_axis(self):
        w = torch.randn(8, 12, 3, 3)
        mask = _nm_mask_conv2d_axis(w, N=1, M=4, axis="in")
        self.assertEqual(mask.shape, w.shape)
        ratio = mask.sum().item() / mask.numel()
        self.assertEqual(ratio, 0.25)

    def test_conv2d_mask_out_axis(self):
        w = torch.randn(8, 12, 3, 3)
        mask = _nm_mask_conv2d_axis(w, N=1, M=4, axis="out")
        self.assertEqual(mask.shape, w.shape)
        ratio = mask.sum().item() / mask.numel()
        self.assertEqual(ratio, 0.25)

    def test_linear_mask_row_orientation(self):
        w = torch.randn(16, 32)
        mask = _nm_mask_linear(w, N=1, M=4, orientation="row")
        self.assertEqual(mask.shape, w.shape)
        ratio = mask.sum().item() / mask.numel()
        self.assertEqual(ratio, 0.25)

    def test_linear_mask_col_orientation(self):
        w = torch.randn(16, 32)
        mask = _nm_mask_linear(w, N=1, M=4, orientation="col")
        self.assertEqual(mask.shape, w.shape)
        ratio = mask.sum().item() / mask.numel()
        self.assertEqual(ratio, 0.25)


# ---------------------------------------------------------------------------
# 2. Switchable Mask Injection Tests
# ---------------------------------------------------------------------------

class TestInjectSwitchableMasks(unittest.TestCase):
    """Tests that inject_switchable_masks correctly modifies the model."""

    def setUp(self):
        self.model = TinyModel()
        self.model.eval()

    def test_mask_attached_to_middle_layers_only(self):
        inject_switchable_masks(self.model, N=1, M=4)
        self.assertTrue(hasattr(self.model.fc1, "nm_mask"),
                        "Middle layer fc1 should have nm_mask")
        self.assertFalse(hasattr(self.model.conv1, "nm_mask"),
                         "First layer conv1 should NOT have nm_mask")
        self.assertFalse(hasattr(self.model.fc2, "nm_mask"),
                         "Last layer fc2 should NOT have nm_mask")

    def test_mask_shape_matches_weight(self):
        inject_switchable_masks(self.model, N=1, M=4)
        self.assertEqual(self.model.fc1.nm_mask.shape,
                         self.model.fc1.weight.shape)

    def test_default_mode_is_dense(self):
        inject_switchable_masks(self.model, N=1, M=4)
        self.assertFalse(self.model.fc1.use_sparse)

    def test_dense_forward_preserves_weights(self):
        original_weights = copy.deepcopy(self.model.fc1.weight.data)
        inject_switchable_masks(self.model, N=1, M=4)
        _ = self.model(torch.randn(1, 1, 8, 8))
        self.assertTrue(torch.equal(original_weights, self.model.fc1.weight.data),
                        "Dense forward pass should not modify weights")

    def test_sparse_forward_runs(self):
        inject_switchable_masks(self.model, N=1, M=4)
        self.model.set_sparsity_mode("sparse")
        out = self.model(torch.randn(2, 1, 8, 8))
        self.assertEqual(out.shape, (2, 10))

    def test_output_differs_between_modes(self):
        inject_switchable_masks(self.model, N=1, M=4)
        x = torch.randn(4, 1, 8, 8)

        self.model.set_sparsity_mode("dense")
        out_dense = self.model(x)

        self.model.set_sparsity_mode("sparse")
        out_sparse = self.model(x)

        self.assertFalse(torch.allclose(out_dense, out_sparse),
                         "Dense and sparse outputs should differ")

    def test_toggle_cycle_is_idempotent(self):
        inject_switchable_masks(self.model, N=1, M=4)
        x = torch.randn(2, 1, 8, 8)

        self.model.set_sparsity_mode("dense")
        out_dense = self.model(x)

        self.model.set_sparsity_mode("sparse")
        self.model.set_sparsity_mode("dense")
        out_dense_again = self.model(x)

        self.assertTrue(torch.allclose(out_dense, out_dense_again),
                        "Dense output should be identical after sparse->dense toggle")


# ---------------------------------------------------------------------------
# 3. Sparse Training Isolation Tests
# ---------------------------------------------------------------------------

class TestSparseTrainingIsolation(unittest.TestCase):
    """Tests that masked weights remain frozen during sparse training."""

    def setUp(self):
        self.model = TinyModel()
        inject_switchable_masks(self.model, N=1, M=4)
        self.model.set_sparsity_mode("sparse")
        self.mask = self.model.fc1.nm_mask

    def test_unmasked_weights_update(self):
        optimizer = optim.SGD([
            {"params": [self.model.fc1.weight], "weight_decay": 0},
            {"params": [self.model.conv1.weight, self.model.fc2.weight], "weight_decay": 5e-4},
        ], lr=0.1, momentum=0.9)

        pre = copy.deepcopy(self.model.fc1.weight.data)
        out = self.model(torch.randn(2, 1, 8, 8))
        out.sum().backward()
        optimizer.step()

        updated = self.mask == 1
        self.assertFalse(
            torch.allclose(pre[updated], self.model.fc1.weight.data[updated]),
            "Unmasked weights should change",
        )

    def test_masked_weights_frozen_with_weight_decay(self):
        """Even with weight_decay>0, masked weights must NOT change."""
        optimizer = optim.SGD([
            {"params": [self.model.fc1.weight], "weight_decay": 0},
            {"params": [self.model.conv1.weight, self.model.fc2.weight], "weight_decay": 5e-4},
        ], lr=0.01, momentum=0.9)

        pre = copy.deepcopy(self.model.fc1.weight.data)
        for _ in range(5):
            optimizer.zero_grad()
            out = self.model(torch.randn(4, 1, 8, 8))
            out.sum().backward()
            optimizer.step()

        frozen = self.mask == 0
        self.assertTrue(
            torch.allclose(pre[frozen], self.model.fc1.weight.data[frozen]),
            "Masked weights must not change even with weight_decay > 0",
        )

    def test_unmasked_weights_converge(self):
        """Unmasked weights can learn a simple task over multiple steps."""
        optimizer = optim.SGD([
            {"params": [self.model.fc1.weight], "weight_decay": 0},
            {"params": [self.model.conv1.weight, self.model.fc2.weight], "weight_decay": 5e-4},
        ], lr=0.01, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        pre = copy.deepcopy(self.model.fc1.weight.data)
        for _ in range(10):
            optimizer.zero_grad()
            out = self.model(torch.randn(8, 1, 8, 8))
            targets = torch.randint(0, 10, (8,))
            loss = loss_fn(out, targets)
            loss.backward()
            optimizer.step()

        updated = self.mask == 1
        frozen = self.mask == 0
        self.assertFalse(
            torch.allclose(pre[updated], self.model.fc1.weight.data[updated]),
            "Unmasked weights should change after multiple training steps",
        )
        self.assertTrue(
            torch.allclose(pre[frozen], self.model.fc1.weight.data[frozen]),
            "Masked weights must stay frozen across training",
        )

    def test_non_masked_weights_unchanged_after_sparse_training(self):
        original_weights = self.model.fc1.weight.data.clone()
        frozen_positions = self.mask == 0

        optimizer = optim.SGD([
            {"params": [self.model.fc1.weight], "weight_decay": 0},
            {"params": [self.model.conv1.weight, self.model.fc2.weight]},
        ], lr=0.01, momentum=0.9)
        for _ in range(10):
            optimizer.zero_grad()
            self.model(torch.randn(4, 1, 8, 8)).sum().backward()
            optimizer.step()

        self.assertTrue(
            torch.allclose(original_weights[frozen_positions],
                           self.model.fc1.weight.data[frozen_positions]),
            "Non-masked weights must be untouched after sparse training — superimposition broken"
        )

    def test_mask_unchanged_after_sparse_training(self):
        original_mask = self.model.fc1.nm_mask.clone()
        optimizer = optim.SGD([
            {"params": [self.model.fc1.weight], "weight_decay": 0},
            {"params": [self.model.conv1.weight, self.model.fc2.weight]},
        ], lr=0.01, momentum=0.9)
        for _ in range(5):
            optimizer.zero_grad()
            self.model(torch.randn(4, 1, 8, 8)).sum().backward()
            optimizer.step()

        self.assertTrue(torch.equal(original_mask, self.model.fc1.nm_mask),
                        "nm_mask buffer must not change during training")


# ---------------------------------------------------------------------------
# 4. Mask Persistence Tests
# ---------------------------------------------------------------------------

class TestMaskPersistence(unittest.TestCase):
    """Tests that masks survive save/load cycles."""

    def test_save_load_preserves_mask_buffer(self):
        model = TinyModel()
        inject_switchable_masks(model, N=1, M=4)
        original_mask = model.fc1.nm_mask.clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            torch.save(model.state_dict(), path)

        try:
            new_model = TinyModel()
            inject_switchable_masks(new_model, N=1, M=4)
            for p in new_model.parameters():
                nn.init.normal_(p)

            new_model.load_state_dict(torch.load(path, weights_only=True))

            self.assertTrue(
                torch.equal(original_mask, new_model.fc1.nm_mask),
                "nm_mask buffer should survive save/load",
            )
        finally:
            os.unlink(path)

    def test_sparse_mode_works_after_load(self):
        model = TinyModel()
        inject_switchable_masks(model, N=1, M=4)
        model.set_sparsity_mode("sparse")

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            torch.save(model.state_dict(), path)

        try:
            new_model = TinyModel()
            inject_switchable_masks(new_model, N=1, M=4)
            new_model.load_state_dict(torch.load(path, weights_only=True))
            new_model.set_sparsity_mode("sparse")

            x = torch.randn(2, 1, 8, 8)
            out = new_model(x)
            self.assertEqual(out.shape, (2, 10))
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 5. Dense Training Sanity Check
# ---------------------------------------------------------------------------

class TestDenseTraining(unittest.TestCase):
    """Basic sanity: standard training updates weights."""

    def test_weights_change_after_dense_step(self):
        model = TinyModel()
        pre = copy.deepcopy(model.fc1.weight.data)

        optimizer = optim.SGD(model.parameters(), lr=0.1)
        out = model(torch.randn(1, 1, 8, 8))
        out.sum().backward()
        optimizer.step()

        self.assertFalse(
            torch.allclose(pre, model.fc1.weight.data),
            "Dense training step should change weights",
        )


# ---------------------------------------------------------------------------
# 6. Tune Inverse Tests
# ---------------------------------------------------------------------------

class TestTuneInverse(unittest.TestCase):
    """Tests that tune_inverse mode freezes 1:4 weights and trains 3:4 weights."""

    def setUp(self):
        self.model = TinyModel()
        inject_switchable_masks(self.model, N=1, M=4)
        self.model.set_sparsity_mode("tune_inverse")
        self.mask = self.model.fc1.nm_mask

    def test_tune_inverse_uses_dense_forward(self):
        self.model.set_sparsity_mode("tune_inverse")
        self.assertFalse(self.model.fc1.use_sparse,
                         "tune_inverse should use dense forward (use_sparse=False)")

    def test_inverse_weights_update(self):
        """3:4 weights (inverse mask) should update."""
        optimizer = optim.SGD([
            {"params": [self.model.fc1.weight], "weight_decay": 0},
            {"params": [self.model.conv1.weight, self.model.fc2.weight], "weight_decay": 5e-4},
        ], lr=0.1, momentum=0.9)

        pre = copy.deepcopy(self.model.fc1.weight.data)
        out = self.model(torch.randn(2, 1, 8, 8))
        out.sum().backward()
        optimizer.step()

        inverse = self.mask == 0
        sparse = self.mask == 1

        self.assertFalse(
            torch.allclose(pre[inverse], self.model.fc1.weight.data[inverse]),
            "Inverse-masked (3:4) weights should change in tune_inverse mode",
        )
        self.assertTrue(
            torch.allclose(pre[sparse], self.model.fc1.weight.data[sparse]),
            "Sparse (1:4) weights must stay frozen in tune_inverse mode",
        )

    def test_inverse_training_with_weight_decay(self):
        """Even with weight_decay>0, the 1:4 masked weights must not change."""
        optimizer = optim.SGD([
            {"params": [self.model.fc1.weight], "weight_decay": 0},
            {"params": [self.model.conv1.weight, self.model.fc2.weight], "weight_decay": 5e-4},
        ], lr=0.01, momentum=0.9)

        pre = copy.deepcopy(self.model.fc1.weight.data)
        for _ in range(5):
            optimizer.zero_grad()
            out = self.model(torch.randn(4, 1, 8, 8))
            out.sum().backward()
            optimizer.step()

        sparse = self.mask == 1
        self.assertTrue(
            torch.allclose(pre[sparse], self.model.fc1.weight.data[sparse]),
            "1:4 weights must not change even with weight_decay > 0 in tune_inverse mode",
        )

    def test_hooks_dont_stack(self):
        """Multiple set_sparsity_mode calls should not cause hook stacking."""
        model = TinyModel()
        inject_switchable_masks(model, N=1, M=4)

        model.set_sparsity_mode("dense")
        model.set_sparsity_mode("sparse")
        model.set_sparsity_mode("dense")
        model.set_sparsity_mode("tune_inverse")

        pre = copy.deepcopy(model.fc1.weight.data)
        optimizer = optim.SGD([
            {"params": [model.fc1.weight], "weight_decay": 0},
        ], lr=0.1, momentum=0.9)

        out = model(torch.randn(2, 1, 8, 8))
        out.sum().backward()
        optimizer.step()

        mask = model.fc1.nm_mask
        self.assertTrue(
            torch.allclose(pre[mask == 1], model.fc1.weight.data[mask == 1]),
            "1:4 weights should be frozen even after multiple mode switches",
        )

    def test_tune_inverse_forward_uses_all_weights(self):
        """In tune_inverse mode, forward should use all weights (like dense)."""
        inject_switchable_masks(self.model, N=1, M=4)
        x = torch.randn(2, 1, 8, 8)

        self.model.set_sparsity_mode("dense")
        out_dense = self.model(x)

        self.model.set_sparsity_mode("tune_inverse")
        out_tune = self.model(x)

        self.assertTrue(torch.allclose(out_dense, out_tune),
                        "tune_inverse forward should match dense forward (all weights active)")


# ---------------------------------------------------------------------------
# 7. BatchNorm Freeze Tests
# ---------------------------------------------------------------------------

class TestBatchNormFreeze(unittest.TestCase):
    """Tests that tune_inverse_freeze_bn correctly freezes BatchNorm."""

    def setUp(self):
        self.model = TinyModelWithBN()
        inject_switchable_masks(self.model, N=1, M=4)

    def _train_step(self, model):
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        out = model(torch.randn(4, 1, 8, 8))
        out.sum().backward()
        optimizer.step()

    def test_bn_running_stats_frozen(self):
        """BN running_mean/var should NOT update in freeze_bn mode."""
        self.model.set_sparsity_mode("tune_inverse_freeze_bn")
        pre_mean = self.model.bn1.running_mean.clone()
        pre_var = self.model.bn1.running_var.clone()
        self._train_step(self.model)
        self.assertTrue(torch.equal(pre_mean, self.model.bn1.running_mean),
                        "running_mean should not update in freeze_bn mode")
        self.assertTrue(torch.equal(pre_var, self.model.bn1.running_var),
                        "running_var should not update in freeze_bn mode")

    def test_bn_affine_params_frozen(self):
        """BN weight and bias should NOT change in freeze_bn mode."""
        self.model.set_sparsity_mode("tune_inverse_freeze_bn")
        pre_w = self.model.bn1.weight.clone()
        pre_b = self.model.bn1.bias.clone()
        self._train_step(self.model)
        self.assertTrue(torch.equal(pre_w, self.model.bn1.weight),
                        "BN weight should not change in freeze_bn mode")
        self.assertTrue(torch.equal(pre_b, self.model.bn1.bias),
                        "BN bias should not change in freeze_bn mode")

    def test_bn_stats_update_in_regular_tune_inverse(self):
        """BN running_mean/var SHOULD update in tune_inverse mode (no freeze)."""
        self.model.set_sparsity_mode("tune_inverse")
        pre_mean = self.model.bn1.running_mean.clone()
        pre_var = self.model.bn1.running_var.clone()
        self._train_step(self.model)
        # At least one value should differ (BN sees new data)
        self.assertFalse(torch.equal(pre_mean, self.model.bn1.running_mean),
                         "running_mean should update in tune_inverse mode")
        self.assertFalse(torch.equal(pre_var, self.model.bn1.running_var),
                         "running_var should update in tune_inverse mode")

    def test_bn_affine_params_update_in_regular_tune_inverse(self):
        """BN weight and bias SHOULD change in tune_inverse mode (no freeze)."""
        self.model.set_sparsity_mode("tune_inverse")
        pre_w = self.model.bn1.weight.clone()
        pre_b = self.model.bn1.bias.clone()
        # Multiple steps to ensure a significant change
        optimizer = optim.SGD([
            {"params": [self.model.fc1.weight], "weight_decay": 0},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and p is not self.model.fc1.weight]},
        ], lr=0.1, momentum=0.9)
        for _ in range(3):
            optimizer.zero_grad()
            out = self.model(torch.randn(4, 1, 8, 8))
            out.sum().backward()
            optimizer.step()
        self.assertFalse(torch.allclose(pre_w, self.model.bn1.weight),
                         "BN weight should change in tune_inverse mode")
        self.assertFalse(torch.allclose(pre_b, self.model.bn1.bias),
                         "BN bias should change in tune_inverse mode")

    def test_freeze_bn_hooks_match_inverse(self):
        """Weight gradient hook should be identical between the two inverse modes.
        Compare in eval mode so BN stats match (running stats vs running stats)."""
        self.model.set_sparsity_mode("tune_inverse")
        self.model.eval()
        x = torch.randn(2, 1, 8, 8)
        out_a = self.model(x)

        self.model.set_sparsity_mode("tune_inverse_freeze_bn")
        out_b = self.model(x)

        self.assertTrue(torch.allclose(out_a, out_b),
                        "Forward should be identical (same weight hook, BN just frozen)")

    def test_bn_restored_after_mode_switch(self):
        """BN should become trainable again after switching away from freeze_bn."""
        self.model.set_sparsity_mode("tune_inverse_freeze_bn")
        self.assertTrue(not self.model.bn1.weight.requires_grad,
                        "BN weight should be frozen in freeze_bn mode")
        self.assertTrue(not self.model.bn1.training,
                        "BN should be in eval mode in freeze_bn mode")

        self.model.set_sparsity_mode("dense")
        self.assertTrue(self.model.bn1.weight.requires_grad,
                        "BN weight should be trainable again after switch")
        self.assertTrue(self.model.bn1.training,
                        "BN should be in train mode again after switch")


# ---------------------------------------------------------------------------
# 8. Restore Non-Inverse Params Tests
# ---------------------------------------------------------------------------

class TestRestoreNonInverseParams(unittest.TestCase):
    """Tests for restore_non_inverse_params."""

    def setUp(self):
        self.model = TinyModelWithBN()
        inject_switchable_masks(self.model, N=1, M=4)
        self.saved_sd = copy.deepcopy(self.model.state_dict())

    def test_restores_all_params_by_default(self):
        """If no training happened, restore should be a no-op."""
        restore_non_inverse_params(self.model, self.saved_sd)
        current_sd = self.model.state_dict()
        for key in self.saved_sd:
            self.assertTrue(
                torch.equal(self.saved_sd[key].cpu(), current_sd[key].cpu()),
                f"{key} should match after restore",
            )

    def test_restores_1_of_4_positions_from_saved(self):
        """1:4 weight positions should be restored from saved state."""
        mask = self.model.fc1.nm_mask
        pre = self.model.fc1.weight.data.clone()
        # Modify 1:4 positions in current weights
        self.model.fc1.weight.data[mask == 1] = torch.randn(int(mask.sum().item()))
        # Restore
        restore_non_inverse_params(self.model, self.saved_sd)
        # 1:4 positions should be back to saved values
        restored = self.model.fc1.weight.data
        self.assertTrue(
            torch.equal(restored[mask == 1], self.saved_sd['fc1.weight'][mask == 1].to(restored.device)),
            "1:4 weight positions should be restored",
        )

    def test_keeps_3_of_4_positions_unchanged(self):
        """3:4 weight positions should NOT be overwritten by restore."""
        mask = self.model.fc1.nm_mask
        pre = self.model.fc1.weight.data.clone()
        # Modify 3:4 positions
        n_3of4 = int(mask.numel() - mask.sum().item())
        self.model.fc1.weight.data[mask == 0] = torch.randn(n_3of4)
        restore_non_inverse_params(self.model, self.saved_sd)
        # 3:4 positions should keep their modified values
        restored = self.model.fc1.weight.data
        self.assertFalse(
            torch.equal(restored[mask == 0], self.saved_sd['fc1.weight'][mask == 0].to(restored.device)),
            "3:4 weight positions should NOT be restored",
        )

    def test_restores_bn_stats(self):
        """BN running stats should be fully restored."""
        self.model.bn1.running_mean += 10.0
        self.model.bn1.running_var += 5.0
        restore_non_inverse_params(self.model, self.saved_sd)
        self.assertTrue(
            torch.equal(self.model.bn1.running_mean, self.saved_sd['bn1.running_mean'].to(self.model.bn1.running_mean.device)),
            "BN running_mean should be restored",
        )
        self.assertTrue(
            torch.equal(self.model.bn1.running_var, self.saved_sd['bn1.running_var'].to(self.model.bn1.running_var.device)),
            "BN running_var should be restored",
        )

    def test_restores_first_layer(self):
        """First layer (no mask) should be fully restored."""
        self.model.conv1.weight.data += 1.0
        restore_non_inverse_params(self.model, self.saved_sd)
        self.assertTrue(
            torch.equal(self.model.conv1.weight, self.saved_sd['conv1.weight'].to(self.model.conv1.weight.device)),
            "First layer weights should be fully restored",
        )

    def test_restores_last_layer(self):
        """Last layer (no mask) should be fully restored."""
        self.model.fc2.weight.data += 1.0
        restore_non_inverse_params(self.model, self.saved_sd)
        self.assertTrue(
            torch.equal(self.model.fc2.weight, self.saved_sd['fc2.weight'].to(self.model.fc2.weight.device)),
            "Last layer weights should be fully restored",
        )


# ---------------------------------------------------------------------------
# 9. Tune Inverse Freeze All Tests
# ---------------------------------------------------------------------------

class TestTuneInverseFreezeAll(unittest.TestCase):
    """Tests that tune_inverse_freeze_all freezes everything except 3:4 weights."""

    def setUp(self):
        self.model = TinyModelWithBN()
        inject_switchable_masks(self.model, N=1, M=4)

    def _train_step(self, model):
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        out = model(torch.randn(4, 1, 8, 8))
        out.sum().backward()
        optimizer.step()

    def test_forward_is_dense(self):
        """Forward should use all weights (like dense)."""
        self.model.set_sparsity_mode("dense")
        self.model.eval()
        x = torch.randn(2, 1, 8, 8)
        out_dense = self.model(x)
        self.model.set_sparsity_mode("tune_inverse_freeze_all")
        self.model.eval()
        out_freeze = self.model(x)
        self.assertTrue(torch.allclose(out_dense, out_freeze),
                        "freeze_all forward should match dense forward")

    def test_bn_running_stats_frozen(self):
        """BN running_mean/var should NOT update."""
        self.model.set_sparsity_mode("tune_inverse_freeze_all")
        pre_mean = self.model.bn1.running_mean.clone()
        pre_var = self.model.bn1.running_var.clone()
        self._train_step(self.model)
        self.assertTrue(torch.equal(pre_mean, self.model.bn1.running_mean))
        self.assertTrue(torch.equal(pre_var, self.model.bn1.running_var))

    def test_bn_affine_params_frozen(self):
        """BN weight and bias should NOT change."""
        self.model.set_sparsity_mode("tune_inverse_freeze_all")
        pre_w = self.model.bn1.weight.clone()
        pre_b = self.model.bn1.bias.clone()
        self._train_step(self.model)
        self.assertTrue(torch.equal(pre_w, self.model.bn1.weight))
        self.assertTrue(torch.equal(pre_b, self.model.bn1.bias))

    def test_first_layer_frozen(self):
        """First conv layer (no mask) should NOT change."""
        self.model.set_sparsity_mode("tune_inverse_freeze_all")
        pre = self.model.conv1.weight.clone()
        self._train_step(self.model)
        self.assertTrue(torch.equal(pre, self.model.conv1.weight),
                        "First layer should be frozen")

    def test_last_layer_frozen(self):
        """Last linear layer (no mask) should NOT change."""
        self.model.set_sparsity_mode("tune_inverse_freeze_all")
        pre = self.model.fc2.weight.clone()
        self._train_step(self.model)
        self.assertTrue(torch.equal(pre, self.model.fc2.weight),
                        "Last layer should be frozen")

    def test_masked_layer_bias_frozen(self):
        """Bias of masked layers should NOT change."""
        self.model.set_sparsity_mode("tune_inverse_freeze_all")
        pre = self.model.fc1.bias.clone()
        self._train_step(self.model)
        self.assertTrue(torch.equal(pre, self.model.fc1.bias),
                        "Bias of masked layer should be frozen")

    def test_sparse_weights_frozen(self):
        """1:4 weight positions should NOT change."""
        self.model.set_sparsity_mode("tune_inverse_freeze_all")
        mask = self.model.fc1.nm_mask
        pre = self.model.fc1.weight.data[mask == 1].clone()
        self._train_step(self.model)
        self.assertTrue(
            torch.equal(pre, self.model.fc1.weight.data[mask == 1]),
            "1:4 weights must stay frozen",
        )

    def test_inverse_weights_update(self):
        """3:4 weight positions SHOULD update."""
        self.model.set_sparsity_mode("tune_inverse_freeze_all")
        mask = self.model.fc1.nm_mask
        pre = self.model.fc1.weight.data[mask == 0].clone()
        self._train_step(self.model)
        self.assertFalse(
            torch.equal(pre, self.model.fc1.weight.data[mask == 0]),
            "3:4 weights should update",
        )

    def test_bn_restored_after_mode_switch(self):
        """BN should become trainable again after switching away."""
        self.model.set_sparsity_mode("tune_inverse_freeze_all")
        self.model.set_sparsity_mode("dense")
        self.assertTrue(self.model.bn1.weight.requires_grad)
        self.assertTrue(self.model.bn1.training)

    def test_first_layer_restored_after_mode_switch(self):
        """First layer should become trainable after switching away."""
        self.model.set_sparsity_mode("tune_inverse_freeze_all")
        self.model.set_sparsity_mode("dense")
        self.assertTrue(self.model.conv1.weight.requires_grad)

    def test_masked_bias_restored_after_mode_switch(self):
        """Masked layer bias should become trainable after switching away."""
        self.model.set_sparsity_mode("tune_inverse_freeze_all")
        self.model.set_sparsity_mode("dense")
        self.assertTrue(self.model.fc1.bias.requires_grad)

    def test_hooks_match_tune_inverse(self):
        """Weight gradient hook should be identical to regular tune_inverse."""
        self.model.eval()
        self.model.set_sparsity_mode("tune_inverse")
        x = torch.randn(2, 1, 8, 8)
        out_a = self.model(x)
        self.model.set_sparsity_mode("tune_inverse_freeze_all")
        out_b = self.model(x)
        self.assertTrue(torch.allclose(out_a, out_b),
                        "Forward should be identical (same weight hook)")


if __name__ == "__main__":
    unittest.main()
