"""Test save/load round‑trip for SparsityModel."""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from v3.profiles import LayerSpec, ModeProfile, FreezeProfile, OutputOptions
from v3.model import SparsityModel


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


def main():
    mode_profiles = [
        ModeProfile("all_1_4", [LayerSpec(0, -1, (1, 4)), LayerSpec(-1, None, "dense")]),
    ]
    freeze_profiles = [
        FreezeProfile("freeze_pruned", "inverse_mask", mode_profile="all_1_4"),
    ]

    # ── Build + populate BN buffers ──────────────────────────────────
    sparse = SparsityModel(
        _TinyModel(),
        mode_profiles=mode_profiles,
        freeze_profiles=freeze_profiles,
        output_options=OutputOptions(output_dir="./runs", filename_prefix="test_save"),
    )

    # Run a few forward passes in different modes so BN buffers differ.
    x = torch.randn(8, 3, 32, 32)
    with sparse.mode("all_1_4"):
        sparse._model.train()
        for _ in range(5):
            _ = sparse._model(x)
    # Switch to dense so BN buffers differ.
    with sparse.mode("all_1_4"):
        sparse._model.eval()
        for _ in range(5):
            _ = sparse._model(x)

    # ── Snapshot state before save ──────────────────────────────────
    orig_weights = {n: p.detach().clone() for n, p in sparse.named_parameters()}
    orig_bn_keys = set(sparse._bn_buffers.keys())
    orig_mode_names = set(sparse._mode_profiles.keys())
    orig_freeze_names = set(sparse._freeze_profiles.keys())

    # ── Save ────────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    try:
        sparse.save(ckpt_path)
        assert not os.path.getsize(ckpt_path) == 0, "Checkpoint is empty!"
        print("[SAVE] successful")

        # ── Load into fresh model ────────────────────────────────────
        loaded = SparsityModel.load(ckpt_path, _TinyModel())

        # ── Verify profiles ──────────────────────────────────────────
        assert set(loaded._mode_profiles.keys()) == orig_mode_names, "Mode profiles mismatch!"
        assert set(loaded._freeze_profiles.keys()) == orig_freeze_names, "Freeze profiles mismatch!"
        print("[CHECK] profiles: ok")

        # ── Verify weights ───────────────────────────────────────────
        for name, p in loaded.named_parameters():
            assert torch.allclose(p.detach(), orig_weights[name], atol=1e-8), f"Weight mismatch: {name}"
        print("[CHECK] weights: ok")

        # ── Verify masks ─────────────────────────────────────────────
        name_to_mod_orig = dict(sparse._model.named_modules())
        name_to_mod_loaded = dict(loaded._model.named_modules())
        for fqn, mod_orig in name_to_mod_orig.items():
            if not hasattr(mod_orig, "nm_masks"):
                continue
            mod_loaded = name_to_mod_loaded[fqn]
            assert hasattr(mod_loaded, "nm_masks"), f"Missing nm_masks on {fqn}"
            for key, mask_orig in mod_orig.nm_masks.items():
                mask_loaded = mod_loaded.nm_masks.get(key)
                if mask_orig is None:
                    assert mask_loaded is None, f"Mask {fqn}.{key} expected None"
                else:
                    assert torch.allclose(mask_loaded, mask_orig.cpu(), atol=1e-8), f"Mask mismatch: {fqn}.{key}"
        print("[CHECK] masks: ok")

        # ── Verify BN buffers ────────────────────────────────────────
        assert set(loaded._bn_buffers.keys()) == orig_bn_keys, "BN buffer keys mismatch!"
        for mode_name in orig_bn_keys:
            orig_bn = sparse._bn_buffers[mode_name]
            loaded_bn = loaded._bn_buffers[mode_name]
            for mname, orig_stats in orig_bn.items():
                loaded_stats = loaded_bn[mname]
                for k in ("running_mean", "running_var", "num_batches_tracked"):
                    assert torch.allclose(loaded_stats[k], orig_stats[k].cpu(), atol=1e-8), \
                        f"BN mismatch: {mode_name}.{mname}.{k}"
        print("[CHECK] BN buffers: ok")

        # ── Verify nothing active ────────────────────────────────────
        assert loaded._active_mode is None, "Active mode should be None after load!"
        assert loaded._active_freeze == [], "Active freeze should be empty after load!"
        print("[CHECK] deactivated: ok")

        # ── Verify reactivation works ────────────────────────────────
        loaded._set_mode_profile("all_1_4")
        loaded._set_freeze_profile("freeze_pruned")
        assert loaded._active_mode == "all_1_4"
        assert loaded._active_freeze == ["freeze_pruned"]
        # Forward pass in active mode should work.
        with torch.no_grad():
            loaded._model.eval()
            out = loaded._model(x)
            assert out.shape == (8, 10)
        print("[CHECK] reactivation: ok")

        print("\nAll save/load checks passed.")

    finally:
        os.unlink(ckpt_path)


if __name__ == "__main__":
    main()
