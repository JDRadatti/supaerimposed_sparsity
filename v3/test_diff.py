"""Tests: SparsityModel vs truly pruned model — weight diff + output equivalence.

Uses a small in‑memory model (no CIFAR loading) so it runs in <1 s.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from v3.profiles import LayerSpec, ModeProfile, OutputOptions
from v3.model import SparsityModel
from v3.testing import build_truly_pruned, compare_single_input


class _TinyModel(nn.Module):
    """Small model so tests run instantly."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main():
    model = _TinyModel()

    mode_profiles = [
        ModeProfile(
            name="all_1_4",
            layer_specs=[LayerSpec(0, None, (1, 4))],
        ),
        ModeProfile(
            name="mixed",
            layer_specs=[
                LayerSpec(0, 1, "dense"),
                LayerSpec(1, 3, (1, 4)),
                LayerSpec(3, None, (1, 32)),
            ],
        ),
    ]

    sparse = SparsityModel(
        model,
        mode_profiles=mode_profiles,
        output_options=OutputOptions(output_dir="./runs", filename_prefix="test_diff"),
    )
    print(sparse)

    # Single random input for inference check.
    x = torch.randn(4, 3, 32, 32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_passed = True

    for profile in mode_profiles:
        print(f"\n--- {profile.name} ---")
        pruned = build_truly_pruned(model, profile)

        # Weight‑level diff.
        sparse.visualize_diff(pruned, mode_name=profile.name)
        print(f"  wrote visualize_diff_{profile.name}.csv")

        # Output equivalence on single input.
        with sparse.mode(profile.name):
            result = compare_single_input(sparse, pruned, x, device=device)

        status = "PASSED" if result["all_close"] else "FAILED"
        if not result["all_close"]:
            all_passed = False
        print(f"  all_close={result['all_close']}  "
              f"max_abs_diff={result['max_abs_diff']:.2e}  -> {status}")

    print(f"\nOverall: {'PASSED' if all_passed else 'FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
