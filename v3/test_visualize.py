"""Quick integration test: build a VGG model with multiple sparsity modes
and verify that `visualize()` produces per‑mode CSV files with correct stats."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3.profiles import LayerSpec, ModeProfile, OutputOptions
from v3.model import SparsityModel
import vgg


def main():
    # 1. Build a VGG16 model.
    model = vgg.build_model(num_classes=10, batch_norm=True, classifier_hidden=512, dropout_p=0.5)

    # 2. Define mode profiles.
    mode_profiles = [
        ModeProfile(
            name="dense_only",
            layer_specs=[
                LayerSpec(0, None, "dense"),
            ],
        ),
        ModeProfile(
            name="all_1_4",
            layer_specs=[
                LayerSpec(0, -1, (1, 4)),
                LayerSpec(-1, None, "dense"),
            ],
        ),
        ModeProfile(
            name="mixed",
            layer_specs=[
                LayerSpec(0, 1, "dense"),
                LayerSpec(1, 8, (1, 4)),
                LayerSpec(8, None, (1, 32)),
            ],
        ),
    ]

    # 3. Wrap in SparsityModel.
    sparse = SparsityModel(
        model,
        mode_profiles=mode_profiles,
        output_options=OutputOptions(output_dir="./runs", filename_prefix="test_visualize"),
    )

    print(sparse)
    print()

    # 4. Visualize.
    sparse.visualize(first_n=50)

    # 5. Check output files exist.
    run_dir = sparse._logger.run_dir
    print(f"\nRun directory: {run_dir}")
    for mode_name in [mp.name for mp in mode_profiles]:
        csv_path = os.path.join(run_dir, f"visualize_{mode_name}.csv")
        exists = os.path.exists(csv_path)
        size = os.path.getsize(csv_path) if exists else 0
        print(f"  {csv_path}:  exists={exists}  size={size} bytes")

    # 6. Quick content check: count M's in the 1:4 CSV (should be ~25%).
    csv_path = os.path.join(run_dir, "visualize_all_1_4.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            lines = f.readlines()
        # Find the first data layer's stats header.
        n_kept = n_total = 0
        in_data = False
        for line in lines:
            if line.startswith("#") and "kept_by_mask=" in line:
                # Parse: ..., kept_by_mask=123, pruned=456, ...
                parts = line.split()
                for p in parts:
                    if p.startswith("kept_by_mask="):
                        n_kept = int(p.split("=")[1].rstrip(","))
                    if p.startswith("total_elements="):
                        n_total = int(p.split("=")[1].rstrip(","))
                if n_total > 0:
                    pct = 100.0 * n_kept / n_total
                    print(f"\n  first masked layer: kept={n_kept} / total={n_total} = {pct:.1f}% (expect ~25% for 1:4)")
                    break

    print("\nDone.")


if __name__ == "__main__":
    main()
