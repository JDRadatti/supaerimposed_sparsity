"""
Phase 2: Compute 1:4 masks and save original weights

Workflow:
1. Load dense model from phase1_dense.pt
2. Compute 1:4 sparsity masks (keep 1 largest per group of 4 by magnitude)
3. Save original weights as model buffers for later
4. Save masks to phase2_masks.pt
"""

import torch
from pathlib import Path

from vgg import VGG16
from sparse import compute_nm_masks, save_original_weights


def run(n, m):
    runs_dir = Path("runs/week3")
    checkpoint_path = runs_dir / "phase1_dense.pt"
    masks_path = runs_dir / "phase2_masks.pt"

    print("=" * 60)
    print("PHASE 2: Compute 1:4 masks and save original weights")
    print("=" * 60)

    # Check if already done
    if masks_path.exists():
        print(f"Phase 2 already complete. Loading existing masks...")
        data = torch.load(masks_path, map_location="cpu")
        print(f"Masks for {len(data['masks'])} layers")
        print(f"N:M = {data['N']}:{data['M']}")
        return

    # Load dense model
    print(f"Loading model from {checkpoint_path}...")
    model = VGG16(num_classes=10, batch_norm=True, classifier_hidden=512, dropout_p=0.5)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    epochs_trained = ckpt.get("epoch", 0)
    val_acc = ckpt.get("val_acc", 0.0)
    print(f"Loaded model (trained {epochs_trained} epochs, val_acc={val_acc:.2f}%)")

    # Compute masks
    print(f"Computing {n}:{m} masks...")
    masks = compute_nm_masks(model, N=n, M=m, skip_first_last=True)
    print(f"Computed masks for {len(masks)} layers")

    total_sparse = sum(m.sum().item() for m in masks.values())
    total_params = sum(m.numel() for m in masks.values())
    sparsity_pct = 100.0 * (total_params - total_sparse) / total_params
    print(f"Sparsity: {total_sparse:,} / {total_params:,} = {sparsity_pct:.2f}%")

    # Save original weights as model buffers
    save_original_weights(model, buffer_name="orig_weights")

    # Save masks
    print(f"Saving masks to {masks_path}...")
    torch.save(
        {
            "masks": masks,
            "N": n,
            "M": m,
        },
        masks_path,
    )

    print(f"\nPhase 2 complete!")
    print(f"Masks saved to: {masks_path}")
    print(f"Original weights saved in model._saved_weights_info")

