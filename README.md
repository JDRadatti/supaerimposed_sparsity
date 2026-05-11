# Superimposed Sparsity Research

## Install

```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## How to run

```
source .venv/bin/activate
python3 main.py
```

Checkpoints, comparison files, and an overview are saved to `./runs/`.

## Project Goal

Train a model such that a single set of parameters works at multiple sparsity levels (both 1:4 sparse and 1:1 dense).

## Key Mechanisms

**Switchable Masks:**
- `inject_switchable_mask` registers N:M masks as a dict buffer on each
  Conv2d/Linear module (`module.nm_masks[sparsity_key]`).
- The forward pass of every masked module is patched with a closure that
  checks `module.active_mask_key`. If the key is a registered sparsity
  level, the weight is multiplied by the mask before calling the underlying
  `conv2d`/`linear` function. If the key is `"dense"`, the original forward
  is used unchanged.
- First and last Conv/Linear layers are skipped (no mask applied).

**Element Freeze:**
- `freeze_mask` / `freeze_inverse_mask` register boolean tensors in
  `_element_freeze_masks` (keyed by Parameter identity) indicating which
  individual elements should not receive gradient updates.
- During each training step: gradients are zeroed at frozen positions,
  current values are snapshotted, `optimizer.step()` runs (which would
  corrupt frozen positions via weight_decay), and the snapshots are
  restored.

**Validation:**
- Every five epochs, the model parameters are checked against a snapshot to
  verify the frozen and masked elements are behaving correctly.
- Raises RuntimeError if snapshot test fails.

**Visualization:**
- The model_dict() is saved before and after each phase and a parameter diff is
  saved to `./runs/`

## Workflow / Phases

| Phase | Description | Mask | Frozen
|-------|------------|------|--------|
| 1 | Dense training (30 epochs) | None | Nothing |
| 2 | Sparse 1:4 training (30 epochs) | 1:4 (active) | Inverse mask |
| 3a | Freeze mask, train inverse mask (10 epochs) | 1:4 (eval only) | mask==1 |
| 3b | Freeze mask + biases, train inverse mask (10 epochs) | 1:4 (eval only) | mask==1 + all biases |
| 3c | Freeze everything except pruned weights (10 epochs) | 1:4 (eval only) | mask==1 + biases + BN + non-masked layers |
| 3d | Weight transplant: Phase 3a weights → Phase 2 skeleton | 1:4 (eval) | N/A (eval only) |

