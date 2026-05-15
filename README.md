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

Train a model such that a single set of parameters works at
multiple sparsity levels (both 1:4 sparse and 1:1 dense).

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

**Checkpoint Caching:**
- Each phase checks for an existing `.pt` checkpoint before training.
  If found, the phase is skipped and the checkpoint is loaded directly.
- Delete a checkpoint file to force re-run that phase.

**BatchNorm Recalibration:**
- When switching between sparsity modes (e.g., dense → 1:4), the
  activation distribution changes because different weight elements
  are zeroed. This shifts the input statistics seen by BatchNorm layers.
- `recalibrate_bn(model, loader, device)` runs one forward pass over
  the training set in `train()` mode, letting each BN layer recompute
  its `running_mean` / `running_var` for the currently active mask.
  The model is returned to `eval()` mode afterwards.
- `evaluate_all_sparsity_modes` calls this recalibration for every
  sparsity key before computing accuracy. Without this step, BN stats
  from dense training would be stale for sparse-mode evaluation and
  vice versa, producing misleading accuracy numbers.

## Workflow / Phases

| Phase | Description | Mask | Frozen | Eval Modes |
|-------|------------|------|--------|------------|
| 1 | Dense training (90 epochs) | None | Nothing | dense |
| 2 | Sparse 1:4 training (30 epochs) | 1:4 (active) | Inverse mask (pruned positions) | 1:4, dense |
| 3a | Freeze mask, train pruned in dense (30 epochs) | 1:4 (eval only) | mask==1 | 1:4, dense |
| 3b | Freeze mask + biases, train pruned (30 epochs) | 1:4 (eval only) | mask==1 + all biases | 1:4, dense |
| 3c | Freeze everything except pruned (30 epochs) | 1:4 (eval only) | mask==1 + biases + BN + non-masked | 1:4, dense |

## Output Files

All files are written to `./runs/`:

| File | Contents |
|------|----------|
| `overview.txt` | Summary of all phases with mask, freeze, and eval results |
| `phase*_comparison.txt` | Per-element parameter diff with mask and freeze overlay |
| `phase*.pt` | Checkpoint with model state dict, masks, freeze info, and eval metrics |

