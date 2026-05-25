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

**Per‑Sparsity BN Buffers:**
- Each sparsity key has its own saved BN buffers (`running_mean`,
  `running_var`, `num_batches_tracked`).
- `save_active_bn_stats(model)` snapshots the current BN buffers
  under the currently active sparsity key.
- `restore_bn_stats(model, key)` restores the saved buffers for a
  given key before evaluation, ensuring BN statistics match that
  mask's activation distribution.
- `train()` automatically calls `save_active_bn_stats` at the end
  of training. `evaluate_all_sparsity_modes` restores the correct
  buffers for each key before evaluating.
- Buffers are persisted in checkpoints alongside masks and freeze
  info, so the pipeline is fully resumable.

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
| `phase*.pt` | Checkpoint with model state dict, masks, freeze info, BN buffers, and eval metrics |

