# ML Sparsity Research

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

## Experiment 1 (Week 3)

**Goal:** Train a model once so that one set of parameters works with different sparsity levels.

### Workflow

| Phase | Action | Epochs | Output | Val Acc |
|-------|--------|--------|--------|---------|
| 1 | Train dense VGG-16 (w_orig) | 30 | phase1_dense.pt | 85.96% |
| 2 | Compute 1:4 mask | - | phase2_masks.pt | - |
| 3 | Sparse fine-tune with mask (w') | 30 | phase3_sparse.pt | 92.05% |
| 4A | Create mixed weights, evaluate BEFORE retrain | - | phase4_mixed.pt | 77.30% |
| 4B | Retrain with frozen weights | 30 | phase4_mixed.pt | - |
| 4C | Evaluate AFTER retrain | - | phase4_mixed.pt | TBD% |

