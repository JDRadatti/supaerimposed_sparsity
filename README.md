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

## Project Goal 

Train a model such that a single set of parameters works at
multiple sparsity levels (both 1:4 sparse and 1:1 dense). 

## Workflow

1. **Build and train dense VGG-16** on CIFAR-10 for 30 epochs.
2. **Inject switchable masks** (1:4 N:M) into middle layers. The mask is
   computed from the dense weights and stored as a persistent
   buffer. First and last layers are skipped. Default mode is **dense**.
3. **Fine-tune in sparse mode** for 30 epochs. The 1:4 active weights are
   updated via gradient descent. The 3:4 masked weights are frozen (gradient
   hook zeros their gradients, weight decay is disabled for masked params).
4. **Evaluate both modes** on the same weight tensor:

     | Mode | Sparsity | Accuracy |
     |------|----------|----------|
     | Sparse | 1:4 | 91.34% |
     | Dense | 1:1 | 25.44% |
