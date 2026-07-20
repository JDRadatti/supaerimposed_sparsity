# Superimposed Sparsity Research

One set of parameters, multiple N:M sprasity modes.

## Install

```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Run `main.py`

`main.py` uses a progressive freezing technique to train the superimposed
sparsity model. First, it trains a dense VGG16 model for 200 epochs. Then, it
trains the same weights with a 1:32 mask. Next, it freezes the 1:32 mask,
preventing the weights from changing, and trains a 1:16 mask. Repeats for 1:8
and 1:4 masks, each time freezing the previous mask. Finally, it freezes the
1:4 mask and fine-tunes the dense non-masked weights. The first and last layer,
as well as the BN buffers are frozen for each training after the first dense
training. Each sparsity mode maintains its own copy of the BN stats buffer.

Total epochs = 200 + 30 * 5 = 350

```
source .venv/bin/activate
python3 main.py
```

## Run `main_joint.py`

`main_joint.py` uses a joint training method from "Slimmable Neural Networks"
(Yu et al., 2019). Each batch computes losses for all sparsity modes,
accumulates gradients, and performs a single shared optimizer step. Each
sparsity mode maintains its own copy of the BN stats buffer.

Total epochs = 200

```
source .venv/bin/activate
python3 main_joint.py
```

## Output

Checkpoints, comparison files, and an overview are saved to `./runs/`.

