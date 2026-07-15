"""test_serialize — full train / eval / save / load / verify pipeline.

Usage:
    python test_serialize.py train    # train, eval, save
    python test_serialize.py load     # load, eval, verify matches saved
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn as nn
import torch.utils.data as tud
from torch.optim import lr_scheduler

from v3.profiles import LayerSpec, ModeProfile, FreezeProfile, TrainingProfile, OutputOptions
from v3.model import SparsityModel
from v3.testing import compare_single_input


_ROOT = os.path.dirname(os.path.dirname(__file__))
RUN_DIR = os.path.join(_ROOT, "runs", "test_serialize")
CKPT_PATH = os.path.join(RUN_DIR, "checkpoint.pt")
RESULTS_PATH = os.path.join(RUN_DIR, "results.json")
VAL_DATA_PATH = os.path.join(RUN_DIR, "val_data.pt")
SEED = 42


class _SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _build_sparse():
    return SparsityModel(
        _SmallModel(),
        mode_profiles=[
            ModeProfile("all_1_4", [
                LayerSpec(0, 1, "dense"), LayerSpec(1, -1, (1, 4)),
                LayerSpec(-1, None, "dense"),
            ]),
        ],
        freeze_profiles=[
            FreezeProfile("freeze_pruned", freeze_type="inverse_mask",
                          mode_profile="all_1_4"),
        ],
        training_profiles=[
            TrainingProfile(
                name="default", epochs=5, batch_size=32, num_workers=0,
                scheduler_fn=lambda opt: lr_scheduler.StepLR(
                    opt, step_size=3, gamma=0.5,
                ),
            ),
        ],
        output_options=OutputOptions(output_dir="./runs",
                                     filename_prefix="test_serialize"),
    )


class _RandDataset(tud.Dataset):
    def __len__(self): return 200
    def __getitem__(self, i): return torch.randn(3, 32, 32), torch.randint(0, 10, (1,))


def make_loaders(batch_size=32):
    ds = _RandDataset()
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = tud.random_split(ds, [n_train, n_val])
    tl = tud.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    vl = tud.DataLoader(val_ds, batch_size=batch_size)
    return tl, vl


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def cmd_train():
    torch.manual_seed(SEED)
    os.makedirs(RUN_DIR, exist_ok=True)
    sparse = _build_sparse()
    print(sparse._model)
    print(f"Parameters: {sum(p.numel() for p in sparse.parameters()):,}")

    tl, vl = make_loaders()

    # Collect val data before training consumes it — save as .pt,
    # and build a fixed loader used for BOTH train val and final eval.
    val_images, val_targets = [], []
    for img, tgt in vl:
        val_images.append(img.clone())
        val_targets.append(tgt.clone())
    torch.save({"images": torch.cat(val_images), "targets": torch.cat(val_targets)}, VAL_DATA_PATH)
    val_fixed_ds = tud.TensorDataset(torch.cat(val_images), torch.cat(val_targets).squeeze(1))
    val_fixed = tud.DataLoader(val_fixed_ds, batch_size=32)

    # Fresh train loader (random data) only.
    tl2, _ = make_loaders()

    print("\n--- Training ---")
    sparse.train(mode="all_1_4", freeze="freeze_pruned", training="default",
                 train_loader=tl2, val_loader=val_fixed)

    print("\n--- Evaluate ---")
    results = sparse.evaluate(val_loader=val_fixed)

    # Fixed-input reference output.
    torch.manual_seed(SEED + 1)
    x_fixed = torch.randn(8, 3, 32, 32)
    sparse._model.eval()
    with torch.no_grad():
        ref_out = sparse._model(x_fixed)
    ref_out_data = ref_out.tolist()

    sparse.save(CKPT_PATH)

    # Checksums + BN buffers + eval results + reference output.
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location="cpu")
    checksums = {n: t.sum().item() for n, t in ckpt["model_state_dict"].items()}
    bn_bufs = {}
    for mode_name, mdict in sparse._bn_buffers.items():
        bn_bufs[mode_name] = {
            mname: {k: v.tolist() for k, v in stats.items()}
            for mname, stats in mdict.items()
        }
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "eval": results,
            "checksums": checksums,
            "bn_buffers": bn_bufs,
            "ref_out": ref_out_data,
        }, f, indent=2)
    print(f"Saved eval results to {RESULTS_PATH}")
    print("Done.")


# ---------------------------------------------------------------------------
# Load & verify
# ---------------------------------------------------------------------------

def cmd_load():
    if not os.path.exists(CKPT_PATH):
        print(f"Checkpoint not found: {CKPT_PATH}\nRun 'train' first.")
        sys.exit(1)

    loaded = SparsityModel.load(CKPT_PATH, _SmallModel())
    print(loaded._model)
    print(f"Parameters: {sum(p.numel() for p in loaded.parameters()):,}")

    with open(RESULTS_PATH) as f:
        saved = json.load(f)

    # Reconstruct saved val data from .pt file (exact tensor precision).
    val_saved = torch.load(VAL_DATA_PATH, weights_only=True, map_location="cpu")
    val_dataset = tud.TensorDataset(val_saved["images"], val_saved["targets"].squeeze(1))
    vl = tud.DataLoader(val_dataset, batch_size=32)

    print("\n--- Evaluate (loaded) ---")
    results = loaded.evaluate(val_loader=vl)

    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location="cpu")
    all_ok = True

    # 1. Weight checksums.
    for name, tensor in ckpt["model_state_dict"].items():
        expected = saved["checksums"].get(name)
        if expected is not None and abs(tensor.sum().item() - expected) > 1e-6:
            all_ok = False
            print(f"  CHECKSUM FAIL: {name}")
    if all_ok:
        print("[CHECK] Weight checksums: ok")

    # 2. BN buffers.
    bn_ok = True
    for bn_key in saved.get("bn_buffers", {}):
        for mname in saved["bn_buffers"][bn_key]:
            for k in ("running_mean", "running_var"):
                sv = torch.tensor(saved["bn_buffers"][bn_key][mname][k])
                lv = loaded._bn_buffers[bn_key][mname][k]
                if not torch.allclose(sv, lv.cpu(), atol=1e-6):
                    bn_ok = False
                    print(f"  BN MISMATCH: {bn_key}.{mname}.{k}")
    if bn_ok:
        print("[CHECK] BN buffers: ok")
    else:
        all_ok = False

    # 3. Fixed-input output comparison.
    torch.manual_seed(SEED + 1)
    x_fixed = torch.randn(8, 3, 32, 32)
    loaded._set_mode_profile("all_1_4")
    loaded._model.eval()
    with torch.no_grad():
        out_loaded = loaded._model(x_fixed)
    ref_out = torch.tensor(saved["ref_out"])
    out_match = torch.allclose(ref_out, out_loaded, atol=1e-6)
    max_diff = (ref_out - out_loaded).abs().max().item()
    if out_match:
        print(f"[CHECK] Fixed-input output: ok  max_abs_diff={max_diff:.2e}")
    else:
        print(f"[CHECK] Fixed-input output: MISMATCH  max_abs_diff={max_diff:.2e}")
        all_ok = False

    # 4. Eval results — must match exactly (same model, same data).
    eval_match = True
    print("\nEval results:")
    for mode in sorted(results.keys()):
        sl, sa = saved["eval"][mode]
        ll, la = results[mode]
        l_ok = abs(sl - ll) < 1e-6
        a_ok = abs(sa - la) < 1e-6
        st = "PASS" if (l_ok and a_ok) else "FAIL"
        if not (l_ok and a_ok):
            eval_match = False
            all_ok = False
        print(f"  {mode:<14} saved=(l={sl:.4f}, a={sa:.2f})  loaded=(l={ll:.4f}, a={la:.2f})  -> {st}")

    if all_ok:
        print("\nAll checks passed.")
    else:
        print("\nFAILED")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd")
    sp.add_parser("train")
    sp.add_parser("load")
    args = p.parse_args()

    if args.cmd == "train":
        cmd_train()
    elif args.cmd == "load":
        cmd_load()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
