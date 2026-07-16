import copy
import torch
import torch.nn as nn

import vgg
import sparse


# ---------------------------------------------------------------------------
# Model Cache
# ---------------------------------------------------------------------------
model_cache = {}


def get_model_deep(model_name) -> nn.Module | None:
    return copy.deepcopy(model_cache[model_name]["model"]) if model_name in model_cache else None


def get_model_shallow(model_name) -> nn.Module | None:
    return model_cache[model_name]["model"] if model_name in model_cache else None


def cache_model_copy(model_src, model_name):
    if model_src not in model_cache:
        raise RuntimeError("ERROR: Model not found")
    if model_name in model_cache:
        raise RuntimeError("ERROR: Duplicate model name")
    cache_model(get_model_deep(model_src), model_name)


def cache_model(model, model_name, mask=None):
    if model_name in model_cache:
        raise RuntimeError("ERROR: Duplicate model name")
    model_cache[model_name] = {"model": model, "mask": mask}


def compare_models(model1, model2, cutoff=25) -> str:
    lines = []
    for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            lines.append(f"  MISMATCH: {name1} vs {name2}")
            continue
        p1f, p2f = p1.detach().flatten(), p2.detach().flatten()
        total = p1f.numel()
        k = min(cutoff, total)
        if k <= 0:
            lines.append(f"  {name1}: shape={tuple(p1.shape)} (skipped)")
            continue
        idxs = torch.randperm(total, device=p1f.device)[:k]
        sd = (p1f[idxs] - p2f[idxs]).abs()
        md, me, nd = sd.max().item(), sd.mean().item(), (sd > 1e-8).sum().item()
        lines.append(f"  {name1}: shape={tuple(p1.shape)} max_diff≈{md:.8f} mean_diff≈{me:.8f} n_diff_in_sample={nd}/{k} total={total}")
        if nd > 0:
            vals, sidxs = torch.topk(sd, min(k, nd), largest=True)
            for pos, val in zip(sidxs.tolist(), vals.tolist()):
                idx = idxs[pos].item()
                lines.append(f"    [{idx}] m1={p1f[idx].item():.6f} m2={p2f[idx].item():.6f} diff={val:.6f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers for sparse model materialisation
# ---------------------------------------------------------------------------
def _extract_masks(sparse_model) -> dict[str, torch.Tensor]:
    masks = {}
    for name, module in sparse_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight_mask"):
            masks[name] = module.weight_mask.detach().cpu()
    return masks


def _materialize_state_dict(sparse_model) -> dict[str, torch.Tensor]:
    src = sparse_model.state_dict()
    dst = {}
    for key, value in src.items():
        if key.endswith("_mask"):
            continue
        if key.endswith("_orig"):
            wk = key[:-5]
            mk = wk + "_mask"
            dst[wk] = (value * src[mk]).cpu() if mk in src else value.cpu()
        else:
            dst[key] = value.cpu()
    return dst


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------
def multi_snapshot(models: dict[str, nn.Module]) -> dict:
    snap = {"params": {}, "model_names": list(models.keys())}
    for name in models:
        for pname, p in models[name].named_parameters():
            if pname not in snap["params"]:
                snap["params"][pname] = {}
            snap["params"][pname][name] = p.detach().clone()
    return snap


def compare_multi(
    snapshot: dict,
    masks: dict[str, torch.Tensor] | None = None,
    line_cap: int = 50,
) -> str:
    lines = []
    names = snapshot["model_names"]
    header = "  [idx] " + "  ".join(f"{n:>14}" for n in names)
    if masks:
        header += "  mask"
    lines.append("=" * 90)
    lines.append("MULTI-MODEL COMPARISON")
    lines.append("=" * 90)
    lines.append(header)

    for pname in snapshot["params"]:
        vals = {n: snapshot["params"][pname][n].flatten() for n in names}
        n = list(vals.values())[0].numel()
        shape = tuple(snapshot["params"][pname][names[0]].shape)

        mod_name = pname.rsplit(".", 1)[0] if pname.endswith(".weight") else None
        mask_flat = None
        if masks is not None and mod_name is not None and mod_name in masks:
            mask_flat = masks[mod_name].flatten()

        lines.append(f"\n  {pname}  ({n} elem)  shape={shape}")
        limit = min(line_cap, n)

        for idx in range(limit):
            parts = [f"  [{idx:4d}]"]
            for name in names:
                parts.append(f"{vals[name][idx].item():>14.6f}")
            if mask_flat is not None:
                parts.append(f"({int(mask_flat[idx].item())})")
            lines.append("  ".join(parts))

        if line_cap < n:
            remaining = n - line_cap
            mask_info = ""
            if mask_flat is not None:
                n1 = mask_flat[line_cap:].sum().item()
                n0 = remaining - n1
                mask_info = f"  (remaining: {n0}×mask=0, {n1}×mask=1)"
            lines.append(f"    ... ({remaining} more elements{mask_info})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.datasets\.cifar")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, M = 1, 4
    print(f"Device: {device}   N={N} M={M}\n")

    # 1. Train dense
    print("=" * 60)
    print("Phase 1: Dense training")
    print("=" * 60)
    model = vgg.build_model().to(device)
    vgg.train(model, epochs=60)
    cache_model(copy.deepcopy(model).to("cpu"), "dense")
    print("  [CACHED] 'dense'\n")

    # 2. Train sparse (pruned)
    print("=" * 60)
    print(f"Phase 2: Sparse {N}:{M} training")
    print("=" * 60)
    sparse_model = vgg.build_model().to(device)
    sparse_model.load_state_dict(get_model_shallow("dense").state_dict())
    sparse.prune_nm(sparse_model, N=N, M=M, conv_axis="out", linear_orientation="row")
    vgg.train(sparse_model, epochs=30)

    masks = _extract_masks(sparse_model)
    materialized_sd = _materialize_state_dict(sparse_model)
    sparse_materialized = vgg.build_model()
    sparse_materialized.load_state_dict(materialized_sd)
    cache_model(sparse_materialized.to("cpu"), "sparse", mask=masks)
    print("  [CACHED] 'sparse'\n")

    # 3. Build superimposed
    print("=" * 60)
    print("Creating superimposed model")
    print("=" * 60)
    dense_model = get_model_shallow("dense")
    sparse_model_cached = get_model_shallow("sparse")
    superimposed_sd = copy.deepcopy(dense_model.state_dict())
    for mod_name, mask in masks.items():
        wk = f"{mod_name}.weight"
        if wk not in superimposed_sd:
            continue
        sw = sparse_model_cached.get_submodule(mod_name).weight.detach()
        superimposed_sd[wk][mask.bool()] = sw[mask.bool()]
    superimposed = vgg.build_model()
    superimposed.load_state_dict(superimposed_sd)
    cache_model(superimposed.to("cpu"), "superimposed")
    print("  [CACHED] 'superimposed'\n")

    # 4. Compare all three
    print("=" * 60)
    print("Compare: dense vs sparse vs superimposed")
    print("=" * 60)
    snap = multi_snapshot({
        "dense": get_model_shallow("dense"),
        "sparse": get_model_shallow("sparse"),
        "superimposed": get_model_shallow("superimposed"),
    })
    result = compare_multi(snap, masks=masks, line_cap=30)
    print(result)

    with open("multi_comparison.txt", "w") as f:
        f.write(result)
        f.write("\n")
    print("\n  [SAVED] multi_comparison.txt")

    # 5. Evaluate
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    _, test_loader = vgg.create_dataloaders("./data", batch_size=128, num_workers=4)
    for name in ["dense", "sparse", "superimposed"]:
        m = get_model_shallow(name).to(device)
        loss, acc = vgg.evaluate(m, test_loader, device)
        print(f"  {name:15s}:  loss={loss:.4f}  acc={acc:.2f}%")

    print("\nDone.")
