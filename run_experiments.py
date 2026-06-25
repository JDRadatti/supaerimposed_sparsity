"""
run_experiments.py - Run all gate loss variants and compare results.

Orchestrates 7 variants:
  0. baseline                     (pure BCE)
  1. entropy_regularized          (BCE + λ·|p̂_dense − p_true|²)
  2. accuracy_weighted (pos=2)    (BCE with pos_weight=2)
  3. accuracy_weighted (pos=5)    (BCE with pos_weight=5)
  4. accuracy_weighted (pos=10)   (BCE with pos_weight=10)
  5. confidence_calibrated        (BCE + λ·|logit| for wrong preds)
  6. flops_aware                  (BCE + λ·fraction_routed_to_dense)
"""

import os
import json
import logging
from datetime import datetime

import torch

from gate import (
    FeatureExtractor, GateModel,
    load_dense_model, load_sparse_model,
    create_eval_loaders, extract_features_and_predictions,
    generate_labels, evaluate_full, train_gate,
    DEVICE, BASE_DIR, GATE_INPUT_DIM, GATE_HIDDEN_DIM,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_VARIANTS = None


def _get_variants(train_labels):
    p_true = train_labels.float().mean().item()
    return [
        ("baseline",                  "baseline",                  {}),
        ("entropy_regularized",       "entropy_regularized",       {"p_true": p_true}),
        ("accuracy_weighted_pos=2",   "accuracy_weighted",         {"pos_weight": 2.0}),
        ("accuracy_weighted_pos=5",   "accuracy_weighted",         {"pos_weight": 5.0}),
        ("accuracy_weighted_pos=10",  "accuracy_weighted",         {"pos_weight": 10.0}),
        ("confidence_calibrated",     "confidence_calibrated",     {}),
        ("flops_aware",               "flops_aware",               {}),
    ]


def _save_experiment_results(results, exp_dir, variant_name):
    results_path = os.path.join(exp_dir, "gate_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Gate Results — {variant_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Gate Binary Accuracy:      {results['gate_acc']:.2f}%\n")
        f.write(f"Precision (needs-dense):   {results['precision']:.2f}%\n")
        f.write(f"Recall (needs-dense):      {results['recall']:.2f}%\n\n")
        f.write(f"Gated System Accuracy:     {results['gated_acc']:.2f}%\n")
        f.write(f"Dense Only Accuracy:       {results['dense_only_acc']:.2f}%\n")
        f.write(f"Sparse Only Accuracy:      {results['sparse_only_acc']:.2f}%\n")
        f.write(f"Improvement over sparse:   {results['gated_acc'] - results['sparse_only_acc']:+.2f}%\n")
        f.write(f"Gap to dense:              {results['dense_only_acc'] - results['gated_acc']:+.2f}%\n\n")
        f.write(f"2x2x2 Breakdown:\n")
        for name, count in results["categories"].items():
            f.write(f"  {name}: {count}\n")
        f.write(f"\nMode Distribution:\n")
        f.write(f"  1:32:  {results['mode_distribution']['1:32']}\n")
        f.write(f"  Dense: {results['mode_distribution']['dense']}\n")
    log.info(f"Results saved to {results_path}")


def _print_comparison(all_results):
    print()
    print("=" * 120)
    print("  GATE LOSS EXPERIMENT COMPARISON")
    print("=" * 120)
    print()
    print(f"{'Variant':<28} {'GateAcc':>8} {'GatedAcc':>9} {'ΔSparse':>9} {'ΔDense':>8} {'Route%':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 120)
    for name, r in all_results:
        delta_sparse = r["gated_acc"] - r["sparse_only_acc"]
        delta_dense = r["dense_only_acc"] - r["gated_acc"]
        route_pct = 100 * r["mode_distribution"]["dense"] / (r["mode_distribution"]["dense"] + r["mode_distribution"]["1:32"])
        print(f"{name:<28} {r['gate_acc']:>7.2f}% {r['gated_acc']:>8.2f}% "
              f"{delta_sparse:>+8.2f}% {delta_dense:>+7.2f}% "
              f"{route_pct:>7.1f}% {r['precision']:>7.2f}% {r['recall']:>7.2f}%")
    print("-" * 120)
    print()


def _save_comparison_json(all_results, path):
    serializable = []
    for name, r in all_results:
        d = {"variant": name}
        for k, v in r.items():
            if isinstance(v, (int, float, str, bool)):
                d[k] = v
            elif isinstance(v, dict):
                d[k] = v
        serializable.append(d)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    log.info(f"Comparison JSON saved to {path}")


def main():
    print("=" * 65)
    print("  GATE LOSS EXPERIMENT SUITE")
    print("=" * 65)
    print(f"  Device: {DEVICE}")
    print()

    os.makedirs(BASE_DIR, exist_ok=True)

    log.info("Loading models (once)...")
    dense_model = load_dense_model()
    sparse_model = load_sparse_model()

    log.info("Extracting features and predictions (once)...")
    train_loader, val_loader = create_eval_loaders("./data", 128, 4)

    train_features, train_dense_preds, train_sparse_preds, train_targets = \
        extract_features_and_predictions(dense_model, sparse_model, train_loader)
    train_labels = generate_labels(train_dense_preds, train_sparse_preds, train_targets)

    val_features, val_dense_preds, val_sparse_preds, val_targets = \
        extract_features_and_predictions(dense_model, sparse_model, val_loader)
    val_labels = generate_labels(val_dense_preds, val_sparse_preds, val_targets)

    variants = _get_variants(train_labels)

    all_results = []
    for i, (variant_name, loss_mode, kwargs) in enumerate(variants):
        print()
        print("—" * 65)
        print(f"  [{i+1}/{len(variants)}] {variant_name}")
        print("—" * 65)

        exp_dir = os.path.join(BASE_DIR, variant_name)
        os.makedirs(exp_dir, exist_ok=True)

        gate = GateModel()
        gate, best_val_acc = train_gate(
            gate, train_features, train_labels, val_features, val_labels,
            loss_mode=loss_mode,
            log_path=os.path.join(exp_dir, "training_log.csv"),
            **kwargs,
        )

        ckpt_path = os.path.join(exp_dir, "gate_checkpoint.pt")
        torch.save({
            "gate_state_dict": gate.state_dict(),
            "best_val_acc": best_val_acc,
            "input_dim": GATE_INPUT_DIM,
            "hidden_dim": GATE_HIDDEN_DIM,
            "loss_mode": loss_mode,
            "variant_name": variant_name,
        }, ckpt_path)
        log.info(f"Checkpoint saved to {ckpt_path}")

        results = evaluate_full(
            gate, val_features, val_labels,
            val_dense_preds, val_sparse_preds, val_targets,
        )

        _save_experiment_results(results, exp_dir, variant_name)
        all_results.append((variant_name, results))

    _print_comparison(all_results)
    _save_comparison_json(all_results, os.path.join(BASE_DIR, "comparison.json"))


if __name__ == "__main__":
    main()
