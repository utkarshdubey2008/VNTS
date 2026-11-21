"""
Unified Benchmark Suite:
Empirical evaluation comparing DATS-SI against PyTorch Backpropagation (AdamW and SGD).
Evaluates on:
1. Continuous XOR (Non-linear separation)
2. Concentric Circles (Topological boundary)
3. Two-Spirals (Complex non-linear manifold)
4. Handwritten Digits (64-dim real vision manifold)
5. Deep Architecture Scaling (L in [2, 4, 8, 16] layers)
"""

import os
import sys
import json
import time
import resource
import torch
import numpy as np
from typing import Dict, Any, List, Optional

from src.dats.model import DATSNetwork
from src.dats.trainer import DATSTrainer
from src.baselines.backprop_baseline import StandardMLP, BackpropTrainer
from benchmarks.datasets import (
    generate_xor_dataset,
    generate_circles_dataset,
    generate_two_spirals,
    load_digits_dataset
)


def get_peak_memory_mb() -> float:
    """Returns peak resident set size (RSS) in megabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_single_comparison(
    task_name: str,
    layer_dims: List[int],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 40,
    batch_size: Optional[int] = None,
    dats_kwargs: Dict[str, Any] = None,
    seed: int = 42
) -> Dict[str, Any]:
    if dats_kwargs is None:
        dats_kwargs = {"beta": 0.35, "alpha": 0.4, "gamma": 1e-3, "reg_lambda": 1e-3, "normalize_targets": True}

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {task_name.upper()} | Architecture: {layer_dims} | Epochs: {epochs}")
    print(f"{'='*70}")

    results = {}

    # 1. Evaluate Conventional Backprop (AdamW)
    torch.manual_seed(seed)
    bp_adam_model = StandardMLP(layer_dims, seed=seed)
    bp_adam_trainer = BackpropTrainer(bp_adam_model, optimizer_name="adamw", lr=2e-3, weight_decay=1e-4)
    
    t0 = time.time()
    history_adam = bp_adam_trainer.fit(x_train, y_train, x_test, y_test, epochs=epochs, batch_size=batch_size or 64)
    adam_time = time.time() - t0
    adam_eval = bp_adam_trainer.evaluate(x_test, y_test)
    
    results["Backprop_AdamW"] = {
        "test_acc": adam_eval["accuracy"],
        "test_loss": adam_eval["loss"],
        "train_acc": history_adam["train_acc"][-1],
        "train_loss": history_adam["train_loss"][-1],
        "total_time_s": adam_time,
        "peak_rss_mb": get_peak_memory_mb()
    }
    print(f"[Backprop-AdamW] Test Acc: {adam_eval['accuracy']*100:.2f}% | Test Loss: {adam_eval['loss']:.4f} | Time: {adam_time:.3f}s")

    # 2. Evaluate Conventional Backprop (SGD+Momentum)
    torch.manual_seed(seed)
    bp_sgd_model = StandardMLP(layer_dims, seed=seed)
    bp_sgd_trainer = BackpropTrainer(bp_sgd_model, optimizer_name="sgd", lr=0.05, momentum=0.9, weight_decay=1e-4)
    
    t0 = time.time()
    history_sgd = bp_sgd_trainer.fit(x_train, y_train, x_test, y_test, epochs=epochs, batch_size=batch_size or 64)
    sgd_time = time.time() - t0
    sgd_eval = bp_sgd_trainer.evaluate(x_test, y_test)
    
    results["Backprop_SGD"] = {
        "test_acc": sgd_eval["accuracy"],
        "test_loss": sgd_eval["loss"],
        "train_acc": history_sgd["train_acc"][-1],
        "train_loss": history_sgd["train_loss"][-1],
        "total_time_s": sgd_time,
        "peak_rss_mb": get_peak_memory_mb()
    }
    print(f"[Backprop-SGD]   Test Acc: {sgd_eval['accuracy']*100:.2f}% | Test Loss: {sgd_eval['loss']:.4f} | Time: {sgd_time:.3f}s")

    # 3. Evaluate DATS-SI (Dual-Stream Algebraic Target Synthesis)
    torch.manual_seed(seed)
    dats_model = DATSNetwork(layer_dims, seed=seed)
    dats_trainer = DATSTrainer(
        dats_model,
        beta=dats_kwargs.get("beta", 0.35),
        alpha=dats_kwargs.get("alpha", 0.4),
        gamma=dats_kwargs.get("gamma", 1e-3),
        reg_lambda=dats_kwargs.get("reg_lambda", 1e-3),
        normalize_targets=dats_kwargs.get("normalize_targets", True),
        margin=1.0
    )
    
    t0 = time.time()
    history_dats = dats_trainer.fit(x_train, y_train, x_test, y_test, epochs=epochs, batch_size=batch_size)
    dats_time = time.time() - t0
    dats_eval = dats_trainer.evaluate(x_test, y_test)
    
    results["DATS_SI"] = {
        "test_acc": dats_eval["accuracy"],
        "test_loss": dats_eval["loss"],
        "train_acc": history_dats["train_acc"][-1],
        "train_loss": history_dats["train_loss"][-1],
        "best_val_acc": max(history_dats["val_acc"]) if history_dats["val_acc"] else dats_eval["accuracy"],
        "total_time_s": dats_time,
        "peak_rss_mb": get_peak_memory_mb()
    }
    print(f"[DATS-SI (Ours)] Test Acc: {dats_eval['accuracy']*100:.2f}% (Best: {results['DATS_SI']['best_val_acc']*100:.2f}%) | Test Loss: {dats_eval['loss']:.4f} | Time: {dats_time:.3f}s")

    return results


def run_all():
    os.makedirs("results", exist_ok=True)
    all_benchmarks = {}

    # Task 1: Continuous XOR
    X_tr, y_tr, X_te, y_te = generate_xor_dataset(n_samples=1200, noise=0.15, seed=42)
    all_benchmarks["xor"] = run_single_comparison(
        "Non-linear XOR",
        [2, 32, 16, 2],
        X_tr, y_tr, X_te, y_te,
        epochs=30,
        batch_size=None,
        dats_kwargs={"beta": 0.35, "alpha": 0.35, "gamma": 1e-3, "reg_lambda": 1e-3, "normalize_targets": True}
    )

    # Task 2: Concentric Circles
    X_tr, y_tr, X_te, y_te = generate_circles_dataset(n_samples=1200, noise=0.08, seed=42)
    all_benchmarks["circles"] = run_single_comparison(
        "Concentric Circles",
        [2, 32, 16, 2],
        X_tr, y_tr, X_te, y_te,
        epochs=30,
        batch_size=None,
        dats_kwargs={"beta": 0.35, "alpha": 0.35, "gamma": 1e-3, "reg_lambda": 1e-3, "normalize_targets": True}
    )

    # Task 3: Two-Spirals
    X_tr, y_tr, X_te, y_te = generate_two_spirals(n_samples=1200, noise=0.08, seed=42)
    all_benchmarks["two_spirals"] = run_single_comparison(
        "Two-Spirals Interleaved Manifold",
        [2, 128, 128, 64, 2],
        X_tr, y_tr, X_te, y_te,
        epochs=35,
        batch_size=None,
        dats_kwargs={"beta": 0.35, "alpha": 0.4, "gamma": 1e-3, "reg_lambda": 1e-4, "normalize_targets": True}
    )

    # Task 4: Real Handwritten Digits
    X_tr, y_tr, X_te, y_te = load_digits_dataset(seed=42)
    all_benchmarks["digits"] = run_single_comparison(
        "Handwritten Digits (8x8 Vision)",
        [64, 128, 64, 10],
        X_tr, y_tr, X_te, y_te,
        epochs=30,
        batch_size=None,
        dats_kwargs={"beta": 0.25, "alpha": 0.35, "gamma": 1e-3, "reg_lambda": 1e-3, "normalize_targets": True}
    )

    # Task 5: Depth-Scaling Stress Test
    print(f"\n{'='*70}")
    print("DEPTH SCALING EXPERIMENT: Evaluating Vanishing Gradients vs Subspace Stability")
    print(f"{'='*70}")
    
    depth_results = {}
    depths = [2, 4, 8, 16]
    X_tr, y_tr, X_te, y_te = generate_circles_dataset(n_samples=1000, noise=0.1, seed=42)
    
    for d in depths:
        dims = [2] + [32] * (d - 1) + [2]
        depth_results[f"depth_{d}"] = run_single_comparison(
            f"Depth Scaling (L={d})",
            dims,
            X_tr, y_tr, X_te, y_te,
            epochs=25,
            batch_size=None,
            dats_kwargs={"beta": 0.3, "alpha": 0.35, "gamma": 1e-3, "reg_lambda": 1e-3, "normalize_targets": True}
        )

    all_benchmarks["depth_scaling"] = depth_results

    # Save to JSON
    out_path = "results/benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_benchmarks, f, indent=2)
    print(f"\n[SUCCESS] All benchmark results recorded to {out_path}")


if __name__ == "__main__":
    run_all()
