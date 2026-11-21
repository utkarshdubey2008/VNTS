"""
Comprehensive Phase 2 Deep Validation Suite for DATS-SI.
Evaluates:
1. Multi-Seed Stability (20 seeds)
2. Sample Complexity Scaling (N in [200, 500, 1000, 2500, 5000])
3. Extreme Depth Scaling (L in [4, 8, 16, 32, 64, 128])
4. Width Scaling (W in [16, 32, 64, 128, 256, 512])
5. Continual Learning & Catastrophic Forgetting (Split Digits: 0-4 -> 5-9)
6. Sequential & Online Streaming Learning (Full-batch vs Mini-batch vs Streaming RLS)
7. Out-Of-Distribution (OOD) Robustness (Gaussian perturbation on test features)
8. Memory & Compute Resource Scaling
"""

import os
import sys
import json
import time
import resource
import torch
import numpy as np
from typing import Dict, Any, List

from src.dats.model import DATSNetwork
from src.dats.trainer import DATSTrainer
from src.dats.algebraic_solver import RecursiveLeastSquaresTracker
from src.baselines.backprop_baseline import StandardMLP, BackpropTrainer
from benchmarks.datasets import (
    generate_xor_dataset,
    generate_circles_dataset,
    generate_two_spirals,
    load_digits_dataset
)


def get_peak_memory_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------
# 1. Multi-Seed Statistical Robustness (20 Seeds)
# ---------------------------------------------------------
def run_seed_stability(n_seeds: int = 20) -> Dict[str, Any]:
    print(f"\n{'='*70}\n[EXPERIMENT 1] Multi-Seed Statistical Robustness ({n_seeds} seeds)\n{'='*70}")
    tasks = ["xor", "circles", "digits"]
    results = {}
    
    for task in tasks:
        dats_accs = []
        bp_accs = []
        
        for s in range(n_seeds):
            if task == "xor":
                X_tr, y_tr, X_te, y_te = generate_xor_dataset(n_samples=1000, seed=s)
                dims = [2, 32, 16, 2]
                epochs = 30
            elif task == "circles":
                X_tr, y_tr, X_te, y_te = generate_circles_dataset(n_samples=1000, seed=s)
                dims = [2, 32, 16, 2]
                epochs = 30
            else:
                X_tr, y_tr, X_te, y_te = load_digits_dataset(seed=s)
                dims = [64, 128, 64, 10]
                epochs = 30
                
            # DATS-SI
            dats_m = DATSNetwork(dims, seed=s)
            dats_t = DATSTrainer(dats_m, beta=0.3, alpha=0.35, gamma=1e-3, reg_lambda=1e-3)
            dats_t.fit(X_tr, y_tr, epochs=epochs, batch_size=None)
            dats_acc = dats_t.evaluate(X_te, y_te)["accuracy"]
            dats_accs.append(dats_acc)
            
            # Backprop (AdamW)
            bp_m = StandardMLP(dims, seed=s)
            bp_t = BackpropTrainer(bp_m, optimizer_name="adamw", lr=2e-3)
            bp_t.fit(X_tr, y_tr, epochs=epochs, batch_size=64)
            bp_acc = bp_t.evaluate(X_te, y_te)["accuracy"]
            bp_accs.append(bp_acc)

        results[task] = {
            "dats_mean": float(np.mean(dats_accs)),
            "dats_std": float(np.std(dats_accs)),
            "dats_min": float(np.min(dats_accs)),
            "dats_max": float(np.max(dats_accs)),
            "bp_mean": float(np.mean(bp_accs)),
            "bp_std": float(np.std(bp_accs)),
            "bp_min": float(np.min(bp_accs)),
            "bp_max": float(np.max(bp_accs)),
        }
        print(f"Task: {task:8s} | DATS: {results[task]['dats_mean']*100:.2f}% ± {results[task]['dats_std']*100:.2f}% (min: {results[task]['dats_min']*100:.1f}%) | "
              f"AdamW: {results[task]['bp_mean']*100:.2f}% ± {results[task]['bp_std']*100:.2f}% (min: {results[task]['bp_min']*100:.1f}%)")
              
    return results


# ---------------------------------------------------------
# 2. Sample Complexity Scaling
# ---------------------------------------------------------
def run_sample_scaling() -> Dict[str, Any]:
    print(f"\n{'='*70}\n[EXPERIMENT 2] Sample Complexity Scaling\n{'='*70}")
    sample_sizes = [200, 500, 1000, 2500, 5000]
    results = {}
    
    for n in sample_sizes:
        X_tr, y_tr, X_te, y_te = generate_circles_dataset(n_samples=n, seed=42)
        dims = [2, 32, 16, 2]
        
        # DATS-SI
        t0 = time.time()
        dats_m = DATSNetwork(dims, seed=42)
        dats_t = DATSTrainer(dats_m, beta=0.35, alpha=0.35, gamma=1e-3, reg_lambda=1e-3)
        dats_t.fit(X_tr, y_tr, epochs=25, batch_size=None)
        dats_time = time.time() - t0
        dats_acc = dats_t.evaluate(X_te, y_te)["accuracy"]
        
        # Backprop
        t0 = time.time()
        bp_m = StandardMLP(dims, seed=42)
        bp_t = BackpropTrainer(bp_m, optimizer_name="adamw", lr=2e-3)
        bp_t.fit(X_tr, y_tr, epochs=25, batch_size=64)
        bp_time = time.time() - t0
        bp_acc = bp_t.evaluate(X_te, y_te)["accuracy"]
        
        results[f"N_{n}"] = {
            "N": n,
            "dats_acc": dats_acc, "dats_time": dats_time,
            "bp_acc": bp_acc, "bp_time": bp_time
        }
        print(f"N = {n:5d} | DATS Acc: {dats_acc*100:.2f}% ({dats_time:.3f}s) | AdamW Acc: {bp_acc*100:.2f}% ({bp_time:.3f}s)")
        
    return results


# ---------------------------------------------------------
# 3. Extreme Depth Scaling (L in [4, 8, 16, 32, 64, 128])
# ---------------------------------------------------------
def run_extreme_depth() -> Dict[str, Any]:
    print(f"\n{'='*70}\n[EXPERIMENT 3] Extreme Depth Scaling (L in [4, 8, 16, 32, 64, 128])\n{'='*70}")
    depths = [4, 8, 16, 32, 64, 128]
    X_tr, y_tr, X_te, y_te = generate_circles_dataset(n_samples=1000, seed=42)
    results = {}
    
    for d in depths:
        dims = [2] + [32] * (d - 1) + [2]
        
        # DATS-SI
        t0 = time.time()
        dats_m = DATSNetwork(dims, seed=42)
        dats_t = DATSTrainer(dats_m, beta=0.25, alpha=0.3, gamma=1e-3, reg_lambda=1e-3)
        dats_h = dats_t.fit(X_tr, y_tr, epochs=25, batch_size=None)
        dats_time = time.time() - t0
        dats_eval = dats_t.evaluate(X_te, y_te)
        
        # Backprop (AdamW)
        t0 = time.time()
        bp_m = StandardMLP(dims, seed=42)
        bp_t = BackpropTrainer(bp_m, optimizer_name="adamw", lr=1e-3)
        bp_h = bp_t.fit(X_tr, y_tr, epochs=25, batch_size=64)
        bp_time = time.time() - t0
        bp_eval = bp_t.evaluate(X_te, y_te)
        
        results[f"L_{d}"] = {
            "depth": d,
            "dats_acc": dats_eval["accuracy"],
            "dats_loss": dats_eval["loss"],
            "dats_time": dats_time,
            "bp_acc": bp_eval["accuracy"],
            "bp_loss": bp_eval["loss"],
            "bp_time": bp_time
        }
        print(f"Depth L = {d:3d} | DATS: Acc = {dats_eval['accuracy']*100:.2f}%, Loss = {dats_eval['loss']:.4f} ({dats_time:.3f}s) | "
              f"AdamW: Acc = {bp_eval['accuracy']*100:.2f}%, Loss = {bp_eval['loss']:.4f} ({bp_time:.3f}s)")
              
    return results


# ---------------------------------------------------------
# 4. Width Scaling (W in [16, 32, 64, 128, 256, 512])
# ---------------------------------------------------------
def run_width_scaling() -> Dict[str, Any]:
    print(f"\n{'='*70}\n[EXPERIMENT 4] Width Scaling (W in [16, 32, 64, 128, 256, 512])\n{'='*70}")
    widths = [16, 32, 64, 128, 256, 512]
    X_tr, y_tr, X_te, y_te = generate_circles_dataset(n_samples=1000, seed=42)
    results = {}
    
    for w in widths:
        dims = [2, w, w, 2]
        
        # DATS-SI
        t0 = time.time()
        dats_m = DATSNetwork(dims, seed=42)
        dats_t = DATSTrainer(dats_m, beta=0.35, alpha=0.35, gamma=1e-3, reg_lambda=1e-3)
        dats_t.fit(X_tr, y_tr, epochs=25, batch_size=None)
        dats_time = time.time() - t0
        dats_acc = dats_t.evaluate(X_te, y_te)["accuracy"]
        
        # Backprop
        t0 = time.time()
        bp_m = StandardMLP(dims, seed=42)
        bp_t = BackpropTrainer(bp_m, optimizer_name="adamw", lr=2e-3)
        bp_t.fit(X_tr, y_tr, epochs=25, batch_size=64)
        bp_time = time.time() - t0
        bp_acc = bp_t.evaluate(X_te, y_te)["accuracy"]
        
        results[f"W_{w}"] = {
            "width": w,
            "dats_acc": dats_acc, "dats_time": dats_time,
            "bp_acc": bp_acc, "bp_time": bp_time
        }
        print(f"Width W = {w:4d} | DATS: Acc = {dats_acc*100:.2f}% ({dats_time:.3f}s) | AdamW: Acc = {bp_acc*100:.2f}% ({bp_time:.3f}s)")
        
    return results


# ---------------------------------------------------------
# 5. Continual Learning & Catastrophic Forgetting
# ---------------------------------------------------------
def run_continual_learning() -> Dict[str, Any]:
    print(f"\n{'='*70}\n[EXPERIMENT 5] Continual Learning & Catastrophic Forgetting (Split-Digits 0-4 -> 5-9)\n{'='*70}")
    X_tr, y_tr, X_te, y_te = load_digits_dataset(seed=42)
    
    # Split Task 1: classes 0-4; Task 2: classes 5-9
    mask_tr_1 = y_tr < 5
    mask_te_1 = y_te < 5
    X_tr_1, y_tr_1 = X_tr[mask_tr_1], y_tr[mask_tr_1]
    X_te_1, y_te_1 = X_te[mask_te_1], y_te[mask_te_1]
    
    mask_tr_2 = y_tr >= 5
    mask_te_2 = y_te >= 5
    X_tr_2, y_tr_2 = X_tr[mask_tr_2], y_tr[mask_tr_2]
    X_te_2, y_te_2 = X_te[mask_te_2], y_te[mask_te_2]
    
    dims = [64, 128, 64, 10]
    
    # --- DATS-SI Continual Evaluation ---
    dats_m = DATSNetwork(dims, seed=42)
    dats_t = DATSTrainer(dats_m, beta=0.25, alpha=0.35, gamma=1e-3, reg_lambda=1e-3)
    
    # Step 1: Train on Task 1
    dats_t.fit(X_tr_1, y_tr_1, epochs=30, batch_size=None)
    dats_t1_after_t1 = dats_t.evaluate(X_te_1, y_te_1)["accuracy"]
    
    # Step 2: Train on Task 2 without Task 1 data
    dats_t.fit(X_tr_2, y_tr_2, epochs=30, batch_size=None)
    dats_t2_after_t2 = dats_t.evaluate(X_te_2, y_te_2)["accuracy"]
    dats_t1_after_t2 = dats_t.evaluate(X_te_1, y_te_1)["accuracy"]
    dats_forgetting = dats_t1_after_t1 - dats_t1_after_t2
    
    # --- Backprop Continual Evaluation ---
    bp_m = StandardMLP(dims, seed=42)
    bp_t = BackpropTrainer(bp_m, optimizer_name="adamw", lr=1e-3)
    
    bp_t.fit(X_tr_1, y_tr_1, epochs=30, batch_size=32)
    bp_t1_after_t1 = bp_t.evaluate(X_te_1, y_te_1)["accuracy"]
    
    bp_t.fit(X_tr_2, y_tr_2, epochs=30, batch_size=32)
    bp_t2_after_t2 = bp_t.evaluate(X_te_2, y_te_2)["accuracy"]
    bp_t1_after_t2 = bp_t.evaluate(X_te_1, y_te_1)["accuracy"]
    bp_forgetting = bp_t1_after_t1 - bp_t1_after_t2
    
    results = {
        "DATS": {
            "Task1_initial_acc": dats_t1_after_t1,
            "Task2_final_acc": dats_t2_after_t2,
            "Task1_retained_acc": dats_t1_after_t2,
            "forgetting_drop": dats_forgetting
        },
        "Backprop_AdamW": {
            "Task1_initial_acc": bp_t1_after_t1,
            "Task2_final_acc": bp_t2_after_t2,
            "Task1_retained_acc": bp_t1_after_t2,
            "forgetting_drop": bp_forgetting
        }
    }
    
    print(f"DATS-SI:  Task 1 Initial: {dats_t1_after_t1*100:.2f}% -> Retained after Task 2: {dats_t1_after_t2*100:.2f}% (Forgetting: {dats_forgetting*100:.2f}%)")
    print(f"Backprop: Task 1 Initial: {bp_t1_after_t1*100:.2f}% -> Retained after Task 2: {bp_t1_after_t2*100:.2f}% (Forgetting: {bp_forgetting*100:.2f}%)")
    return results


# ---------------------------------------------------------
# 6. Sequential & Online Streaming Learning
# ---------------------------------------------------------
def run_sequential_online() -> Dict[str, Any]:
    print(f"\n{'='*70}\n[EXPERIMENT 6] Sequential & Online Streaming Learning\n{'='*70}")
    X_tr, y_tr, X_te, y_te = generate_circles_dataset(n_samples=1000, seed=42)
    dims = [2, 32, 16, 2]
    results = {}
    
    # 1. Full-batch DATS-SI
    m_full = DATSNetwork(dims, seed=42)
    t_full = DATSTrainer(m_full, beta=0.35, alpha=0.35, gamma=1e-3, reg_lambda=1e-3)
    t_full.fit(X_tr, y_tr, epochs=30, batch_size=None)
    results["full_batch_acc"] = t_full.evaluate(X_te, y_te)["accuracy"]
    
    # 2. Mini-batch DATS-SI (batch size = 32)
    m_mini = DATSNetwork(dims, seed=42)
    t_mini = DATSTrainer(m_mini, beta=0.35, alpha=0.2, gamma=1e-3, reg_lambda=1e-2)
    t_mini.fit(X_tr, y_tr, epochs=30, batch_size=32)
    results["mini_batch_32_acc"] = t_mini.evaluate(X_te, y_te)["accuracy"]
    
    # 3. Streaming Recursive Least Squares (RLS) Single-Pass
    # We test whether single-pass RLS on the final output layer preserves representations
    # (using fixed random or frozen feature extractor)
    dats_feat = DATSNetwork(dims, seed=42)
    rls = RecursiveLeastSquaresTracker(16, 2, delta=10.0, forgetting_factor=0.999)
    
    # Single streaming pass over samples
    for i in range(len(X_tr)):
        x_i = X_tr[i:i+1]
        y_i = y_tr[i:i+1]
        
        # Forward through hidden layers
        _, hs = dats_feat.forward_intermediates(x_i)
        h_last = hs[-2]  # output of layer before final: shape (1, 16)
        ones = torch.ones((1, 1))
        h_aug = torch.cat([h_last, ones], dim=1)
        
        target = torch.full((1, 2), -1.0)
        target[0, y_i.item()] = 1.0
        rls.update(h_aug, target)
        
    # Evaluate streaming RLS
    _, hs_te = dats_feat.forward_intermediates(X_te)
    H_te_last = hs_te[-2]
    ones_te = torch.ones((len(X_te), 1))
    H_te_aug = torch.cat([H_te_last, ones_te], dim=1)
    preds_rls = H_te_aug @ rls.W_aug
    rls_acc = (preds_rls.argmax(1) == y_te).float().mean().item()
    results["streaming_rls_single_pass_acc"] = rls_acc
    
    print(f"Full-batch DATS Acc:        {results['full_batch_acc']*100:.2f}%")
    print(f"Mini-batch DATS (B=32) Acc: {results['mini_batch_32_acc']*100:.2f}%")
    print(f"Single-pass Streaming RLS:  {results['streaming_rls_single_pass_acc']*100:.2f}%")
    return results


# ---------------------------------------------------------
# 7. Out-Of-Distribution (OOD) Robustness
# ---------------------------------------------------------
def run_ood_robustness() -> Dict[str, Any]:
    print(f"\n{'='*70}\n[EXPERIMENT 7] Out-Of-Distribution (OOD) Robustness\n{'='*70}")
    X_tr, y_tr, X_te, y_te = generate_circles_dataset(n_samples=1000, seed=42)
    dims = [2, 32, 16, 2]
    
    # Train both models on clean data
    dats_m = DATSNetwork(dims, seed=42)
    dats_t = DATSTrainer(dats_m, beta=0.35, alpha=0.35, gamma=1e-3, reg_lambda=1e-3)
    dats_t.fit(X_tr, y_tr, epochs=30, batch_size=None)
    
    bp_m = StandardMLP(dims, seed=42)
    bp_t = BackpropTrainer(bp_m, optimizer_name="adamw", lr=2e-3)
    bp_t.fit(X_tr, y_tr, epochs=30, batch_size=64)
    
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    results = {}
    
    for sigma in noise_levels:
        X_te_noisy = X_te + sigma * torch.randn_like(X_te)
        d_acc = dats_t.evaluate(X_te_noisy, y_te)["accuracy"]
        bp_acc = bp_t.evaluate(X_te_noisy, y_te)["accuracy"]
        results[f"noise_{sigma}"] = {"sigma": sigma, "dats_acc": d_acc, "bp_acc": bp_acc}
        print(f"Noise sigma = {sigma:4.2f} | DATS-SI Acc: {d_acc*100:.2f}% | Backprop Acc: {bp_acc*100:.2f}%")
        
    return results


# ---------------------------------------------------------
# 8. Memory & Compute Resource Scaling
# ---------------------------------------------------------
def run_resource_scaling() -> Dict[str, Any]:
    print(f"\n{'='*70}\n[EXPERIMENT 8] Memory & Compute Resource Scaling\n{'='*70}")
    dims = [64, 256, 128, 10]
    X_tr, y_tr, X_te, y_te = load_digits_dataset(seed=42)
    
    # Memory benchmark
    m_init = get_peak_memory_mb()
    
    t0 = time.time()
    dats_m = DATSNetwork(dims, seed=42)
    dats_t = DATSTrainer(dats_m, beta=0.25, alpha=0.35, gamma=1e-3, reg_lambda=1e-3)
    dats_t.fit(X_tr, y_tr, epochs=40, batch_size=None)
    dats_time = time.time() - t0
    m_dats = get_peak_memory_mb()
    
    t0 = time.time()
    bp_m = StandardMLP(dims, seed=42)
    bp_t = BackpropTrainer(bp_m, optimizer_name="adamw", lr=1e-3)
    bp_t.fit(X_tr, y_tr, epochs=40, batch_size=64)
    bp_time = time.time() - t0
    m_bp = get_peak_memory_mb()
    
    results = {
        "dats_total_time": dats_time,
        "dats_peak_rss_mb": m_dats,
        "bp_total_time": bp_time,
        "bp_peak_rss_mb": m_bp
    }
    print(f"DATS-SI:  Time = {dats_time:.3f}s | Peak RSS = {m_dats:.1f} MB")
    print(f"Backprop: Time = {bp_time:.3f}s | Peak RSS = {m_bp:.1f} MB")
    return results


def run_full_phase2_suite():
    os.makedirs("results", exist_ok=True)
    suite_results = {}
    
    suite_results["seed_stability"] = run_seed_stability(n_seeds=20)
    suite_results["sample_scaling"] = run_sample_scaling()
    suite_results["extreme_depth"] = run_extreme_depth()
    suite_results["width_scaling"] = run_width_scaling()
    suite_results["continual_learning"] = run_continual_learning()
    suite_results["sequential_online"] = run_sequential_online()
    suite_results["ood_robustness"] = run_ood_robustness()
    suite_results["resource_scaling"] = run_resource_scaling()
    
    out_file = "results/deep_validation_results.json"
    with open(out_file, "w") as f:
        json.dump(suite_results, f, indent=2)
    print(f"\n{'='*70}\n[SUCCESS] Deep Validation Suite completed! Results saved to {out_file}\n{'='*70}")


if __name__ == "__main__":
    run_full_phase2_suite()
