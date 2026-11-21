"""
Empirical Benchmark: OSM-RME Successor vs DATS-SI vs Conventional Backpropagation (AdamW).
Evaluates Continual Learning, Catastrophic Forgetting, and Sequential Adaptability.
"""

import os
import json
import time
import torch
import numpy as np
from typing import Dict, Any, List

from src.osm.model import OSMNetwork
from src.dats.model import DATSNetwork
from src.dats.trainer import DATSTrainer
from src.baselines.backprop_baseline import StandardMLP, BackpropTrainer
from benchmarks.datasets import load_digits_dataset


def run_3_task_continual_benchmark():
    print(f"\n{'='*70}")
    print("3-TASK CONTINUAL LEARNING BENCHMARK (Split-Digits: [0,1,2] -> [3,4,5] -> [6,7,8,9])")
    print(f"{'='*70}")
    
    X_tr, y_tr, X_te, y_te = load_digits_dataset(seed=42)
    
    # Task 1: classes 0, 1, 2
    m1_tr = (y_tr <= 2)
    m1_te = (y_te <= 2)
    X1_tr, y1_tr, X1_te, y1_te = X_tr[m1_tr], y_tr[m1_tr], X_te[m1_te], y_te[m1_te]
    
    # Task 2: classes 3, 4, 5
    m2_tr = (y_tr >= 3) & (y_tr <= 5)
    m2_te = (y_te >= 3) & (y_te <= 5)
    X2_tr, y2_tr, X2_te, y2_te = X_tr[m2_tr], y_tr[m2_tr], X_te[m2_te], y_te[m2_te]
    
    # Task 3: classes 6, 7, 8, 9
    m3_tr = (y_tr >= 6)
    m3_te = (y_te >= 6)
    X3_tr, y3_tr, X3_te, y3_te = X_tr[m3_tr], y_tr[m3_tr], X_te[m3_te], y_te[m3_te]
    
    print(f"Task 1 samples: {len(X1_tr)} train / {len(X1_te)} test")
    print(f"Task 2 samples: {len(X2_tr)} train / {len(X2_te)} test")
    print(f"Task 3 samples: {len(X3_tr)} train / {len(X3_te)} test")
    
    results = {}

    # =========================================================================
    # 1. SUCCESSOR PARADIGM: OSM-RME (Orthogonal Subspace Memory)
    # =========================================================================
    print("\n--- Training OSM-RME (Successor Paradigm) ---")
    t0 = time.time()
    osm = OSMNetwork(in_dim=64, expansion_dim=512, out_dim=10, variance_threshold=0.99, seed=42)
    
    # Train Task 1
    osm.learn_task(X1_tr, y1_tr, num_classes=10)
    acc_t1_init_osm = osm.evaluate(X1_te, y1_te)["accuracy"]
    
    # Train Task 2
    osm.learn_task(X2_tr, y2_tr, num_classes=10)
    acc_t2_init_osm = osm.evaluate(X2_te, y2_te)["accuracy"]
    acc_t1_after_t2_osm = osm.evaluate(X1_te, y1_te)["accuracy"]
    
    # Train Task 3
    osm.learn_task(X3_tr, y3_tr, num_classes=10)
    acc_t3_init_osm = osm.evaluate(X3_te, y3_te)["accuracy"]
    
    # Final evaluations across all 3 tasks
    acc_t1_final_osm = osm.evaluate(X1_te, y1_te)["accuracy"]
    acc_t2_final_osm = osm.evaluate(X2_te, y2_te)["accuracy"]
    acc_t3_final_osm = osm.evaluate(X3_te, y3_te)["accuracy"]
    avg_retained_osm = (acc_t1_final_osm + acc_t2_final_osm + acc_t3_final_osm) / 3.0
    osm_time = time.time() - t0
    
    results["OSM_RME"] = {
        "T1_initial": acc_t1_init_osm,
        "T2_initial": acc_t2_init_osm,
        "T3_initial": acc_t3_init_osm,
        "T1_final": acc_t1_final_osm,
        "T2_final": acc_t2_final_osm,
        "T3_final": acc_t3_final_osm,
        "average_retained": avg_retained_osm,
        "total_time_s": osm_time
    }
    print(f"OSM-RME Final Retained: T1={acc_t1_final_osm*100:.2f}%, T2={acc_t2_final_osm*100:.2f}%, T3={acc_t3_final_osm*100:.2f}% | "
          f"Average: {avg_retained_osm*100:.2f}% (Time: {osm_time:.3f}s)")

    # =========================================================================
    # 2. DATS-SI PARADIGM
    # =========================================================================
    print("\n--- Training DATS-SI ---")
    t0 = time.time()
    dats = DATSNetwork([64, 128, 64, 10], seed=42)
    dats_trainer = DATSTrainer(dats, beta=0.25, alpha=0.35, gamma=1e-3, reg_lambda=1e-3)
    
    dats_trainer.fit(X1_tr, y1_tr, epochs=25, batch_size=None)
    acc_t1_init_dats = dats_trainer.evaluate(X1_te, y1_te)["accuracy"]
    
    dats_trainer.fit(X2_tr, y2_tr, epochs=25, batch_size=None)
    acc_t2_init_dats = dats_trainer.evaluate(X2_te, y2_te)["accuracy"]
    
    dats_trainer.fit(X3_tr, y3_tr, epochs=25, batch_size=None)
    acc_t3_init_dats = dats_trainer.evaluate(X3_te, y3_te)["accuracy"]
    
    acc_t1_final_dats = dats_trainer.evaluate(X1_te, y1_te)["accuracy"]
    acc_t2_final_dats = dats_trainer.evaluate(X2_te, y2_te)["accuracy"]
    acc_t3_final_dats = dats_trainer.evaluate(X3_te, y3_te)["accuracy"]
    avg_retained_dats = (acc_t1_final_dats + acc_t2_final_dats + acc_t3_final_dats) / 3.0
    dats_time = time.time() - t0
    
    results["DATS_SI"] = {
        "T1_initial": acc_t1_init_dats,
        "T2_initial": acc_t2_init_dats,
        "T3_initial": acc_t3_init_dats,
        "T1_final": acc_t1_final_dats,
        "T2_final": acc_t2_final_dats,
        "T3_final": acc_t3_final_dats,
        "average_retained": avg_retained_dats,
        "total_time_s": dats_time
    }
    print(f"DATS-SI Final Retained: T1={acc_t1_final_dats*100:.2f}%, T2={acc_t2_final_dats*100:.2f}%, T3={acc_t3_final_dats*100:.2f}% | "
          f"Average: {avg_retained_dats*100:.2f}% (Time: {dats_time:.3f}s)")

    # =========================================================================
    # 3. CONVENTIONAL BACKPROPAGATION (AdamW)
    # =========================================================================
    print("\n--- Training Conventional Backpropagation (AdamW) ---")
    t0 = time.time()
    bp = StandardMLP([64, 128, 64, 10], seed=42)
    bp_trainer = BackpropTrainer(bp, optimizer_name="adamw", lr=1e-3)
    
    bp_trainer.fit(X1_tr, y1_tr, epochs=25, batch_size=32)
    acc_t1_init_bp = bp_trainer.evaluate(X1_te, y1_te)["accuracy"]
    
    bp_trainer.fit(X2_tr, y2_tr, epochs=25, batch_size=32)
    acc_t2_init_bp = bp_trainer.evaluate(X2_te, y2_te)["accuracy"]
    
    bp_trainer.fit(X3_tr, y3_tr, epochs=25, batch_size=32)
    acc_t3_init_bp = bp_trainer.evaluate(X3_te, y3_te)["accuracy"]
    
    acc_t1_final_bp = bp_trainer.evaluate(X1_te, y1_te)["accuracy"]
    acc_t2_final_bp = bp_trainer.evaluate(X2_te, y2_te)["accuracy"]
    acc_t3_final_bp = bp_trainer.evaluate(X3_te, y3_te)["accuracy"]
    avg_retained_bp = (acc_t1_final_bp + acc_t2_final_bp + acc_t3_final_bp) / 3.0
    bp_time = time.time() - t0
    
    results["Backprop_AdamW"] = {
        "T1_initial": acc_t1_init_bp,
        "T2_initial": acc_t2_init_bp,
        "T3_initial": acc_t3_init_bp,
        "T1_final": acc_t1_final_bp,
        "T2_final": acc_t2_final_bp,
        "T3_final": acc_t3_final_bp,
        "average_retained": avg_retained_bp,
        "total_time_s": bp_time
    }
    print(f"Backprop-AdamW Final:   T1={acc_t1_final_bp*100:.2f}%, T2={acc_t2_final_bp*100:.2f}%, T3={acc_t3_final_bp*100:.2f}% | "
          f"Average: {avg_retained_bp*100:.2f}% (Time: {bp_time:.3f}s)")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/continual_successor_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return results


if __name__ == "__main__":
    run_3_task_continual_benchmark()
