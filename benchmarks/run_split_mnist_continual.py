"""
Split-MNIST 5-Task Continual Learning Benchmark.
Curriculum:
Task 1: [0, 1]
Task 2: [2, 3]
Task 3: [4, 5]
Task 4: [6, 7]
Task 5: [8, 9]

Compares:
1. Deep-OSM (Hierarchical Orthogonal Subspace Memory)
2. Conventional Backpropagation (AdamW)
"""

import os
import json
import time
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import Dict, List, Any

from src.osm.deep_osm import DeepOSMNetwork
from src.baselines.backprop_baseline import StandardMLP, BackpropTrainer


def load_mnist_all():
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    X_tr = train_set.data.float().view(-1, 784) / 255.0
    y_tr = train_set.targets.long()
    
    X_te = test_set.data.float().view(-1, 784) / 255.0
    y_te = test_set.targets.long()
    
    return X_tr, y_tr, X_te, y_te


def run_split_mnist():
    print(f"\n{'='*70}")
    print("5-TASK SPLIT-MNIST CONTINUAL LEARNING BENCHMARK")
    print("Tasks: [0,1] -> [2,3] -> [4,5] -> [6,7] -> [8,9]")
    print(f"{'='*70}")
    
    X_tr, y_tr, X_te, y_te = load_mnist_all()
    
    # Create 5 task datasets
    task_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    tasks = []
    for c1, c2 in task_pairs:
        tr_mask = (y_tr == c1) | (y_tr == c2)
        te_mask = (y_te == c1) | (y_te == c2)
        # Sample 1000 train samples per task for fast benchmark execution
        idx_tr = torch.where(tr_mask)[0][:1000]
        idx_te = torch.where(te_mask)[0][:400]
        tasks.append((X_tr[idx_tr], y_tr[idx_tr], X_te[idx_te], y_te[idx_te]))

    # --- 1. Evaluate Deep-OSM ---
    print("\n--- Training Deep-OSM on 5 Sequential Tasks ---")
    deep_osm = DeepOSMNetwork(
        in_dim=784,
        hidden_dims=[1024, 1024, 512],
        out_dim=10,
        variance_threshold=0.95,
        reg_lambda=1e-2,
        seed=42
    )
    
    t0 = time.time()
    task_initial_accs_osm = []
    for t_idx, (x_t, y_t, x_te, y_te) in enumerate(tasks):
        deep_osm.learn_task(x_t, y_t, num_classes=10)
        acc_init = deep_osm.evaluate(x_te, y_te)["accuracy"]
        task_initial_accs_osm.append(acc_init)
        print(f"Task {t_idx+1} {task_pairs[t_idx]} Learned -> Immediate Acc: {acc_init*100:.2f}%")
        
    time_osm = time.time() - t0
    
    # Final evaluation on all 5 tasks
    task_final_accs_osm = []
    for t_idx, (_, _, x_te, y_te) in enumerate(tasks):
        acc_final = deep_osm.evaluate(x_te, y_te)["accuracy"]
        task_final_accs_osm.append(acc_final)
        
    avg_osm = sum(task_final_accs_osm) / len(task_final_accs_osm)
    print(f"\nDeep-OSM Final Retained Accuracies across Tasks 1-5: {[round(a*100, 1) for a in task_final_accs_osm]}")
    print(f"Deep-OSM Average Retained Accuracy: {avg_osm*100:.2f}% (Total Time: {time_osm:.3f}s)")

    # --- 2. Evaluate Conventional Backprop (AdamW) ---
    print("\n--- Training Backprop (AdamW) on 5 Sequential Tasks ---")
    bp = StandardMLP([784, 256, 128, 10], seed=42)
    bp_trainer = BackpropTrainer(bp, optimizer_name="adamw", lr=1e-3)
    
    t0 = time.time()
    task_initial_accs_bp = []
    for t_idx, (x_t, y_t, x_te, y_te) in enumerate(tasks):
        bp_trainer.fit(x_t, y_t, epochs=15, batch_size=32)
        acc_init = bp_trainer.evaluate(x_te, y_te)["accuracy"]
        task_initial_accs_bp.append(acc_init)
        print(f"Task {t_idx+1} {task_pairs[t_idx]} Learned -> Immediate Acc: {acc_init*100:.2f}%")
        
    time_bp = time.time() - t0
    
    # Final evaluation on all 5 tasks
    task_final_accs_bp = []
    for t_idx, (_, _, x_te, y_te) in enumerate(tasks):
        acc_final = bp_trainer.evaluate(x_te, y_te)["accuracy"]
        task_final_accs_bp.append(acc_final)
        
    avg_bp = sum(task_final_accs_bp) / len(task_final_accs_bp)
    print(f"\nBackprop Final Retained Accuracies across Tasks 1-5: {[round(a*100, 1) for a in task_final_accs_bp]}")
    print(f"Backprop Average Retained Accuracy: {avg_bp*100:.2f}% (Total Time: {time_bp:.3f}s)")

    results = {
        "Deep_OSM": {
            "initial_accs": task_initial_accs_osm,
            "final_accs": task_final_accs_osm,
            "average_retained": avg_osm,
            "time_s": time_osm
        },
        "Backprop_AdamW": {
            "initial_accs": task_initial_accs_bp,
            "final_accs": task_final_accs_bp,
            "average_retained": avg_bp,
            "time_s": time_bp
        }
    }
    
    os.makedirs("results", exist_ok=True)
    out_file = "results/split_mnist_continual_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Split-MNIST benchmark results saved to {out_file}")
    return results


if __name__ == "__main__":
    run_split_mnist()
