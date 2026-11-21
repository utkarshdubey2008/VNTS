"""
Real-World Vision Benchmark: MNIST (784 Inputs, 10 Classes)
Compares:
1. Deep-OSM (4-Layer Hierarchical Orthogonal Subspace Memory)
2. OSM-RME (Single-Layer Orthogonal Expansion)
3. DATS-SI (Phase 1 Algebraic Target Inversion)
4. Conventional Backprop (AdamW)
5. Conventional Backprop (SGD with Momentum)
"""

import os
import sys
import json
import time
import resource
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import Dict, Any

from src.osm.deep_osm import DeepOSMNetwork
from src.osm.model import OSMNetwork
from src.dats.model import DATSNetwork
from src.dats.trainer import DATSTrainer
from src.baselines.backprop_baseline import StandardMLP, BackpropTrainer


def get_peak_memory_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_mnist_data(n_train: int = 5000, n_test: int = 2000, seed: int = 42):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    torch.manual_seed(seed)
    
    # Extract training samples
    perm_tr = torch.randperm(len(train_set))[:n_train]
    X_tr = train_set.data[perm_tr].float().view(n_train, -1) / 255.0
    y_tr = train_set.targets[perm_tr].long()
    
    # Extract test samples
    perm_te = torch.randperm(len(test_set))[:n_test]
    X_te = test_set.data[perm_te].float().view(n_test, -1) / 255.0
    y_te = test_set.targets[perm_te].long()
    
    return X_tr, y_tr, X_te, y_te


def run_mnist_experiment():
    print(f"\n{'='*70}")
    print("REAL VISION BENCHMARK: MNIST (784-Dimensional Input, 10 Classes)")
    print(f"{'='*70}")
    
    n_train = 5000
    n_test = 2000
    X_tr, y_tr, X_te, y_te = load_mnist_data(n_train=n_train, n_test=n_test, seed=42)
    print(f"Dataset: {n_train} Train samples, {n_test} Test samples (784 features each)")
    
    results = {}

    # 1. Evaluate Conventional Backprop (AdamW)
    print("\n--- 1. Evaluating Conventional Backprop (AdamW) ---")
    bp_adam = StandardMLP([784, 256, 128, 10], seed=42)
    bp_adam_trainer = BackpropTrainer(bp_adam, optimizer_name="adamw", lr=2e-3, weight_decay=1e-4)
    t0 = time.time()
    bp_adam_trainer.fit(X_tr, y_tr, X_te, y_te, epochs=20, batch_size=64)
    time_adam = time.time() - t0
    eval_adam = bp_adam_trainer.evaluate(X_te, y_te)
    results["Backprop_AdamW"] = {
        "test_acc": eval_adam["accuracy"],
        "test_loss": eval_adam["loss"],
        "time_s": time_adam,
        "peak_rss_mb": get_peak_memory_mb()
    }
    print(f"Backprop-AdamW: Test Acc = {eval_adam['accuracy']*100:.2f}% | Test Loss = {eval_adam['loss']:.4f} | Time = {time_adam:.3f}s")

    # 2. Evaluate Conventional Backprop (SGD+Momentum)
    print("\n--- 2. Evaluating Conventional Backprop (SGD+Momentum) ---")
    bp_sgd = StandardMLP([784, 256, 128, 10], seed=42)
    bp_sgd_trainer = BackpropTrainer(bp_sgd, optimizer_name="sgd", lr=0.05, momentum=0.9, weight_decay=1e-4)
    t0 = time.time()
    bp_sgd_trainer.fit(X_tr, y_tr, X_te, y_te, epochs=20, batch_size=64)
    time_sgd = time.time() - t0
    eval_sgd = bp_sgd_trainer.evaluate(X_te, y_te)
    results["Backprop_SGD"] = {
        "test_acc": eval_sgd["accuracy"],
        "test_loss": eval_sgd["loss"],
        "time_s": time_sgd,
        "peak_rss_mb": get_peak_memory_mb()
    }
    print(f"Backprop-SGD:   Test Acc = {eval_sgd['accuracy']*100:.2f}% | Test Loss = {eval_sgd['loss']:.4f} | Time = {time_sgd:.3f}s")

    # 3. Evaluate DATS-SI (Phase 1 Paradigm)
    print("\n--- 3. Evaluating DATS-SI (Phase 1) ---")
    dats = DATSNetwork([784, 256, 128, 10], seed=42)
    dats_trainer = DATSTrainer(dats, beta=0.25, alpha=0.35, gamma=1e-3, reg_lambda=1e-3)
    t0 = time.time()
    dats_trainer.fit(X_tr, y_tr, X_te, y_te, epochs=20, batch_size=None)
    time_dats = time.time() - t0
    eval_dats = dats_trainer.evaluate(X_te, y_te)
    results["DATS_SI"] = {
        "test_acc": eval_dats["accuracy"],
        "test_loss": eval_dats["loss"],
        "time_s": time_dats,
        "peak_rss_mb": get_peak_memory_mb()
    }
    print(f"DATS-SI:        Test Acc = {eval_dats['accuracy']*100:.2f}% | Test Loss = {eval_dats['loss']:.4f} | Time = {time_dats:.3f}s")

    # 4. Evaluate OSM-RME (Phase 2 Successor)
    print("\n--- 4. Evaluating OSM-RME (Single Expansion Layer) ---")
    osm = OSMNetwork(in_dim=784, expansion_dim=1024, out_dim=10, variance_threshold=0.999, seed=42)
    t0 = time.time()
    osm.learn_task(X_tr, y_tr, num_classes=10)
    time_osm = time.time() - t0
    eval_osm = osm.evaluate(X_te, y_te)
    results["OSM_RME"] = {
        "test_acc": eval_osm["accuracy"],
        "test_loss": eval_osm["loss"],
        "time_s": time_osm,
        "peak_rss_mb": get_peak_memory_mb()
    }
    print(f"OSM-RME:        Test Acc = {eval_osm['accuracy']*100:.2f}% | Test Loss = {eval_osm['loss']:.4f} | Time = {time_osm:.3f}s")

    # 5. Evaluate Deep-OSM (Phase 3 Hierarchical Multi-Layer)
    print("\n--- 5. Evaluating Deep-OSM (4-Layer Hierarchical) ---")
    deep_osm = DeepOSMNetwork(
        in_dim=784,
        hidden_dims=[512, 512, 256],
        out_dim=10,
        variance_threshold=0.999,
        reg_lambda=1e-2,
        seed=42
    )
    t0 = time.time()
    deep_osm.learn_task(X_tr, y_tr, num_classes=10)
    time_deep_osm = time.time() - t0
    eval_deep_osm = deep_osm.evaluate(X_te, y_te)
    results["Deep_OSM"] = {
        "test_acc": eval_deep_osm["accuracy"],
        "test_loss": eval_deep_osm["loss"],
        "time_s": time_deep_osm,
        "peak_rss_mb": get_peak_memory_mb()
    }
    print(f"Deep-OSM:       Test Acc = {eval_deep_osm['accuracy']*100:.2f}% | Test Loss = {eval_deep_osm['loss']:.4f} | Time = {time_deep_osm:.3f}s")

    # Save results
    os.makedirs("results", exist_ok=True)
    out_file = "results/mnist_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] MNIST Benchmark complete! Results saved to {out_file}")
    return results


if __name__ == "__main__":
    run_mnist_experiment()
