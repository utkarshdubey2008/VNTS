"""
VNTS Experiment: Depth Scaling & Memory Invariance Test
Evaluates Backpropagation vs Forward Residual Plasticity (FRP)
across network depths L in [2, 4, 8, 16, 32] layers.
Measures peak process RSS memory, gradient stability, convergence loss, and accuracy.
"""

import os
import json
import torch
import torch.nn as nn
from vnts.models import CleanMLP
from vnts.benchmarks import generate_multimanifold_dataset
from vnts.trainers import BackpropTrainer, FRPTrainer
from vnts.metrics import ExperimentTracker, compute_classification_accuracy

def run_depth_scaling_experiment(depths=[2, 4, 8, 16, 32], epochs=25, batch_size=128):
    print("=" * 65)
    print("  EXPERIMENT 1: DEPTH SCALING & MEMORY INVARIANCE TEST")
    print("=" * 65)

    (train_X, train_y), (test_X, test_y) = generate_multimanifold_dataset(n_samples=4000, n_classes=4, seed=42)
    n_train = len(train_X)
    hidden_dim = 64
    criterion = nn.CrossEntropyLoss()

    results = []

    for depth in depths:
        hidden_dims = [hidden_dim] * depth
        print(f"\n---> Testing Network Depth: {depth} layers (hidden_dims={hidden_dim} x {depth})")

        # 1. Backpropagation (AdamW)
        model_bp = CleanMLP(in_features=4, hidden_dims=hidden_dims, out_features=4, task='classification')
        model_bp.reset_parameters(seed=42)
        trainer_bp = BackpropTrainer(model_bp, optimizer_name='adamw', lr=2e-3)
        tracker_bp = ExperimentTracker(f"Backprop-Depth-{depth}")
        tracker_bp.start()

        n_batches = (n_train + batch_size - 1) // batch_size
        for epoch in range(epochs):
            perm = torch.randperm(n_train)
            X_shuff = train_X[perm]
            y_shuff = train_y[perm]

            epoch_loss = 0.0
            epoch_flops = 0
            for b in range(n_batches):
                bx = X_shuff[b*batch_size : (b+1)*batch_size]
                by = y_shuff[b*batch_size : (b+1)*batch_size]
                loss, _, flops = trainer_bp.train_step(bx, by, criterion)
                epoch_loss += loss
                epoch_flops += flops
            tracker_bp.record_step(epoch_loss / n_batches, epoch_flops)

        tracker_bp.stop()
        with torch.no_grad():
            preds_bp = model_bp(test_X)
            test_loss_bp = criterion(preds_bp, test_y).item()
            test_acc_bp = compute_classification_accuracy(preds_bp, test_y)
        tracker_bp.record_eval(test_loss_bp, test_acc_bp)
        res_bp = tracker_bp.summary()
        print(f"  [Backprop] Depth {depth}: Test Acc = {test_acc_bp*100:.1f}% | Test Loss = {test_loss_bp:.4f} | Peak RSS = {res_bp['peak_rss_mb']:.1f} MB | Time = {res_bp['total_time_s']:.2f}s")

        # 2. Forward Residual Plasticity (FRP)
        model_frp = CleanMLP(in_features=4, hidden_dims=hidden_dims, out_features=4, task='classification')
        model_frp.reset_parameters(seed=42)
        trainer_frp = FRPTrainer(model_frp, lr=2e-3, alpha_rep=0.01)
        tracker_frp = ExperimentTracker(f"FRP-Depth-{depth}")
        tracker_frp.start()

        for epoch in range(epochs):
            perm = torch.randperm(n_train)
            X_shuff = train_X[perm]
            y_shuff = train_y[perm]

            epoch_loss = 0.0
            epoch_flops = 0
            for b in range(n_batches):
                bx = X_shuff[b*batch_size : (b+1)*batch_size]
                by = y_shuff[b*batch_size : (b+1)*batch_size]
                loss, _, flops = trainer_frp.train_step(bx, by, criterion)
                epoch_loss += loss
                epoch_flops += flops
            tracker_frp.record_step(epoch_loss / n_batches, epoch_flops)

        tracker_frp.stop()
        with torch.no_grad():
            preds_frp = model_frp(test_X)
            test_loss_frp = criterion(preds_frp, test_y).item()
            test_acc_frp = compute_classification_accuracy(preds_frp, test_y)
        tracker_frp.record_eval(test_loss_frp, test_acc_frp)
        res_frp = tracker_frp.summary()
        print(f"  [FRP]      Depth {depth}: Test Acc = {test_acc_frp*100:.1f}% | Test Loss = {test_loss_frp:.4f} | Peak RSS = {res_frp['peak_rss_mb']:.1f} MB | Time = {res_frp['total_time_s']:.2f}s")

        results.append({
            "depth": depth,
            "backprop": res_bp,
            "frp": res_frp
        })

    out_file = "/root/vnts_clone/results/exp_depth_scaling.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDepth scaling results saved to {out_file}")
    return results

if __name__ == "__main__":
    run_depth_scaling_experiment()
