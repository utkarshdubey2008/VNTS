"""
VNTS Experiment: Real-World Vision Benchmark (MNIST)
Evaluates Backpropagation (AdamW, SGD), Direct Feedback Alignment (DFA),
and Forward Residual Plasticity (FRP) on 28x28 handwritten digit recognition.
"""

import os
import json
import torch
import torch.nn as nn
from vnts.models import CleanMLP
from vnts.mnist_loader import load_mnist_data
from vnts.trainers import BackpropTrainer, DFATrainer, FRPTrainer
from vnts.metrics import ExperimentTracker, compute_classification_accuracy

def run_mnist_experiment(epochs=15, batch_size=128):
    print("=" * 65)
    print("  EXPERIMENT 2: REAL-WORLD VISION BENCHMARK (MNIST)")
    print("=" * 65)

    (train_X, train_y), (test_X, test_y) = load_mnist_data(n_train=10000, n_test=2000)
    n_train = len(train_X)
    hidden_dims = [256, 128]
    in_features = 784
    out_features = 10
    criterion = nn.CrossEntropyLoss()

    algorithms = [
        ("Backprop-AdamW", lambda m: BackpropTrainer(m, optimizer_name='adamw', lr=1e-3)),
        ("Backprop-SGD", lambda m: BackpropTrainer(m, optimizer_name='sgd', lr=0.05)),
        ("Direct Feedback Alignment (DFA)", lambda m: DFATrainer(m, lr=1e-3)),
        ("Forward Residual Plasticity (FRP)", lambda m: FRPTrainer(m, lr=1e-3, alpha_rep=0.01))
    ]

    mnist_results = []

    for name, init_fn in algorithms:
        print(f"\n---> Testing [{name}] on MNIST (10,000 images, {epochs} epochs) ...", flush=True)
        model = CleanMLP(in_features, hidden_dims, out_features, task='classification')
        model.reset_parameters(seed=42)
        trainer = init_fn(model)
        tracker = ExperimentTracker(name)
        tracker.start()

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
                loss, _, flops = trainer.train_step(bx, by, criterion)
                epoch_loss += loss
                epoch_flops += flops
            tracker.record_step(epoch_loss / n_batches, epoch_flops)

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    preds = model(test_X)
                    test_loss = criterion(preds, test_y).item()
                    test_acc = compute_classification_accuracy(preds, test_y)
                print(f"     Epoch {epoch+1:2d}/{epochs}: Test Loss = {test_loss:.4f} | Test Acc = {test_acc*100:.2f}%")

        tracker.stop()
        with torch.no_grad():
            preds = model(test_X)
            test_loss = criterion(preds, test_y).item()
            test_acc = compute_classification_accuracy(preds, test_y)
        tracker.record_eval(test_loss, test_acc)
        res = tracker.summary()
        mnist_results.append(res)
        print(f"  --> {name}: Final Test Acc = {test_acc*100:.2f}% | Wall-Clock = {res['total_time_s']:.2f}s | Peak RSS = {res['peak_rss_mb']:.1f} MB")

    out_file = "/root/vnts_clone/results/exp_mnist.json"
    with open(out_file, "w") as f:
        json.dump(mnist_results, f, indent=2)
    print(f"\nMNIST results saved to {out_file}")
    return mnist_results

if __name__ == "__main__":
    run_mnist_experiment()
