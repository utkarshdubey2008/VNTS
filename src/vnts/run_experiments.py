"""
VNTS: Master Experiment Runner
Runs controlled comparative experiments across all baselines and candidates.
Saves detailed JSON summaries.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn

try:
    from .benchmarks import (
        generate_multimanifold_dataset,
        generate_nonlinear_regression_dataset,
        generate_structured_patterns
    )
    from .models import CleanMLP
    from .metrics import (
        ExperimentTracker,
        compute_classification_accuracy,
        compute_r2_score
    )
    from .trainers import (
        BackpropTrainer,
        DFATrainer,
        FRPTrainer,
        SPSABaselineTrainer,
        LSFGTrainer,
        CLATrainer
    )
except ImportError:
    from benchmarks import (
        generate_multimanifold_dataset,
        generate_nonlinear_regression_dataset,
        generate_structured_patterns
    )
    from models import CleanMLP
    from metrics import (
        ExperimentTracker,
        compute_classification_accuracy,
        compute_r2_score
    )
    from trainers import (
        BackpropTrainer,
        DFATrainer,
        FRPTrainer,
        SPSABaselineTrainer,
        LSFGTrainer,
        CLATrainer
    )

def run_single_benchmark(benchmark_name, get_data_fn, task_type, in_features, out_features, epochs=30, batch_size=128):
    print(f"\n========================================================")
    print(f"  RUNNING BENCHMARK: {benchmark_name.upper()} ({task_type})")
    print(f"========================================================")
    
    (train_X, train_y), (test_X, test_y) = get_data_fn()
    n_train = len(train_X)
    hidden_dims = [128, 128, 64]
    
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
        eval_fn = compute_classification_accuracy
    else:
        criterion = nn.MSELoss()
        eval_fn = compute_r2_score

    algorithms = [
        ("Backprop-AdamW", lambda m: BackpropTrainer(m, optimizer_name='adamw', lr=2e-3)),
        ("Backprop-SGD", lambda m: BackpropTrainer(m, optimizer_name='sgd', lr=0.05)),
        ("DFA (Direct Feedback Alignment)", lambda m: DFATrainer(m, lr=2e-3)),
        ("Candidate 1 (FRP: Forward Residual)", lambda m: FRPTrainer(m, lr=2e-3, alpha_rep=0.01)),
        ("SPSA (Isotropic Forward-Grad)", lambda m: SPSABaselineTrainer(m, lr=1e-3, epsilon=1e-3)),
        ("Candidate 2 (L-SFG: Structured Forward-Grad)", lambda m: LSFGTrainer(m, lr=1e-3, epsilon=1e-3)),
        ("Candidate 3 (CLA: Closed-Form Latent)", lambda m: CLATrainer(m, lr=0.1, ridge_lambda=1e-3))
    ]

    benchmark_results = []

    for name, trainer_init in algorithms:
        print(f"--> Testing [{name}] ...", flush=True)
        model = CleanMLP(in_features, hidden_dims, out_features, task=task_type)
        model.reset_parameters(seed=42)
        trainer = trainer_init(model)
        tracker = ExperimentTracker(name)
        tracker.start()

        n_batches = (n_train + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            perm = torch.randperm(n_train)
            train_X_shuff = train_X[perm]
            train_y_shuff = train_y[perm]

            epoch_loss = 0.0
            epoch_flops = 0
            for b in range(n_batches):
                bx = train_X_shuff[b*batch_size : (b+1)*batch_size]
                by = train_y_shuff[b*batch_size : (b+1)*batch_size]
                
                loss, _, flops = trainer.train_step(bx, by, criterion)
                epoch_loss += loss
                epoch_flops += flops

            avg_train_loss = epoch_loss / n_batches
            tracker.record_step(avg_train_loss, epoch_flops)

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    test_preds = model(test_X)
                    test_loss = criterion(test_preds, test_y).item()
                    test_metric = eval_fn(test_preds, test_y)
                tracker.record_eval(test_loss, test_metric)

        tracker.stop()
        res = tracker.summary()
        benchmark_results.append(res)
        print(f"    Done in {res['total_time_s']}s | Peak RSS: {res['peak_rss_mb']} MB (Net: {res['net_memory_mb']} MB) | "
              f"Test Loss: {res['final_test_loss']:.4f} | Test Score: {res['final_test_metric']:.4f}")

    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="VNTS Benchmark Suite")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save experimental results")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per benchmark")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    if args.output_dir is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(curr_dir, "../../results"))
    else:
        output_dir = os.path.abspath(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)
    all_experiments = {}

    # Benchmark 1: Nonlinear Multi-Manifold Spiral Classification (4 classes, 4 inputs)
    b1_results = run_single_benchmark(
        benchmark_name="b1_multimanifold_classification",
        get_data_fn=lambda: generate_multimanifold_dataset(n_samples=4000, n_classes=4, seed=42),
        task_type="classification",
        in_features=4,
        out_features=4,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    all_experiments["benchmark_1_multimanifold"] = b1_results

    # Benchmark 2: Hierarchical Non-Linear Regression (16 features, 1 output)
    b2_results = run_single_benchmark(
        benchmark_name="b2_hierarchical_regression",
        get_data_fn=lambda: generate_nonlinear_regression_dataset(n_samples=4000, n_features=16, seed=42),
        task_type="regression",
        in_features=16,
        out_features=1,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    all_experiments["benchmark_2_regression"] = b2_results

    # Benchmark 3: High-Dimensional Structured Pattern Discrimination (64 features, 5 classes)
    b3_results = run_single_benchmark(
        benchmark_name="b3_highdim_patterns",
        get_data_fn=lambda: generate_structured_patterns(n_samples=4000, n_features=64, n_classes=5, seed=42),
        task_type="classification",
        in_features=64,
        out_features=5,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    all_experiments["benchmark_3_highdim_patterns"] = b3_results

    out_file = os.path.join(output_dir, "experimental_results.json")
    with open(out_file, "w") as f:
        json.dump(all_experiments, f, indent=2)

    print(f"\nAll benchmarks completed successfully! Results written to {out_file}")

if __name__ == "__main__":
    main()
