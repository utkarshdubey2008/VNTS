"""
Phase 4 Unified Benchmark Suite:
1. Infinite Lifelong Learning with Spectral Compaction (SMC-DN) across 10 tasks.
2. Causal Sequence Modeling without BPTT (COMT).
3. Self-Supervised Subspace Discovery (CSD).
4. Spatial Subspace Convolution (Spatial-OSM).
"""

import os
import sys
import json
import time
import resource
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import Dict, Any, List

from src.osm.model import OSMNetwork
from src.osm.deep_osm import DeepOSMNetwork
from src.osm.compaction import SpectralMemoryCompactor
from src.osm.causal_sequence import CausalOrthogonalSequenceModel
from src.osm.unsupervised import ContrastiveSubspaceLearner
from src.osm.spatial_conv import SpatialOSMNetwork
from src.baselines.backprop_baseline import StandardMLP, BackpropTrainer
from benchmarks.datasets import load_digits_dataset


def get_peak_mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# -----------------------------------------------------------------------------
# 1. Benchmark: Infinite Lifelong Learning with Spectral Compaction (10 Tasks)
# -----------------------------------------------------------------------------
def benchmark_infinite_lifelong(n_tasks: int = 10) -> Dict[str, Any]:
    print(f"\n{'='*70}\n[BENCHMARK 4.1] 10-Task Lifelong Learning with Spectral Compaction\n{'='*70}")
    
    # Load full dataset
    X_tr, y_tr, X_te, y_te = load_digits_dataset(seed=42)
    
    # Generate 10 sequential tasks using random feature permutations (Permuted-Digits)
    torch.manual_seed(42)
    tasks = []
    d = X_tr.shape[1]
    for k in range(n_tasks):
        perm = torch.randperm(d) if k > 0 else torch.arange(d)
        X_tr_k = X_tr[:, perm]
        X_te_k = X_te[:, perm]
        tasks.append((X_tr_k, y_tr, X_te_k, y_te))
        
    print(f"Generated {n_tasks} sequential permuted tasks. Evaluating capacity saturation vs spectral compaction...")
    
    # Model with Spectral Compaction enabled:
    # in_dim=64, expansion_dim=512, out_dim=10
    osm = OSMNetwork(
        in_dim=64,
        expansion_dim=512,
        out_dim=10,
        variance_threshold=0.98,
        seed=42
    )
    
    t0 = time.time()
    initial_accs = []
    occupied_dims = []
    
    for t_idx, (x_tr_k, y_tr_k, x_te_k, y_te_k) in enumerate(tasks):
        dim = osm.learn_task(x_tr_k, y_tr_k, num_classes=10)
        acc_init = osm.evaluate(x_te_k, y_te_k)["accuracy"]
        initial_accs.append(acc_init)
        occupied_dims.append(dim)
        print(f"Task {t_idx+1:2d} Learned -> Immediate Acc: {acc_init*100:.2f}% (Basis Dimension: {dim}/512)")
        
    lifelong_time = time.time() - t0
    
    # Final evaluation on all 10 tasks to measure retention across the entire lifespan
    final_retained_accs = []
    for t_idx, (_, _, x_te_k, y_te_k) in enumerate(tasks):
        acc_retained = osm.evaluate(x_te_k, y_te_k)["accuracy"]
        final_retained_accs.append(acc_retained)
        
    avg_retained = sum(final_retained_accs) / len(final_retained_accs)
    print(f"\nFinal Retained Accuracies across 10 Tasks: {[round(a*100, 1) for a in final_retained_accs]}")
    print(f"Lifelong Average Retained Accuracy: {avg_retained*100:.2f}% (Time: {lifelong_time:.3f}s)")
    
    # Compare with Backprop on the same 10-task permuted sequence
    print("\n--- Evaluating PyTorch Backprop on 10 Sequential Tasks ---")
    bp = StandardMLP([64, 256, 128, 10], seed=42)
    bp_trainer = BackpropTrainer(bp, optimizer_name="adamw", lr=1e-3)
    t0 = time.time()
    for t_idx, (x_tr_k, y_tr_k, _, _) in enumerate(tasks):
        bp_trainer.fit(x_tr_k, y_tr_k, epochs=10, batch_size=32)
    bp_time = time.time() - t0
    
    bp_retained = [bp_trainer.evaluate(x_te_k, y_te_k)["accuracy"] for _, _, x_te_k, y_te_k in tasks]
    avg_bp_retained = sum(bp_retained) / len(bp_retained)
    print(f"Backprop Retained across 10 Tasks: {[round(a*100, 1) for a in bp_retained]}")
    print(f"Backprop Average Retained Accuracy: {avg_bp_retained*100:.2f}% (Time: {bp_time:.3f}s)")

    return {
        "OSM_Compacted": {
            "initial_accs": initial_accs,
            "final_retained_accs": final_retained_accs,
            "average_retained": avg_retained,
            "basis_dimensions": occupied_dims,
            "time_s": lifelong_time
        },
        "Backprop_AdamW": {
            "final_retained_accs": bp_retained,
            "average_retained": avg_bp_retained,
            "time_s": bp_time
        }
    }


# -----------------------------------------------------------------------------
# 2. Benchmark: Causal Sequence Modeling Without BPTT (COMT)
# -----------------------------------------------------------------------------
def benchmark_causal_sequence() -> Dict[str, Any]:
    print(f"\n{'='*70}\n[BENCHMARK 4.2] Causal Sequence Modeling Without BPTT (COMT)\n{'='*70}")
    
    # Structured sequence: 5000 tokens
    torch.manual_seed(42)
    vocab_size = 16
    tokens = [i % vocab_size for i in range(5000)]
    train_tokens = tokens[:4000]
    test_tokens = tokens[4000:]
    
    # 1. COMT (Our gradient-free, BPTT-free model)
    t0 = time.time()
    comt = CausalOrthogonalSequenceModel(
        vocab_size=vocab_size,
        embed_dim=128,
        context_decay=0.85,
        forgetting_factor=0.9995,
        seed=42
    )
    res_tr = comt.train_sequence(train_tokens)
    comt_time = time.time() - t0
    res_te = comt.evaluate_sequence(test_tokens)
    print(f"COMT (Zero BPTT): Test Next-Token Acc = {res_te['acc']*100:.2f}% | Time = {comt_time:.3f}s")
    
    # 2. PyTorch LSTM with BPTT
    import torch.nn as nn
    class LSTMBaseline(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
            self.fc = nn.Linear(embed_dim, vocab_size)
        def forward(self, x):
            return self.fc(self.lstm(self.embed(x))[0])
            
    lstm = LSTMBaseline(vocab_size, embed_dim=128)
    opt = torch.optim.AdamW(lstm.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()
    
    inp_tr = torch.tensor(train_tokens[:-1]).unsqueeze(0)
    tgt_tr = torch.tensor(train_tokens[1:]).unsqueeze(0)
    
    t0 = time.time()
    for ep in range(15):
        opt.zero_grad()
        out = lstm(inp_tr)
        loss = crit(out.view(-1, vocab_size), tgt_tr.view(-1))
        loss.backward()
        opt.step()
    lstm_time = time.time() - t0
    
    with torch.no_grad():
        inp_te = torch.tensor(test_tokens[:-1]).unsqueeze(0)
        tgt_te = torch.tensor(test_tokens[1:])
        preds_lstm = lstm(inp_te).argmax(dim=-1).squeeze(0)
        lstm_acc = (preds_lstm == tgt_te).float().mean().item()
        
    print(f"PyTorch LSTM (BPTT): Test Acc = {lstm_acc*100:.2f}% | Time = {lstm_time:.3f}s")
    print(f"COMT vs LSTM Speedup: {lstm_time / max(1e-4, comt_time):.2f}x faster")
    
    return {
        "COMT": {"test_acc": res_te["acc"], "time_s": comt_time},
        "PyTorch_LSTM": {"test_acc": lstm_acc, "time_s": lstm_time}
    }


# -----------------------------------------------------------------------------
# 3. Benchmark: Self-Supervised Subspace Discovery (CSD)
# -----------------------------------------------------------------------------
def benchmark_unsupervised_csd() -> Dict[str, Any]:
    print(f"\n{'='*70}\n[BENCHMARK 4.3] Self-Supervised Subspace Discovery (CSD - Zero Labels)\n{'='*70}")
    
    X_tr, y_tr, X_te, y_te = load_digits_dataset(seed=42)
    
    # 1. Unsupervised training (no labels provided)
    t0 = time.time()
    noise_sigma = 0.1
    view_a = X_tr + noise_sigma * torch.randn_like(X_tr)
    view_b = X_tr + noise_sigma * torch.randn_like(X_tr)
    
    csl = ContrastiveSubspaceLearner(in_dim=64, latent_dim=32, reg_lambda=1e-2, seed=42)
    csl.fit_views(view_a, view_b)
    unsupervised_time = time.time() - t0
    
    # 2. Linear probe evaluation
    from src.dats.algebraic_solver import solve_ridge_subspace
    Z_tr = csl.transform(X_tr)
    Z_te = csl.transform(X_te)
    
    Y_tr = torch.full((len(y_tr), 10), -1.0)
    Y_tr.scatter_(1, y_tr.unsqueeze(1), 1.0)
    
    Z_tr_aug = torch.cat([Z_tr, torch.ones(len(Z_tr), 1)], dim=1)
    Z_te_aug = torch.cat([Z_te, torch.ones(len(Z_te), 1)], dim=1)
    
    W_probe = solve_ridge_subspace(Z_tr_aug, Y_tr, reg_lambda=1e-3)
    preds = Z_te_aug @ W_probe
    probe_acc = (preds.argmax(1) == y_te).float().mean().item()
    
    print(f"CSD Unsupervised Fitting Time: {unsupervised_time:.4f}s")
    print(f"Linear Readout Probe Accuracy:  {probe_acc*100:.2f}% (Without a single training label)")
    
    return {
        "unsupervised_time_s": unsupervised_time,
        "linear_probe_acc": probe_acc
    }


# -----------------------------------------------------------------------------
# 4. Benchmark: Spatial Subspace Convolution (Spatial-OSM)
# -----------------------------------------------------------------------------
def benchmark_spatial_conv() -> Dict[str, Any]:
    print(f"\n{'='*70}\n[BENCHMARK 4.4] 2D Spatial Subspace Convolution (Spatial-OSM)\n{'='*70}")
    
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_tr = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    mnist_te = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    
    X_tr = mnist_tr.data[:2000].float().unsqueeze(1) / 255.0
    y_tr = mnist_tr.targets[:2000].long()
    
    X_te = mnist_te.data[:1000].float().unsqueeze(1) / 255.0
    y_te = mnist_te.targets[:1000].long()
    
    t0 = time.time()
    spatial_osm = SpatialOSMNetwork(
        in_channels=1,
        image_size=28,
        patch_size=7,
        patch_embed_dim=256,
        out_dim=10,
        variance_threshold=0.98,
        seed=42
    )
    spatial_osm.learn_task(X_tr, y_tr, num_classes=10)
    spatial_time = time.time() - t0
    
    eval_res = spatial_osm.evaluate(X_te, y_te)
    print(f"Spatial-OSM Test Accuracy: {eval_res['accuracy']*100:.2f}% | Time: {spatial_time:.3f}s")
    
    return {
        "spatial_acc": eval_res["accuracy"],
        "spatial_time_s": spatial_time
    }


def run_phase4_suite():
    os.makedirs("results", exist_ok=True)
    results = {}
    
    results["lifelong_compaction"] = benchmark_infinite_lifelong(n_tasks=10)
    results["causal_sequence"] = benchmark_causal_sequence()
    results["unsupervised_csd"] = benchmark_unsupervised_csd()
    results["spatial_conv"] = benchmark_spatial_conv()
    
    out_file = "results/phase4_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}\n[SUCCESS] Phase 4 Benchmark Suite completed! Results saved to {out_file}\n{'='*70}")


if __name__ == "__main__":
    run_phase4_suite()
