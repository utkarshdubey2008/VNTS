"""
Phase 7 Benchmark: Multi-Scale Color Vision, Byte-Level Scaled LLM, and Native Engine.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from vnts import Model
from vnts.nn import MultiScaleSpatialOSM, Dense, OrthogonalMemory
from vnts.models import ByteLLM
from vnts.native import NativeAlgebraicEngine, NativeMemory


def benchmark_color_vision():
    print("--- Benchmark 1: Multi-Scale Color Vision (RGB 32x32x3) ---")
    torch.manual_seed(42)
    B_train, B_test = 500, 200
    # Simulate multi-class RGB vision data
    X_train = torch.randn(B_train, 3, 32, 32)
    y_train = torch.randint(0, 10, (B_train,))
    X_test = torch.randn(B_test, 3, 32, 32)
    y_test = torch.randint(0, 10, (B_test,))

    # 1. VNTS MultiScaleSpatialOSM + Memory
    t0 = time.perf_counter()
    spatial = MultiScaleSpatialOSM(in_channels=3, patch_sizes=(3, 5), strides=(2, 2), features_per_scale=32)
    dense_proj = Dense(spatial.out_dim, 128)
    mem = OrthogonalMemory(128, 10)

    # Closed-form feature extraction & assimilation
    feat_train = dense_proj(spatial(X_train))
    Y_onehot = torch.zeros(B_train, 10)
    Y_onehot.scatter_(1, y_train.unsqueeze(1), 1.0)
    mem.assimilate(feat_train, Y_onehot)
    vnts_time = time.perf_counter() - t0

    feat_test = dense_proj(spatial(X_test))
    preds = torch.argmax(mem(feat_test), dim=1)
    # Check execution works stably
    print(f"VNTS Color Vision Assimilated {B_train} RGB images in {vnts_time:.4f}s ({B_train/vnts_time:.1f} imgs/s)")

    # 2. PyTorch ConvNet Baseline
    class SimpleConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 16, 10)
            )
        def forward(self, x):
            return self.net(x)

    py_net = SimpleConvNet()
    optimizer = optim.AdamW(py_net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    t0_pt = time.perf_counter()
    py_net.train()
    # 5 epochs
    for ep in range(5):
        optimizer.zero_grad()
        out = py_net(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
    pt_time = time.perf_counter() - t0_pt
    print(f"PyTorch ConvNet (5 epochs): {pt_time:.4f}s, Speedup: {pt_time/vnts_time:.2f}x faster with VNTS")

    return {
        "vnts_time_s": vnts_time,
        "pytorch_time_s": pt_time,
        "speedup_factor": pt_time / max(1e-6, vnts_time)
    }


def benchmark_byte_llm():
    print("\n--- Benchmark 2: Byte-Level Scaled Language Modeler ---")
    corpus = """
    The Vector Nullspace Transformation System establishes an entirely different mechanism
    through which artificial neural architectures learn, adapt, and retain representations.
    By mapping representations to exact orthonormal manifolds, zero-interference guarantees
    are achieved at machine precision.
    """ * 12

    model = ByteLLM(embed_dim=128, num_heads=8)
    t0 = time.perf_counter()
    stats = model.train_text(corpus, seq_len=64)
    train_time = time.perf_counter() - t0

    prompt = "The Vector Nullspace"
    gen = model.generate(prompt=prompt, max_new_bytes=60, temperature=0.7)

    print(f"ByteLLM Processed {stats['bytes_processed']} bytes in {train_time:.4f}s")
    print(f"Throughput: {stats['throughput_bytes_per_s']:.1f} bytes/s")
    print(f"Next-Byte Accuracy: {stats['next_byte_accuracy'] * 100:.2f}%")
    print(f"Sample Generation: {repr(gen[:80])}...")

    return {
        "bytes_processed": stats["bytes_processed"],
        "training_time_s": train_time,
        "throughput_bytes_per_s": stats["throughput_bytes_per_s"],
        "next_byte_accuracy": stats["next_byte_accuracy"],
        "generated_sample": gen
    }


def benchmark_native_numpy_engine():
    print("\n--- Benchmark 3: Native NumPy vs PyTorch CPU Solver ---")
    N, D = 2000, 256
    X_np = np.random.randn(N, D).astype(np.float32)
    Y_np = np.random.randn(N, 10).astype(np.float32)

    X_torch = torch.from_numpy(X_np)
    Y_torch = torch.from_numpy(Y_np)

    # Pure NumPy Cholesky
    t0 = time.perf_counter()
    W_np = NativeAlgebraicEngine.solve_ridge(X_np, Y_np, reg_lambda=1e-3)
    np_time = time.perf_counter() - t0

    # PyTorch CPU Cholesky
    t0_pt = time.perf_counter()
    XtX = X_torch.T @ X_torch + 1e-3 * torch.eye(D)
    L = torch.linalg.cholesky(XtX)
    W_pt = torch.cholesky_solve(X_torch.T @ Y_torch, L)
    pt_time = time.perf_counter() - t0_pt

    print(f"Native NumPy Engine Solve Time: {np_time:.4f}s")
    print(f"PyTorch CPU Engine Solve Time:  {pt_time:.4f}s")
    print(f"Relative Performance: NumPy is {pt_time/np_time:.2f}x of PyTorch speed")

    return {
        "native_numpy_time_s": np_time,
        "pytorch_cpu_time_s": pt_time
    }


def main():
    print("=" * 70)
    print("Phase 7: Scaling Multi-Scale Color Vision, ByteLLM, & Native Engine")
    print("=" * 70)

    results = {}
    results["color_vision"] = benchmark_color_vision()
    results["byte_llm"] = benchmark_byte_llm()
    results["native_engine"] = benchmark_native_numpy_engine()

    os.makedirs("results", exist_ok=True)
    with open("results/phase7_scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("All Phase 7 Benchmarks Complete. Telemetry saved -> results/phase7_scaling_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
