"""
Unit tests for Phase 7: Scaling Multi-Scale Color Vision, ByteLLM, and Native Standalone Engine.
"""

import numpy as np
import torch
import pytest

from vnts.nn import MultiScaleSpatialOSM
from vnts.models import ByteLLM
from vnts.native import NativeAlgebraicEngine, NativeMemory


def test_multiscale_spatial_color():
    """Verify MultiScaleSpatialOSM extracts multi-scale visual features from RGB images."""
    B, C, H, W = 16, 3, 32, 32
    images = torch.randn(B, C, H, W)

    model = MultiScaleSpatialOSM(
        in_channels=3,
        patch_sizes=(3, 5),
        strides=(2, 2),
        features_per_scale=32
    )
    features = model(images)
    # 2 scales * (32 * 3) = 192 features
    expected_dim = 2 * (32 * 3)
    assert features.shape == (B, expected_dim)
    assert not torch.isnan(features).any()


def test_byte_llm_encoding_and_generation():
    """Verify universal ByteLLM causal next-byte prediction and generation."""
    torch.manual_seed(42)
    corpus = "Machine learning without backpropagation is fast and scalable! " * 8
    model = ByteLLM(embed_dim=64, num_heads=4)

    stats = model.train_text(corpus, seq_len=32)
    assert stats["bytes_processed"] > 0
    assert stats["training_time_s"] < 0.5
    assert stats["next_byte_accuracy"] > 0.35

    prompt = "Machine learning"
    gen = model.generate(prompt=prompt, max_new_bytes=40, temperature=0.7)
    assert len(gen) > len(prompt)
    assert gen.startswith(prompt)


def test_native_engine_pure_numpy():
    """Verify NativeAlgebraicEngine runs on pure NumPy with zero PyTorch dependency."""
    np.random.seed(42)
    N, D = 100, 10
    X = np.random.randn(N, D).astype(np.float32)
    W_true = np.random.randn(D, 2).astype(np.float32)
    Y = X @ W_true + 0.01 * np.random.randn(N, 2).astype(np.float32)

    # Primal solve
    W_est = NativeAlgebraicEngine.solve_ridge(X, Y, reg_lambda=1e-4)
    err = np.linalg.norm(W_est - W_true) / np.linalg.norm(W_true)
    assert err < 0.05, f"Native primal solve error: {err:.4f}"

    # Dual solve N < D
    X_wide = np.random.randn(20, 100).astype(np.float32)
    Y_wide = np.random.randn(20, 2).astype(np.float32)
    W_wide = NativeAlgebraicEngine.solve_ridge(X_wide, Y_wide, reg_lambda=1e-4)
    assert W_wide.shape == (100, 2)


def test_native_memory_continual_invariance():
    """Verify NativeMemory eliminates catastrophic forgetting on pure NumPy."""
    np.random.seed(42)
    d = 64
    mem = NativeMemory(in_features=d, out_features=2)

    # Task 1
    X1 = np.random.randn(40, d).astype(np.float32)
    Y1 = np.random.randn(40, 2).astype(np.float32)
    mem.assimilate(X1, Y1)
    p1_initial = mem.forward(X1)

    # Task 2
    X2 = np.random.randn(40, d).astype(np.float32)
    Y2 = np.random.randn(40, 2).astype(np.float32)
    mem.assimilate(X2, Y2)
    p1_after = mem.forward(X1)

    drift = np.linalg.norm(p1_after - p1_initial) / np.linalg.norm(p1_initial)
    assert drift < 1e-4, f"Native continual representation drift too high: {drift:.4e}"
