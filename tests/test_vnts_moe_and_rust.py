"""
Unit test suite for Orthogonal Subspace MoE and Rust C-FFI backend.
"""

import numpy as np
import torch
import pytest

from vnts.nn import OrthogonalSubspaceMoE
from vnts.models import VNTSMoELanguageModel
from vnts.native import RustNativeEngine


def test_orthogonal_subspace_moe_routing():
    """Verify OSR-MoE routes tokens based on subspace energy without collapse."""
    B, T, D = 4, 16, 64
    x = torch.randn(B, T, D)
    moe = OrthogonalSubspaceMoE(
        in_features=D,
        out_features=128,
        num_experts=4,
        top_k=2,
        subspace_dim=16
    )

    out = moe(x)
    assert out.shape == (B, T, 128)
    assert not out.requires_grad

    # Test token routing weights
    weights, indices = moe.route_tokens(x.reshape(-1, D))
    assert weights.shape == (B * T, 2)
    assert indices.shape == (B * T, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B * T), atol=1e-4)


def test_vnts_moe_lm_fit_and_val():
    """Verify VNTSMoELanguageModel trains on corpus and computes valid validation perplexity."""
    torch.manual_seed(42)
    train_corpus = "The Vector Nullspace Transformation System establishes deep non-gradient learning paradigms. " * 8
    val_corpus = "Deep non-gradient learning paradigms establish robust artificial intelligence systems. " * 4

    lm = VNTSMoELanguageModel.from_corpus(
        train_corpus + val_corpus,
        embed_dim=64,
        num_heads=4,
        num_experts=4,
        top_k=2,
        expert_dim=128
    )

    results = lm.fit_and_evaluate(train_corpus, val_corpus, seq_len=16)
    assert results["tokens_trained"] > 0
    assert results["training_time_s"] < 1.0
    assert results["train_accuracy"] > 0.40
    assert results["val_perplexity"] > 1.0  # Perplexity is positive real number > 1
    assert results["val_loss"] > 0.0

    prompt = "The Vector"
    gen = lm.generate(prompt=prompt, max_tokens=30, temperature=0.7)
    assert len(gen) > len(prompt)
    assert gen.startswith(prompt)


def test_rust_native_c_ffi():
    """Verify direct C-FFI execution of native Rust engine (libvnts_core.so)."""
    assert RustNativeEngine.is_available(), "libvnts_core.so was not loaded"

    np.random.seed(42)
    N, D, K = 200, 32, 4
    X = np.random.randn(N, D).astype(np.float32)
    W_true = np.random.randn(D, K).astype(np.float32)
    Y = X @ W_true + 0.01 * np.random.randn(N, K).astype(np.float32)

    # Solve Ridge in native Rust via C-FFI
    W_est = RustNativeEngine.solve_ridge(X, Y, reg_lambda=1e-3)
    err = np.linalg.norm(W_est - W_true) / np.linalg.norm(W_true)
    assert err < 0.05, f"Rust native solve error too high: {err}"

    # Solve Nullspace in native Rust via C-FFI
    # Generate arbitrary prior basis U (D x r) where U^T U = I
    U_rand = np.random.randn(D, 8).astype(np.float32)
    U, _ = np.linalg.qr(U_rand)

    dW_null = RustNativeEngine.solve_nullspace(X, Y, U, reg_lambda=1e-3)
    # Check that U^T @ dW_null is 0
    drift = np.linalg.norm(U.T @ dW_null)
    assert drift < 1e-4, f"Rust nullspace projection drift too high: {drift}"
