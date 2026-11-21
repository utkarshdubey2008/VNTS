"""
Unit tests for Dual-Stream Algebraic Target Synthesis & Subspace Inversion (DATS-SI).
"""

import pytest
import torch
import numpy as np

from src.dats.activations import InvertibleLeakyReLU, InvertibleLinear, InvertibleTanh
from src.dats.algebraic_solver import regularized_pseudoinverse, solve_ridge_subspace
from src.dats.model import DATSNetwork
from src.dats.trainer import DATSTrainer, encode_classification_targets


def test_activation_invertibility():
    """Verify that InvertibleLeakyReLU satisfies exact bijective reconstruction."""
    act = InvertibleLeakyReLU(negative_slope=0.1)
    z = torch.randn(100, 50)
    h = act.forward(z)
    z_rec = act.inverse(h)
    assert torch.allclose(z, z_rec, atol=1e-6), "LeakyReLU inverse does not match original pre-activations."


def test_linear_invertibility():
    act = InvertibleLinear()
    z = torch.randn(50, 20)
    assert torch.equal(act.forward(z), z)
    assert torch.equal(act.inverse(z), z)


def test_pseudoinverse_properties():
    """Test regularized pseudoinverse satisfies Moore-Penrose approximation."""
    torch.manual_seed(42)
    # Overdetermined
    W1 = torch.randn(64, 16)
    W1_pinv = regularized_pseudoinverse(W1, gamma=1e-6)
    # W1_pinv @ W1 should be close to Identity(16)
    prod1 = W1_pinv @ W1
    assert torch.allclose(prod1, torch.eye(16), atol=1e-3)
    
    # Underdetermined
    W2 = torch.randn(16, 64)
    W2_pinv = regularized_pseudoinverse(W2, gamma=1e-6)
    # W2 @ W2_pinv should be close to Identity(16)
    prod2 = W2 @ W2_pinv
    assert torch.allclose(prod2, torch.eye(16), atol=1e-3)


def test_no_gradients_in_model():
    """Verify strictly zero autograd gradients in DATSNetwork."""
    model = DATSNetwork([10, 32, 16, 2], seed=42)
    for W in model.weights:
        assert not W.requires_grad
        assert W.grad is None
    for b in model.biases:
        assert not b.requires_grad
        assert b.grad is None

    x = torch.randn(20, 10)
    out = model.forward(x)
    assert not out.requires_grad
    assert out.grad_fn is None


def test_ridge_subspace_solver_minimizes_error():
    """Verify that solve_ridge_subspace finds the optimal Tikhonov projector."""
    torch.manual_seed(42)
    N, d_in, d_out = 100, 8, 4
    H = torch.randn(N, d_in)
    ones = torch.ones(N, 1)
    H_aug = torch.cat([H, ones], dim=1)
    
    # Ground truth affine mapping with some noise
    W_true = torch.randn(d_in + 1, d_out)
    Z_target = H_aug @ W_true + 0.01 * torch.randn(N, d_out)
    
    W_opt = solve_ridge_subspace(H_aug, Z_target, reg_lambda=1e-5)
    
    # Check that error is minimal
    residual_opt = torch.norm(H_aug @ W_opt - Z_target)
    residual_random = torch.norm(H_aug @ torch.randn_like(W_opt) - Z_target)
    assert residual_opt < residual_random
    assert torch.allclose(W_opt, W_true, atol=0.1)


def test_dats_training_step_reduces_loss():
    """Verify that DATS training epoch reduces error on synthetic non-linear data."""
    torch.manual_seed(42)
    x = torch.randn(100, 4)
    y = torch.randint(0, 2, (100,))
    
    model = DATSNetwork([4, 16, 8, 2], seed=42)
    trainer = DATSTrainer(model, beta=0.5, gamma=1e-4, reg_lambda=1e-4)
    
    loss_before = trainer.evaluate(x, y)["loss"]
    history = trainer.fit(x, y, epochs=5, verbose=False)
    loss_after = history["train_loss"][-1]
    
    assert loss_after < loss_before, f"DATS training failed to reduce loss: {loss_before} -> {loss_after}"
