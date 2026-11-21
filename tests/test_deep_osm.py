"""
Unit tests for Deep-OSM (Deep Hierarchical Orthogonal Subspace Memory).
"""

import pytest
import torch
from src.osm.deep_osm import DeepOSMLayer, DeepOSMNetwork


def test_deep_osm_zero_autograd():
    """Verify DeepOSMNetwork contains zero gradients or autograd dependencies."""
    model = DeepOSMNetwork(in_dim=20, hidden_dims=[64, 64, 32], out_dim=3, seed=42)
    for i, layer in enumerate(model.layers):
        assert not layer.W_proj.requires_grad, f"Layer {i} W_proj has requires_grad=True!"
        assert not layer.b_proj.requires_grad, f"Layer {i} b_proj has requires_grad=True!"
        assert not layer.memory.W.requires_grad, f"Layer {i} memory.W has requires_grad=True!"
        assert layer.memory.W.grad is None

    x = torch.randn(15, 20)
    out = model.forward(x)
    assert not out.requires_grad
    assert out.grad_fn is None


def test_deep_osm_hierarchical_residual_routing():
    """Verify that multi-layer residual routing successfully reduces residual error across layers."""
    torch.manual_seed(42)
    model = DeepOSMNetwork(in_dim=10, hidden_dims=[32, 32], out_dim=2, seed=42)
    x = torch.randn(50, 10)
    y = torch.randint(0, 2, (50,))
    
    # Train
    model.learn_task(x, y, num_classes=2)
    eval_res = model.evaluate(x, y)
    assert eval_res["accuracy"] >= 0.90, f"Deep-OSM failed to fit training distribution: {eval_res['accuracy']}"


def test_deep_osm_continual_task_preservation():
    """Verify that learning a second task does not erase knowledge of the first task."""
    torch.manual_seed(42)
    model = DeepOSMNetwork(in_dim=16, hidden_dims=[256, 256], out_dim=4, variance_threshold=1.0, seed=42)
    
    # Task 1
    x1 = torch.randn(100, 16) + 1.0
    y1 = torch.randint(0, 2, (100,))
    model.learn_task(x1, y1, num_classes=4)
    acc1_init = model.evaluate(x1, y1)["accuracy"]
    assert acc1_init >= 0.85
    
    # Task 2
    x2 = torch.randn(100, 16) - 1.0
    y2 = torch.randint(2, 4, (100,))
    model.learn_task(x2, y2, num_classes=4)
    acc2_init = model.evaluate(x2, y2)["accuracy"]
    assert acc2_init >= 0.85
    
    # Evaluate Task 1 retention
    acc1_retained = model.evaluate(x1, y1)["accuracy"]
    assert acc1_retained >= 0.70, f"Deep-OSM suffered excessive forgetting: {acc1_retained}"
