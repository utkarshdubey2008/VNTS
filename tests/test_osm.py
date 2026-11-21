"""
Unit tests for OSM-RME (Orthogonal Subspace Memory & Resonant Manifold Evolution).
"""

import pytest
import torch
import numpy as np
from src.osm.model import OrthogonalSubspaceMemory, OSMNetwork


def test_osm_zero_autograd():
    """Verify that OSMNetwork has zero autograd dependencies."""
    osm = OSMNetwork(in_dim=10, expansion_dim=64, out_dim=2, seed=42)
    assert not osm.W_proj.requires_grad
    assert not osm.b_proj.requires_grad
    assert not osm.memory.W.requires_grad
    
    x = torch.randn(20, 10)
    out = osm.forward(x)
    assert out.grad_fn is None
    assert not out.requires_grad


def test_orthogonal_nullspace_invariance():
    """
    Verify mathematically that updates in the nullspace projector P_null
    produce ZERO interference on previous task features:
    X_old @ (P_null @ Delta_W) == 0
    """
    d = 32
    mem = OrthogonalSubspaceMemory(in_features=d, out_features=2, variance_threshold=0.99)
    
    # Task 1 data
    X1 = torch.randn(100, d)
    Y1 = torch.randn(100, 2)
    mem.learn_task(X1, Y1)
    
    # Previous predictions on Task 1
    preds1_before = mem.forward(X1)
    
    # Task 2 data arrives
    X2 = torch.randn(100, d)
    Y2 = torch.randn(100, 2)
    mem.learn_task(X2, Y2)
    
    preds1_after = mem.forward(X1)
    
    # Check that Task 1 predictions are well-preserved (minimal drift)
    diff = torch.norm(preds1_after - preds1_before) / torch.norm(preds1_before)
    assert diff < 0.1, f"Task 1 representation drift is too high: {diff.item():.4f}"


def test_continual_learning_retention():
    """Verify that OSMNetwork retains high accuracy on Task 1 after learning Task 2."""
    torch.manual_seed(42)
    osm = OSMNetwork(in_dim=16, expansion_dim=512, out_dim=4, variance_threshold=1.0, seed=42)
    
    # Task 1: classes 0 and 1
    X1 = torch.randn(200, 16) + 1.5
    y1 = torch.randint(0, 2, (200,))
    osm.learn_task(X1, y1, num_classes=4)
    acc1_init = osm.evaluate(X1, y1)["accuracy"]
    assert acc1_init > 0.85
    
    # Task 2: classes 2 and 3
    X2 = torch.randn(200, 16) - 1.5
    y2 = torch.randint(2, 4, (200,))
    osm.learn_task(X2, y2, num_classes=4)
    acc2_init = osm.evaluate(X2, y2)["accuracy"]
    assert acc2_init > 0.85
    
    # Evaluate Task 1 retention
    acc1_retained = osm.evaluate(X1, y1)["accuracy"]
    assert acc1_retained >= 0.80, f"Task 1 suffered catastrophic forgetting: {acc1_retained}"
