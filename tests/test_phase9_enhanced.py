"""
Unit tests for Phase 9+ Enhanced VNTS architectures:
  - PCAProjectedORMA
  - EnsembleORMA
  - IncrementalORMA
All tests use tiny synthetic data, 1 BLAS thread, complete in seconds.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import pytest
import torch.nn.functional as F
from vnts.nn.enhanced import PCAProjectedORMA, EnsembleORMA, IncrementalORMA


torch.set_num_threads(1)
torch.manual_seed(42)


def make_cls_data(n=200, d=64, num_classes=4):
    """Tiny synthetic classification dataset."""
    X = torch.randn(n, d)
    # Build structured labels (not random) so accuracy > chance is achievable
    W = torch.randn(d, num_classes)
    y = torch.argmax(X @ W, dim=1)
    return X, y, num_classes


# ─────────────────────────────────────────────────────────────────────────────
class TestPCAProjectedORMA:
    def test_fit_and_forward_shape(self):
        X, y, nc = make_cls_data(n=120, d=32, num_classes=4)
        model = PCAProjectedORMA(in_features=32, pca_dim=16, hidden_dim=32, out_features=4, depth=3)
        model.fit(X[:100], F.one_hot(y[:100], nc).float())
        logits = model(X[100:])
        assert logits.shape == (20, 4), f"Bad shape: {logits.shape}"

    def test_pca_compresses_input(self):
        X, y, nc = make_cls_data(n=100, d=64, num_classes=2)
        model = PCAProjectedORMA(in_features=64, pca_dim=8, hidden_dim=32, out_features=2, depth=2)
        model.fit(X, F.one_hot(y, nc).float())
        # Internal V should be (64, 8)
        assert model._V is not None
        assert model._V.shape == (64, 8), f"V shape wrong: {model._V.shape}"

    def test_accuracy_beats_chance(self):
        """On structured data, PCA-ORMA should beat random (25% for 4-class)."""
        torch.manual_seed(0)
        X, y, nc = make_cls_data(n=300, d=64, num_classes=4)
        model = PCAProjectedORMA(in_features=64, pca_dim=32, hidden_dim=64, out_features=4, depth=4)
        model.fit(X[:250], F.one_hot(y[:250], nc).float())
        with torch.no_grad():
            preds = torch.argmax(model(X[250:]), dim=1)
            acc = (preds == y[250:]).float().mean().item()
        assert acc > 0.3, f"Accuracy below chance: {acc:.3f}"  # > 25% random

    def test_no_requires_grad(self):
        """PCAProjectedORMA must never use autograd."""
        X, y, nc = make_cls_data(n=80, d=16, num_classes=2)
        model = PCAProjectedORMA(in_features=16, pca_dim=8, hidden_dim=16, out_features=2, depth=2)
        model.fit(X, F.one_hot(y, nc).float())
        for name, p in model._parameters.items():
            if p is not None:
                assert not p.requires_grad, f"{name} has requires_grad=True"


# ─────────────────────────────────────────────────────────────────────────────
class TestEnsembleORMA:
    def test_fit_forward_shape(self):
        X, y, nc = make_cls_data(n=150, d=32, num_classes=3)
        ens = EnsembleORMA(in_features=32, pca_dim=16, hidden_dim=32,
                           out_features=3, n_estimators=3, depth=2)
        ens.fit(X[:120], F.one_hot(y[:120], nc).float())
        logits = ens(X[120:])
        assert logits.shape == (30, 3)

    def test_ensemble_beats_single(self):
        """Ensemble of 5 should not be worse than a single model (soft test)."""
        torch.manual_seed(1)
        X, y, nc = make_cls_data(n=400, d=32, num_classes=4)
        Xtr, ytr = X[:300], y[:300]
        Xte, yte = X[300:], y[300:]

        single = PCAProjectedORMA(32, 16, 64, 4, depth=4)
        single.fit(Xtr, F.one_hot(ytr, nc).float())
        with torch.no_grad():
            acc_single = (torch.argmax(single(Xte), 1) == yte).float().mean().item()

        ens = EnsembleORMA(32, 16, 64, 4, n_estimators=5, depth=4)
        ens.fit(Xtr, F.one_hot(ytr, nc).float())
        with torch.no_grad():
            acc_ens = (torch.argmax(ens(Xte), 1) == yte).float().mean().item()

        # Ensemble accuracy >= single - 5% (usually better, sometimes same)
        assert acc_ens >= acc_single - 0.05, f"Ensemble ({acc_ens:.3f}) much worse than single ({acc_single:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
class TestIncrementalORMA:
    def test_accumulate_and_solve(self):
        X, y, nc = make_cls_data(n=200, d=16, num_classes=3)
        model = IncrementalORMA(in_features=16, out_features=3, reg_lambda=1e-3)

        # Stream in 4 mini-batches
        bs = 50
        for i in range(0, 200, bs):
            model.accumulate(X[i:i+bs], F.one_hot(y[i:i+bs], nc).float())

        W = model.solve()
        assert W.shape == (16, 3)
        logits = model(X)
        assert logits.shape == (200, 3)

    def test_incremental_vs_batch_match(self):
        """IncrementalORMA(full stream) should match BatchRidge on same data."""
        from vnts.engine.solver import AlgebraicEngine
        torch.manual_seed(5)
        X = torch.randn(300, 20)
        Y = torch.randn(300, 4)

        # Batch ridge
        W_batch = AlgebraicEngine.solve_ridge(X, Y, reg_lambda=1e-3)

        # Incremental
        inc = IncrementalORMA(in_features=20, out_features=4, reg_lambda=1e-3)
        for i in range(0, 300, 60):
            inc.accumulate(X[i:i+60], Y[i:i+60])
        W_inc = inc.solve()

        # Should match to numerical precision
        max_diff = (W_batch - W_inc).abs().max().item()
        assert max_diff < 1e-4, f"Incremental vs batch mismatch: {max_diff}"

    def test_no_requires_grad(self):
        model = IncrementalORMA(in_features=8, out_features=2)
        X = torch.randn(40, 8)
        Y = torch.randn(40, 2)
        model.accumulate(X, Y)
        model.solve()
        for name, p in model._parameters.items():
            if p is not None:
                assert not p.requires_grad
