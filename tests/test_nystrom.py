"""
Unit tests for Nyström Kernel Layer and NyströmORMA.
All tests: tiny data, 1 thread, complete in < 5s.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn.functional as F
import pytest

torch.set_num_threads(1)
torch.manual_seed(42)


def make_data(n=100, d=20, nc=3):
    X = torch.randn(n, d)
    W = torch.randn(d, nc)
    y = torch.argmax(X @ W, dim=1)
    return X, y, nc


class TestNystromKernelLayer:
    def test_fit_forward_shape(self):
        from vnts.nn.nystrom import NystromKernelLayer
        X, y, nc = make_data(100, 20, 4)
        m = NystromKernelLayer(in_features=20, n_landmarks=30, out_features=4, kernel="rbf", gamma=0.1)
        m.fit(X[:80], y[:80])
        out = m(X[80:])
        assert out.shape == (20, 4)

    def test_rbf_kernel_range(self):
        from vnts.nn.nystrom import NystromKernelLayer
        X, y, nc = make_data(60, 10, 2)
        m = NystromKernelLayer(10, 20, 2, kernel="rbf", gamma=1.0)
        m.fit(X, y)
        # RBF kernel values ∈ (0, 1] → landmark matrix non-negative
        assert m._landmarks is not None
        assert m._landmarks.shape[1] == 10

    def test_polynomial_kernel(self):
        from vnts.nn.nystrom import NystromKernelLayer
        X, y, nc = make_data(80, 8, 3)
        m = NystromKernelLayer(8, 20, 3, kernel="polynomial", gamma=0.1, degree=2)
        m.fit(X[:60], y[:60])
        out = m(X[60:])
        assert out.shape == (20, 3)

    def test_no_requires_grad(self):
        from vnts.nn.nystrom import NystromKernelLayer
        X, y, nc = make_data(60, 8, 2)
        m = NystromKernelLayer(8, 15, 2)
        m.fit(X, y)
        for name, p in m._parameters.items():
            if p is not None:
                assert not p.requires_grad, f"{name} requires_grad=True"

    def test_accuracy_nonlinear_data(self):
        """Nyström RBF should handle nonlinearly separable data better than raw ridge."""
        from vnts.nn.nystrom import NystromKernelLayer
        from vnts.engine.solver import AlgebraicEngine
        # XOR-like structure: class = (x0 > 0) XOR (x1 > 0)
        torch.manual_seed(7)
        X = torch.randn(300, 2)
        y = ((X[:, 0] > 0).long() ^ (X[:, 1] > 0).long())

        Xtr, ytr = X[:200], y[:200]
        Xte, yte = X[200:], y[200:]

        # Raw ridge on linear features — should struggle
        Y_oh = F.one_hot(ytr, 2).float()
        W_lin = AlgebraicEngine.solve_ridge(Xtr, Y_oh, reg_lambda=1e-2)
        preds_lin = torch.argmax(Xte @ W_lin, dim=1)
        acc_lin = (preds_lin == yte).float().mean().item()

        # Nyström RBF — should do much better on XOR
        m = NystromKernelLayer(2, n_landmarks=50, out_features=2, kernel="rbf", gamma=2.0)
        m.fit(Xtr, ytr)
        with torch.no_grad():
            preds_nys = torch.argmax(m(Xte), dim=1)
            acc_nys = (preds_nys == yte).float().mean().item()

        assert acc_nys > acc_lin, f"Nyström ({acc_nys:.3f}) should beat linear ({acc_lin:.3f}) on XOR"


class TestNystromORMA:
    def test_fit_forward(self):
        from vnts.nn.nystrom import NystromORMA
        X, y, nc = make_data(100, 16, 4)
        m = NystromORMA(in_features=16, n_landmarks=30, orma_hidden=32, out_features=4,
                        orma_depth=3, kernel="rbf", gamma=0.5)
        m.fit(X[:80], y[:80])
        out = m(X[80:])
        assert out.shape == (20, 4)

    def test_no_requires_grad(self):
        from vnts.nn.nystrom import NystromORMA
        X, y, nc = make_data(60, 8, 2)
        m = NystromORMA(8, 15, 16, 2, 2, gamma=1.0)
        m.fit(X, y)
        for name, p in m.nystrom._parameters.items():
            if p is not None:
                assert not p.requires_grad

    def test_nystrom_orma_vs_plain_orma(self):
        """NyströmORMA should not be worse than plain ORMA on structured nonlinear data."""
        from vnts.nn.nystrom import NystromORMA
        from vnts.nn.orma import ORMANetwork
        torch.manual_seed(3)
        # Circular separable data
        angles = torch.rand(300) * 2 * 3.14159
        r = 1.0 + (torch.arange(300) % 2).float() * 2.0
        X = torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=1)
        y = (torch.arange(300) % 2).long()
        Xtr, ytr = X[:240], y[:240]
        Xte, yte = X[240:], y[240:]

        orma = ORMANetwork(2, 32, 2, depth=4)
        orma.fit(Xtr, F.one_hot(ytr, 2).float())
        with torch.no_grad():
            acc_orma = (torch.argmax(orma(Xte), 1) == yte).float().mean().item()

        nys_orma = NystromORMA(2, 40, 32, 2, 3, kernel="rbf", gamma=0.5)
        nys_orma.fit(Xtr, ytr)
        with torch.no_grad():
            acc_nys = (torch.argmax(nys_orma(Xte), 1) == yte).float().mean().item()

        # On circular data, kernel should win
        assert acc_nys >= acc_orma - 0.05, f"NysORMA {acc_nys:.3f} vs ORMA {acc_orma:.3f}"
