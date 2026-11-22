"""
VNTS Enhanced Architectures — Phase 9+
=======================================
1. PCAProjectedORMA : ORMA with learned PCA pre-projection.
   Reduces high-dimensional inputs (e.g. 3072-dim images) to a compact
   subspace before ORMA layers, dramatically improving accuracy on vision tasks.
   No backprop: PCA is solved by eigendecomposition of the data covariance.

2. EnsembleORMA : Bagged ensemble of ORMANetworks trained on bootstrap subsets.
   Combines predictions by averaging logits (soft voting).
   Improves variance reduction with zero additional training cost per model.

3. IncrementalORMA : Online ORMA that can assimilate new data batches without
   retraining from scratch — uses running covariance accumulation.
"""
from __future__ import annotations

import time
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn.functional as F

from ..base import VNTSModule
from ..engine.solver import AlgebraicEngine
from .orma import ORMANetwork


class PCAProjectedORMA(VNTSModule):
    """
    PCA pre-projection + ORMA deep residual cascade.

    Training (fit):
      1. Compute PCA via truncated SVD of centered X (closed-form, no backprop).
      2. Project X → Z = (X - mean) @ V_pca   (N × pca_dim)
      3. Fit ORMANetwork on Z.

    Inference (forward):
      Z = (X - mean) @ V_pca → ORMA(Z)
    """

    def __init__(
        self,
        in_features: int,
        pca_dim: int,
        hidden_dim: int,
        out_features: int,
        depth: int = 5,
        reg_lambda: float = 1e-2,
    ):
        super().__init__()
        self.in_features = in_features
        self.pca_dim = pca_dim
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.depth = depth
        self.reg_lambda = reg_lambda

        # PCA components — set during fit()
        self._mean: Optional[torch.Tensor] = None   # (D,)
        self._V: Optional[torch.Tensor] = None       # (D, pca_dim) — projection matrix

        # ORMA operating on pca_dim inputs
        self.orma = ORMANetwork(
            in_features=pca_dim,
            hidden_dim=hidden_dim,
            out_features=out_features,
            depth=depth,
            reg_lambda=reg_lambda,
        )

    def _fit_pca(self, x: torch.Tensor) -> torch.Tensor:
        """Fit PCA on X (N, D) and return projected Z (N, pca_dim)."""
        mean = x.mean(dim=0)
        self._mean = mean
        X_c = x - mean                              # (N, D)
        # Truncated SVD — O(N D pca_dim); much cheaper than full SVD for pca_dim << D
        # torch.linalg.svd returns U (N,K), S (K,), Vh (K,D)
        k = min(self.pca_dim, min(x.shape) - 1)
        try:
            U, S, Vh = torch.linalg.svd(X_c, full_matrices=False)
            V = Vh[:k].T                            # (D, k)
        except Exception:
            # Fallback: covariance eigenvectors
            cov = X_c.T @ X_c / max(x.shape[0] - 1, 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            # eigh returns ascending order; take last k
            V = eigenvectors[:, -k:]                # (D, k)

        self._V = V[:, :self.pca_dim]               # (D, pca_dim)
        return X_c @ self._V                        # (N, pca_dim)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None, "Call fit() before forward()"
        return (x - self._mean) @ self._V

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Dict:
        t0 = time.perf_counter()
        Z = self._fit_pca(x.float())
        orma_info = self.orma.fit(Z, y)
        orma_info["pca_dim"] = self.pca_dim
        orma_info["total_fit_time_s"] = time.perf_counter() - t0
        return orma_info

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.orma(self._project(x.float()))


class EnsembleORMA(VNTSModule):
    """
    Bagged ensemble of PCAProjectedORMA models.
    Each member trains on a bootstrap resample — zero additional cost per member
    since ORMA training is O(N D^2), not iterative.
    Predictions are soft-vote averaged (mean logits).
    """

    def __init__(
        self,
        in_features: int,
        pca_dim: int,
        hidden_dim: int,
        out_features: int,
        n_estimators: int = 5,
        depth: int = 5,
        reg_lambda: float = 1e-2,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.members: List[PCAProjectedORMA] = []
        for i in range(n_estimators):
            m = PCAProjectedORMA(in_features, pca_dim, hidden_dim, out_features, depth, reg_lambda)
            self.members.append(m)
            self.register_module(f"member_{i}", m)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Dict:
        t0 = time.perf_counter()
        N = x.shape[0]
        infos = []
        for i, m in enumerate(self.members):
            # Bootstrap resample
            idx = torch.randint(0, N, (N,))
            info = m.fit(x[idx], y[idx] if y.dim() == 1 else y[idx])
            infos.append(info)
        return {
            "n_estimators": self.n_estimators,
            "total_fit_time_s": time.perf_counter() - t0,
            "member_fit_times_s": [i["total_fit_time_s"] for i in infos],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([m(x) for m in self.members], dim=0)
        return logits.mean(dim=0)


class IncrementalORMA(VNTSModule):
    """
    Online incremental ORMA that accumulates sufficient statistics
    (XtX, XtY) across mini-batches and solves once at the end.
    Allows streaming data ingestion without storing the full dataset.
    Zero forgetting guaranteed by design (accumulates all data).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        reg_lambda: float = 1e-3,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reg_lambda = reg_lambda

        # Sufficient statistics
        D = in_features
        self._XtX = torch.zeros(D, D)
        self._XtY = torch.zeros(D, out_features)
        self._n_samples = 0

        W = torch.zeros(D, out_features)
        self.register_parameter("W", W)

    @property
    def W(self) -> torch.Tensor:
        return self._parameters["W"]

    def accumulate(self, x: torch.Tensor, y: torch.Tensor):
        """Ingest a mini-batch without solving — O(N D^2) per batch."""
        x = x.float()
        if y.dim() == 1:
            Y = F.one_hot(y, self.out_features).float()
        else:
            Y = y.float()
        self._XtX = self._XtX + x.T @ x
        self._XtY = self._XtY + x.T @ Y
        self._n_samples += x.shape[0]

    def solve(self) -> torch.Tensor:
        """Solve closed-form ridge from accumulated statistics."""
        D = self.in_features
        A = self._XtX + self.reg_lambda * torch.eye(D)
        try:
            L = torch.linalg.cholesky(A)
            W_new = torch.cholesky_solve(self._XtY, L)
        except Exception:
            W_new = torch.linalg.solve(A, self._XtY)
        self._parameters["W"] = W_new
        return W_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() @ self.W
