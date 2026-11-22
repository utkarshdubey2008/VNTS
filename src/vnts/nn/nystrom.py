"""
VNTS Nyström Kernel Approximation Layer
========================================
Implements closed-form kernel ridge regression via the Nyström approximation.

Standard ORMA layers use random projections (W_proj) which cap out at a linear
feature boundary — good for structured data, weak on complex nonlinear manifolds.

The Nyström method approximates the full kernel matrix K ≈ K_xm K_mm^{-1} K_mx
using m << N landmark points, then solves kernel ridge regression exactly.

This provides proper nonlinear function approximation (RBF, polynomial, etc.)
with:
  - O(N m^2) training cost (vs O(N^3) for exact kernel methods)
  - O(m) inference cost per sample
  - Zero backpropagation — purely algebraic
  - Full mathematical convergence guarantees

Combined with ORMA residual cascades:
  NyströmORMA = Nyström feature map + ORMA deep residual stack on mapped features
"""
from __future__ import annotations

import time
from typing import Optional, Literal, Dict

import torch
import torch.nn.functional as F

from ..base import VNTSModule
from ..engine.solver import AlgebraicEngine
from .orma import ORMANetwork


class NystromKernelLayer(VNTSModule):
    """
    Nyström approximation of a kernel feature map.

    Given training data X (N, D), selects m landmark points X_m,
    computes approximate feature map φ(x) = K(x, X_m) K_mm^{-1/2}
    of dimension m, and trains a ridge regressor on these features.

    Supports: 'rbf' (Gaussian), 'polynomial', 'linear' kernels.
    """

    def __init__(
        self,
        in_features: int,
        n_landmarks: int = 100,
        out_features: int = 1,
        kernel: Literal["rbf", "polynomial", "linear"] = "rbf",
        gamma: float = 1.0,     # RBF bandwidth / polynomial scale
        degree: int = 2,        # polynomial degree
        reg_lambda: float = 1e-3,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_landmarks = n_landmarks
        self.out_features = out_features
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.reg_lambda = reg_lambda

        # Set during fit()
        self._landmarks: Optional[torch.Tensor] = None   # (m, D)
        self._K_mm_inv_sqrt: Optional[torch.Tensor] = None  # (m, m)

        W = torch.zeros(n_landmarks, out_features)
        self.register_parameter("W", W)

    @property
    def W(self) -> torch.Tensor:
        return self._parameters["W"]

    def _compute_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix K(X, Y) of shape (len(X), len(Y))."""
        if self.kernel == "rbf":
            # K(x, y) = exp(-gamma ||x-y||^2)
            dists = torch.cdist(X, Y, p=2).pow(2)
            return torch.exp(-self.gamma * dists)
        elif self.kernel == "polynomial":
            # K(x, y) = (gamma * x^T y + 1)^degree
            dots = X @ Y.T
            return (self.gamma * dots + 1.0).pow(self.degree)
        else:  # linear
            return X @ Y.T

    def _fit_nystrom(self, X: torch.Tensor) -> torch.Tensor:
        """
        Select landmarks and compute K_mm^{-1/2} for feature mapping.
        Returns the mapped features Phi (N, m).
        """
        N = X.shape[0]
        m = min(self.n_landmarks, N)

        # Random landmark selection (without replacement)
        perm = torch.randperm(N)[:m]
        X_m = X[perm]           # (m, D)
        self._landmarks = X_m

        # Kernel between landmarks: K_mm (m, m)
        K_mm = self._compute_kernel(X_m, X_m)

        # Regularize for numerical stability
        K_mm = K_mm + self.reg_lambda * torch.eye(m)

        # K_mm^{-1/2} via eigendecomposition: K_mm = V Λ V^T → K_mm^{-1/2} = V Λ^{-1/2} V^T
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(K_mm)
            eigenvalues = eigenvalues.clamp(min=1e-10)
            self._K_mm_inv_sqrt = eigenvectors @ torch.diag(eigenvalues.pow(-0.5)) @ eigenvectors.T
        except Exception:
            # Fallback: Cholesky-based pseudoinverse
            L = torch.linalg.cholesky(K_mm)
            self._K_mm_inv_sqrt = torch.cholesky_inverse(L)

        # Map training data: Phi = K(X, X_m) K_mm^{-1/2}  (N, m)
        K_xm = self._compute_kernel(X, X_m)   # (N, m)
        return K_xm @ self._K_mm_inv_sqrt       # (N, m)

    def _map(self, X: torch.Tensor) -> torch.Tensor:
        """Map new samples X to Nyström features Phi (N, m)."""
        assert self._landmarks is not None, "Call fit() first"
        K_xm = self._compute_kernel(X, self._landmarks)
        return K_xm @ self._K_mm_inv_sqrt

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> Dict:
        """Fit Nyström feature map and solve kernel ridge regression."""
        t0 = time.perf_counter()
        X = X.float()
        Y = Y.float() if Y.dim() > 1 else F.one_hot(Y, self.out_features).float()

        Phi = self._fit_nystrom(X)        # (N, m)
        W_new = AlgebraicEngine.solve_ridge(Phi, Y, reg_lambda=self.reg_lambda)
        self._parameters["W"] = W_new

        return {
            "n_landmarks": self._landmarks.shape[0],
            "feature_dim": Phi.shape[1],
            "fit_time_s": time.perf_counter() - t0,
        }

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._map(X.float()) @ self.W


class NystromORMA(VNTSModule):
    """
    Nyström Kernel + ORMA Deep Residual Cascade.

    Architecture:
      1. Nyström kernel layer maps X → Phi (nonlinear, N×m features)
      2. ORMA residual cascade on Phi (deep closed-form representation)

    This combines:
      - Proper nonlinear feature richness (kernel trick via Nyström)
      - Depth via monotone residual correction (ORMA)
      - Zero backpropagation end-to-end
    """

    def __init__(
        self,
        in_features: int,
        n_landmarks: int = 128,
        orma_hidden: int = 128,
        out_features: int = 1,
        orma_depth: int = 4,
        kernel: str = "rbf",
        gamma: float = 0.01,
        reg_lambda: float = 1e-2,
    ):
        super().__init__()
        self.nystrom = NystromKernelLayer(
            in_features=in_features,
            n_landmarks=n_landmarks,
            out_features=out_features,
            kernel=kernel,
            gamma=gamma,
            reg_lambda=reg_lambda,
        )
        self.orma = ORMANetwork(
            in_features=n_landmarks,
            hidden_dim=orma_hidden,
            out_features=out_features,
            depth=orma_depth,
            reg_lambda=reg_lambda,
        )
        self._fitted_nystrom = False

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> Dict:
        t0 = time.perf_counter()
        X = X.float()

        # Stage 1: Nyström feature map
        nys_info = self.nystrom.fit(X, Y)
        Phi = self.nystrom._map(X)   # (N, m)
        self._fitted_nystrom = True

        # Stage 2: ORMA on Phi
        if Y.dim() == 1:
            Y_oh = F.one_hot(Y, self.nystrom.out_features).float()
        else:
            Y_oh = Y.float()
        orma_info = self.orma.fit(Phi, Y_oh)

        return {
            "nystrom": nys_info,
            "orma": orma_info,
            "total_fit_time_s": time.perf_counter() - t0,
        }

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Phi = self.nystrom._map(X.float())
        return self.orma(Phi)
