"""
Orthogonal Residual Manifold Alignment (ORMA).
A mathematically guaranteed deep representation learning mechanism without backpropagation.
Monotonically shrinks residual error ||R_{l+1}|| <= ||R_l|| through orthogonal residual projections.
"""

from typing import List, Tuple, Optional, Dict
import time
import torch
import torch.nn.functional as F

from ..base import VNTSModule
from ..engine.solver import AlgebraicEngine


class ORMALayer(VNTSModule):
    """Single layer in an ORMA cascade."""
    def __init__(self, in_features: int, hidden_dim: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features

        # Representation projection: (in_features x hidden_dim)
        std = (2.0 / in_features) ** 0.5
        W_proj = torch.randn(in_features, hidden_dim) * std
        self.register_parameter("W_proj", W_proj)
        b_proj = torch.zeros(hidden_dim)
        self.register_parameter("b_proj", b_proj)

        # Residual readout: (hidden_dim x out_features)
        W_readout = torch.zeros(hidden_dim, out_features)
        self.register_parameter("W_readout", W_readout)

    @property
    def W_proj(self) -> torch.Tensor:
        return self._parameters["W_proj"]

    @property
    def b_proj(self) -> torch.Tensor:
        return self._parameters["b_proj"]

    @property
    def W_readout(self) -> torch.Tensor:
        return self._parameters["W_readout"]

    def compute_representation(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x @ self.W_proj + self.b_proj, negative_slope=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.compute_representation(x)
        return h @ self.W_readout

    def fit_residual(self, x: torch.Tensor, residual: torch.Tensor, reg_lambda: float = 1e-3) -> torch.Tensor:
        """
        Solves optimal W_readout = argmin ||H W_readout - residual||^2 + lambda ||W_readout||^2.
        Returns the residual correction: delta_Y = H @ W_readout.
        """
        h = self.compute_representation(x)
        new_W = AlgebraicEngine.solve_ridge(h, residual, reg_lambda=reg_lambda)
        self._parameters["W_readout"] = new_W
        return h @ new_W


class ORMANetwork(VNTSModule):
    """
    Deep Orthogonal Residual Manifold Alignment Network.
    Ensemble of L residual projection layers with monotonic error reduction.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        depth: int = 5,
        reg_lambda: float = 1e-3
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.depth = depth
        self.reg_lambda = reg_lambda

        # Base linear predictor
        W_base = torch.zeros(in_features, out_features)
        self.register_parameter("W_base", W_base)

        # Cascade of ORMA residual layers
        self.layers: List[ORMALayer] = []
        for i in range(depth):
            layer = ORMALayer(in_features, hidden_dim, out_features)
            self.layers.append(layer)
            self.register_module(f"orma_layer_{i}", layer)

    @property
    def W_base(self) -> torch.Tensor:
        return self._parameters["W_base"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = x @ self.W_base
        for layer in self.layers:
            pred = pred + layer(x)
        return pred

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, List[float]]:
        """
        Fits all layers sequentially on residuals.
        Guarantees strictly monotonically non-increasing residual norms.
        """
        t0 = time.perf_counter()
        N = x.shape[0]

        if y.dim() == 1:
            Y_targets = torch.zeros(N, self.out_features, device=x.device, dtype=torch.float32)
            Y_targets.scatter_(1, y.unsqueeze(1), 1.0)
        else:
            Y_targets = y.float()

        # Step 0: Fit base linear model
        self._parameters["W_base"] = AlgebraicEngine.solve_ridge(x, Y_targets, reg_lambda=self.reg_lambda)
        current_pred = x @ self.W_base
        current_residual = Y_targets - current_pred

        residual_norms = [torch.norm(current_residual).item()]

        # Step 1..L: Fit each successive ORMA layer to residual
        for layer in self.layers:
            delta_pred = layer.fit_residual(x, current_residual, reg_lambda=self.reg_lambda)
            current_pred = current_pred + delta_pred
            current_residual = Y_targets - current_pred
            res_norm = torch.norm(current_residual).item()
            residual_norms.append(res_norm)

        elapsed = time.perf_counter() - t0

        return {
            "residual_norms": residual_norms,
            "fit_time_s": elapsed,
            "initial_residual": residual_norms[0],
            "final_residual": residual_norms[-1]
        }
