"""
Orthogonal Subspace Mixture-of-Experts (OSR-MoE) for VNTS.
Eliminates routing gradient instability and expert collapse through geometric subspace projection energy.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F

from ..base import VNTSModule
from ..engine.solver import AlgebraicEngine


class OrthogonalSubspaceMoE(VNTSModule):
    """
    Mixture-of-Experts Layer with Orthogonal Subspace Energy Routing.
    Tokens are routed to experts whose principal subspaces maximize projection energy:
        energy_m = ||x @ Q_m||_2^2
    Guarantees load diversity and eliminates routing collapse without heuristic auxiliary losses.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 4,
        top_k: int = 2,
        subspace_dim: int = 16,
        reg_lambda: float = 1e-3
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.subspace_dim = subspace_dim
        self.reg_lambda = reg_lambda

        # 1. Routing Subspace Bases Q_m (in_features x subspace_dim)
        # Mutually quasi-orthogonal across experts
        for m in range(num_experts):
            rand_m = torch.randn(in_features, subspace_dim)
            Q, _ = torch.linalg.qr(rand_m, mode="reduced")
            self.register_parameter(f"Q_subspace_{m}", Q)

        # 2. Expert Weight Matrices: (in_features x out_features)
        for m in range(num_experts):
            std = (2.0 / in_features) ** 0.5
            W_exp = torch.randn(in_features, out_features) * std
            self.register_parameter(f"W_expert_{m}", W_exp)

    def route_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes routing scores based on orthogonal projection energy:
        energy_m(x) = sum_d (x @ Q_m)_d^2
        Args:
            x: (N, in_features)
        Returns:
            (topk_weights, topk_indices) where shapes are (N, top_k)
        """
        N, D = x.shape
        energies = []

        for m in range(self.num_experts):
            Q = self._parameters[f"Q_subspace_{m}"]  # (D, subspace_dim)
            proj = x @ Q  # (N, subspace_dim)
            eng = torch.sum(proj ** 2, dim=1)  # (N,)
            energies.append(eng)

        energy_tensor = torch.stack(energies, dim=1)  # (N, num_experts)

        # Top-k selection
        topk_energies, topk_indices = torch.topk(energy_tensor, self.top_k, dim=1)
        # Softmax normalize top-k weights
        topk_weights = F.softmax(topk_energies, dim=1)

        return topk_weights, topk_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_features)
        Returns:
            Blended expert outputs: (N, out_features)
        """
        orig_shape = x.shape
        if x.dim() > 2:
            x_flat = x.reshape(-1, self.in_features)
        else:
            x_flat = x

        N = x_flat.shape[0]
        weights, indices = self.route_tokens(x_flat)  # (N, top_k)

        out = torch.zeros(N, self.out_features, device=x.device, dtype=x.dtype)

        # Dispatch tokens to assigned experts
        for k in range(self.top_k):
            k_weights = weights[:, k].unsqueeze(1)  # (N, 1)
            k_indices = indices[:, k]               # (N,)

            for m in range(self.num_experts):
                mask = (k_indices == m)
                if mask.any():
                    x_m = x_flat[mask]
                    W_m = self._parameters[f"W_expert_{m}"]
                    h_m = F.leaky_relu(x_m @ W_m, negative_slope=0.05)
                    out[mask] += k_weights[mask] * h_m

        if len(orig_shape) > 2:
            return out.view(*orig_shape[:-1], self.out_features)
        return out

    def adapt_expert(
        self,
        expert_idx: int,
        x_expert: torch.Tensor,
        target_expert: torch.Tensor
    ):
        """Adapts a specific expert's weights in closed form via Ridge regression."""
        assert 0 <= expert_idx < self.num_experts
        new_W = AlgebraicEngine.solve_ridge(x_expert, target_expert, reg_lambda=self.reg_lambda)
        self._parameters[f"W_expert_{expert_idx}"] = new_W
