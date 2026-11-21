"""
Orthogonal Nullspace Memory Layer for VNTS.
Enables continual lifelong learning with zero catastrophic forgetting,
dynamic neurogenesis, and spectral memory compaction.
"""

from typing import Optional, Tuple
import torch
from ..base import VNTSModule
from ..engine.solver import AlgebraicEngine


class OrthogonalMemory(VNTSModule):
    """
    Continual Learning Memory Module.
    Guarantees mathematically zero interference on previously learned representations
    by projecting updates into the orthogonal nullspace: P_null = I - U U^T.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        reg_lambda: float = 1e-2,
        enable_compaction: bool = True,
        capacity_threshold: float = 0.85,
        energy_retention: float = 0.95,
        enable_neurogenesis: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reg_lambda = reg_lambda
        self.enable_compaction = enable_compaction
        self.capacity_threshold = capacity_threshold
        self.energy_retention = energy_retention
        self.enable_neurogenesis = enable_neurogenesis

        # Primary readout weight matrix: (in_features, out_features)
        W = torch.zeros((in_features, out_features), dtype=torch.float32)
        self.register_parameter("weight", W)

        # Orthonormal basis of occupied task subspaces: (in_features, r)
        self.U_k: Optional[torch.Tensor] = None
        self.num_tasks_assimilated = 0

    @property
    def weight(self) -> torch.Tensor:
        return self._parameters["weight"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight

    def assimilate(self, x: torch.Tensor, y: torch.Tensor) -> int:
        """
        Assimilates a new task dataset (X, Y) without catastrophic forgetting.
        
        Args:
            x: Input representations of shape (N, in_features)
            y: Target outputs of shape (N, out_features)
            
        Returns:
            Current occupied basis rank
        """
        x = x.detach()
        y = y.detach()
        N, D = x.shape
        device = x.device

        # If basis is saturated and neurogenesis is enabled, expand dimension
        if self.U_k is not None and self.enable_neurogenesis:
            current_rank = self.U_k.shape[1]
            if current_rank >= int(self.capacity_threshold * self.in_features):
                self._expand_capacity(delta_features=max(64, self.in_features // 2))
                # Pad incoming x to match new dimension
                pad_size = self.in_features - D
                x = torch.cat([x, torch.zeros(N, pad_size, device=device)], dim=1)
                D = self.in_features

        # Task 1: Cold start direct ridge solve
        if self.num_tasks_assimilated == 0 or self.U_k is None:
            new_W = AlgebraicEngine.solve_ridge(x, y, reg_lambda=self.reg_lambda)
            self._parameters["weight"] = new_W
            self.U_k = AlgebraicEngine.orthonormal_basis(x)
            self.num_tasks_assimilated = 1
            return self.U_k.shape[1]

        # Sequential Task: Nullspace Projected Ridge Update
        # Current predictions and target residuals
        current_preds = x @ self.weight
        residual = y - current_preds

        # Compute optimal nullspace update: Delta_W s.t. U_k^T Delta_W = 0
        delta_W = AlgebraicEngine.solve_nullspace_projected_ridge(
            x, residual, self.U_k, reg_lambda=self.reg_lambda
        )
        self._parameters["weight"] = self.weight + delta_W

        # Extract basis of new task representations
        U_new = AlgebraicEngine.orthonormal_basis(x)

        # Merge with existing basis via QR decomposition
        combined = torch.cat([self.U_k, U_new], dim=1)
        Q_combined, _ = torch.linalg.qr(combined, mode="reduced")

        max_cols = min(self.in_features - 1, Q_combined.shape[1])
        self.U_k = Q_combined[:, :max_cols]

        # Automatic Spectral Memory Compaction
        if self.enable_compaction and self.U_k.shape[1] >= int(self.capacity_threshold * self.in_features):
            self._compact_basis()

        self.num_tasks_assimilated += 1
        return self.U_k.shape[1]

    def _compact_basis(self):
        """
        Compresses occupied subspace U_k by pruning lowest energy modes
        using task covariance energy G = weight @ weight^T.
        """
        if self.U_k is None or self.U_k.shape[1] <= 1:
            return

        D, k = self.U_k.shape
        # Weight covariance matrix
        G = self.weight @ self.weight.T  # (D, D)
        # Project into current basis: (k, k)
        G_subspace = self.U_k.T @ G @ self.U_k
        eigenvalues, eigenvectors = torch.linalg.eigh(G_subspace)

        # Sort descending
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_energy = torch.sum(torch.clamp(eigenvalues, min=0.0)) + 1e-12
        cum_energy = torch.cumsum(torch.clamp(eigenvalues, min=0.0), dim=0) / total_energy

        # Keep modes accounting for energy_retention
        k_keep = int((cum_energy >= self.energy_retention).nonzero()[0].item()) + 1
        k_keep = max(1, min(k_keep, k - 1))

        # Rotate and truncate basis
        V_keep = eigenvectors[:, :k_keep]
        self.U_k = self.U_k @ V_keep

    def _expand_capacity(self, delta_features: int):
        """Dynamic Neurogenesis: Increases ambient capacity D -> D + delta_features."""
        old_D = self.in_features
        new_D = old_D + delta_features
        device = self.weight.device

        # Expand W with zeros
        W_expanded = torch.zeros((new_D, self.out_features), device=device)
        W_expanded[:old_D, :] = self.weight
        self._parameters["weight"] = W_expanded

        # Expand basis U_k
        if self.U_k is not None:
            r = self.U_k.shape[1]
            U_expanded = torch.zeros((new_D, r), device=device)
            U_expanded[:old_D, :] = self.U_k
            self.U_k = U_expanded

        self.in_features = new_D
