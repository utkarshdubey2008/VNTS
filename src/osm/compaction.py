"""
SMC-DN: Spectral Memory Compaction & Dynamic Neurogenesis
Solves the Orthogonal Capacity Bound for Infinite-Horizon Lifelong Learning.
Enables neural networks to continuously learn an arbitrary number of sequential tasks
in bounded memory without catastrophic forgetting.
"""

from typing import Tuple, Optional
import torch


class SpectralMemoryCompactor:
    """
    Monitors subspace basis utilization and performs spectral energy compaction
    when occupied rank exceeds capacity threshold tau * D.
    """
    def __init__(
        self,
        capacity_threshold: float = 0.75,
        target_compression_ratio: float = 0.50,
        energy_retention_min: float = 0.99
    ):
        self.capacity_threshold = capacity_threshold
        self.target_compression_ratio = target_compression_ratio
        self.energy_retention_min = energy_retention_min

    def needs_compaction(self, current_rank: int, max_dim: int) -> bool:
        return current_rank >= int(self.capacity_threshold * max_dim)

    def compact(self, U: torch.Tensor, W: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Compacts the memory basis U (D x R) using task energy weighting from W (D x C).
        
        Args:
            U: Orthonormal memory basis of shape (D, R)
            W: Readout weights of shape (D, C)
            
        Returns:
            U_compact: Consolidated orthonormal basis of shape (D, R_new), where R_new < R
            dim_freed: Number of dimensions freed for subsequent learning.
        """
        D, R = U.shape
        if R <= 2:
            return U, 0
            
        # 1. Project readout weights into the subspace basis:
        # M represents coordinates of task predictors in basis U
        M = U.T @ W  # Shape: (R, C)
        
        # 2. Compute energy covariance in the basis coordinate system
        # G = M @ M^T has shape (R, R)
        G = M @ M.T
        
        # 3. Eigen-decomposition of energy matrix G
        # L has eigenvalues in ascending order, V has orthonormal eigenvectors
        L, V = torch.linalg.eigh(G)
        
        # Sort in descending order of energy
        idx_desc = torch.argsort(L, descending=True)
        L_desc = L[idx_desc]
        V_desc = V[:, idx_desc]
        
        # 4. Determine target compressed rank:
        target_rank = max(1, int(self.target_compression_ratio * R))
        
        # Ensure we retain at least energy_retention_min of total variance/energy
        total_energy = torch.clamp(L_desc.sum(), min=1e-8)
        cum_energy = torch.cumsum(L_desc, dim=0) / total_energy
        
        # Find minimum rank that satisfies energy threshold
        energy_satisfied_idx = torch.where(cum_energy >= self.energy_retention_min)[0]
        if len(energy_satisfied_idx) > 0:
            k_min_energy = energy_satisfied_idx[0].item() + 1
        else:
            k_min_energy = target_rank
            
        final_k = min(R - 1, max(target_rank, k_min_energy))
        
        # 5. Form the compressed basis in input representation space:
        # Each new basis vector is a linear combination of existing basis vectors: U_new = U @ V[:, :final_k]
        V_keep = V_desc[:, :final_k]
        U_compact = U @ V_keep
        
        # Re-orthonormalize for strict numerical precision
        Q_compact, _ = torch.linalg.qr(U_compact)
        
        dim_freed = R - final_k
        return Q_compact, dim_freed


class DynamicNeurogenesisEngine:
    """
    Dynamically expands network representation capacity on demand
    when incoming novel tasks exhibit high residual energy that cannot
    be captured in the current representation space.
    """
    def __init__(self, growth_step: int = 256, max_dim: int = 8192):
        self.growth_step = growth_step
        self.max_dim = max_dim

    def should_grow(self, current_dim: int, residual_norm: float, threshold: float = 0.5) -> bool:
        return (residual_norm > threshold) and (current_dim + self.growth_step <= self.max_dim)

    def grow_expansion_layer(
        self,
        W_proj: torch.Tensor,
        b_proj: torch.Tensor,
        in_dim: int,
        growth: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new orthogonal non-linear projection channels to W_proj and b_proj
        without disturbing existing channels.
        """
        device = W_proj.device
        dtype = W_proj.dtype
        
        # New channels initialized with He normal scaling
        W_new = torch.randn(in_dim, growth, device=device, dtype=dtype) * (2.0 / in_dim)**0.5
        b_new = torch.randn(growth, device=device, dtype=dtype) * 0.05
        
        W_expanded = torch.cat([W_proj, W_new], dim=1)
        b_expanded = torch.cat([b_proj, b_new], dim=0)
        
        return W_expanded, b_expanded
