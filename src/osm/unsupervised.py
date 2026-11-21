"""
CSD: Contrastive Subspace Dispersion & Unsupervised Manifold Alignment
Enables self-supervised representation learning completely without labels,
without negative pair sampling, and without gradient backpropagation.
Solves representation alignment via exact Generalized Canonical Subspace Decomposition.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn


class ContrastiveSubspaceLearner:
    """
    Self-Supervised Representation Learner using closed-form Canonical Subspace Alignment.
    Finds optimal feature projection W that maximizes mutual information between augmented views
    while preventing representation collapse via orthogonal dispersion.
    """
    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 64,
        reg_lambda: float = 1e-3,
        seed: Optional[int] = None
    ):
        if seed is not None:
            torch.manual_seed(seed)
            
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.reg_lambda = reg_lambda
        
        # Learned projection operator: (in_dim, latent_dim)
        self.W = torch.randn(in_dim, latent_dim) * (1.0 / in_dim)**0.5
        self.W.requires_grad = False

    def fit_views(self, H_a: torch.Tensor, H_b: torch.Tensor):
        """
        Learns optimal alignment matrix W in closed form:
        H_a: Representations of View A (N, in_dim)
        H_b: Representations of View B (N, in_dim)
        """
        N, d = H_a.shape
        
        # Center the representations
        H_a_centered = H_a - H_a.mean(dim=0, keepdim=True)
        H_b_centered = H_b - H_b.mean(dim=0, keepdim=True)
        
        # 1. Compute empirical cross-covariance and auto-covariances
        C_aa = (H_a_centered.T @ H_a_centered) / N + self.reg_lambda * torch.eye(d, device=H_a.device)
        C_bb = (H_b_centered.T @ H_b_centered) / N + self.reg_lambda * torch.eye(d, device=H_b.device)
        C_ab = (H_a_centered.T @ H_b_centered) / N
        
        # 2. Whitening operators via matrix square root / eigen-decomposition
        # C_aa^(-1/2)
        L_a, V_a = torch.linalg.eigh(C_aa)
        inv_sqrt_L_a = torch.diag(1.0 / torch.sqrt(torch.clamp(L_a, min=1e-6)))
        C_aa_inv_sqrt = V_a @ inv_sqrt_L_a @ V_a.T
        
        # C_bb^(-1/2)
        L_b, V_b = torch.linalg.eigh(C_bb)
        inv_sqrt_L_b = torch.diag(1.0 / torch.sqrt(torch.clamp(L_b, min=1e-6)))
        C_bb_inv_sqrt = V_b @ inv_sqrt_L_b @ V_b.T
        
        # 3. Normalized cross-covariance matrix:
        K = C_aa_inv_sqrt @ C_ab @ C_bb_inv_sqrt
        
        # 4. SVD of normalized cross-covariance:
        U, S, Vh = torch.linalg.svd(K)
        
        # 5. The canonical projection matrix for view A:
        # Select top latent_dim modes
        k = min(self.latent_dim, U.shape[1])
        U_k = U[:, :k]
        
        W_opt = C_aa_inv_sqrt @ U_k
        self.W = W_opt.clone()

    def transform(self, H: torch.Tensor) -> torch.Tensor:
        """Projects representations into the aligned contrastive subspace."""
        return H @ self.W
