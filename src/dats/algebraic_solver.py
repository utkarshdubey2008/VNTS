"""
Exact algebraic solvers and pseudoinverse projectors for DATS-SI.
Zero gradients, zero backpropagation.
"""

import torch


def regularized_pseudoinverse(W: torch.Tensor, gamma: float = 1e-4) -> torch.Tensor:
    """
    Computes regularized pseudoinverse W_pinv of shape (d_out, d_in)
    such that Z @ W_pinv projects from (N, d_out) back to (N, d_in).
    
    If W has shape (d_in, d_out):
    - When d_in >= d_out: W_pinv = (W^T W + gamma * I)^(-1) W^T  [shape: (d_out, d_in)]
    - When d_in < d_out:  W_pinv = W^T (W W^T + gamma * I)^(-1)  [shape: (d_out, d_in)]
    """
    d_in, d_out = W.shape
    # Ensure float32 and check for any NaN/Inf
    W_clean = torch.nan_to_num(W, nan=0.0, posinf=1e4, neginf=-1e4)
    
    if d_in >= d_out:
        G = W_clean.T @ W_clean + gamma * torch.eye(d_out, dtype=W.dtype, device=W.device)
        try:
            return torch.linalg.solve(G, W_clean.T)
        except (RuntimeError, torch._C._LinAlgError):
            return torch.linalg.pinv(G) @ W_clean.T
    else:
        G = W_clean @ W_clean.T + gamma * torch.eye(d_in, dtype=W.dtype, device=W.device)
        try:
            X = torch.linalg.solve(G, W_clean)
            return X.T
        except (RuntimeError, torch._C._LinAlgError):
            return (torch.linalg.pinv(G) @ W_clean).T


def solve_ridge_subspace(H_aug: torch.Tensor, Z_target: torch.Tensor, reg_lambda: float = 1e-4) -> torch.Tensor:
    """
    Solves the exact Tikhonov regularized projection:
    argmin_W || H_aug W - Z_target ||_F^2 + reg_lambda * || W ||_F^2
    
    Normal equations:
    (H_aug^T H_aug + reg_lambda * I) W = H_aug^T Z_target
    
    Args:
        H_aug: Augmented input matrix of shape (N, d_in + 1)
        Z_target: Target pre-activations of shape (N, d_out)
        reg_lambda: Ridge regularization parameter (L2 penalty)
        
    Returns:
        W_aug: Optimal affine parameter matrix of shape (d_in + 1, d_out)
    """
    d_feat = H_aug.shape[1]
    cov = H_aug.T @ H_aug
    reg_matrix = reg_lambda * torch.eye(d_feat, dtype=H_aug.dtype, device=H_aug.device)
    A = cov + reg_matrix
    B = H_aug.T @ Z_target
    
    # Use positive definite linear solve (Cholesky-based) for speed and stability
    try:
        W_aug = torch.linalg.solve(A, B)
    except RuntimeError:
        # Fallback to pseudo-inverse if singular
        W_aug = torch.linalg.pinv(A) @ B
        
    return W_aug


class RecursiveLeastSquaresTracker:
    """
    Online / Streaming Recursive Least Squares (RLS) solver using Woodbury rank-1 updates.
    Enables single-pass streaming learning without batch retention or backprop.
    """
    def __init__(self, d_in: int, d_out: int, delta: float = 1.0, forgetting_factor: float = 0.999):
        self.d_in = d_in + 1  # includes bias
        self.d_out = d_out
        self.lambda_forget = forgetting_factor
        self.P = delta * torch.eye(self.d_in, dtype=torch.float32)
        self.W_aug = torch.zeros((self.d_in, d_out), dtype=torch.float32)

    def update(self, h_aug: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """
        Processes a single sample or mini-batch:
        h_aug: (B, d_in + 1)
        z_target: (B, d_out)
        """
        # Batch RLS update
        B = h_aug.shape[0]
        for i in range(B):
            h_i = h_aug[i:i+1].T  # (d_in + 1, 1)
            z_i = z_target[i:i+1].T  # (d_out, 1)
            
            Pi_h = self.P @ h_i  # (d_in + 1, 1)
            denom = self.lambda_forget + (h_i.T @ Pi_h).item()
            k = Pi_h / denom  # Gain vector: (d_in + 1, 1)
            
            error = z_i.T - (h_i.T @ self.W_aug)  # (1, d_out)
            self.W_aug = self.W_aug + (k @ error)
            
            # Rank-1 update of inverse covariance P
            self.P = (self.P - (k @ Pi_h.T)) / self.lambda_forget
            
        return self.W_aug
