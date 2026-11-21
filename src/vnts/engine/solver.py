"""
High-performance algebraic solvers for VNTS.
Provides optimal primal/dual ridge regressions, orthonormal subspace projection,
and streaming Woodbury updates without gradient computation.
"""

from typing import Tuple, Optional
import torch


class AlgebraicEngine:
    """
    Mathematical engine powering closed-form learning in VNTS.
    """

    @staticmethod
    def solve_ridge(
        X: torch.Tensor,
        Y: torch.Tensor,
        reg_lambda: float = 1e-3
    ) -> torch.Tensor:
        """
        Solves the regularized linear inverse problem:
            W* = argmin ||X W - Y||_F^2 + reg_lambda ||W||_F^2
        
        Dynamically selects primal or dual formulation:
        - If N >= D: Solves primal (X^T X + lambda I)^(-1) X^T Y  [O(N D^2 + D^3)]
        - If N < D:  Solves dual   X^T (X X^T + lambda I)^(-1) Y  [O(N^2 D + N^3)]
        
        Args:
            X: Input matrix of shape (N, D)
            Y: Target matrix of shape (N, K)
            reg_lambda: Tikhonov regularization scalar
            
        Returns:
            Optimal weight matrix W of shape (D, K)
        """
        N, D = X.shape
        device = X.device
        dtype = X.dtype

        if N >= D:
            # Primal formulation: Gram matrix (D, D)
            XtX = X.T @ X
            reg_matrix = reg_lambda * torch.eye(D, device=device, dtype=dtype)
            A = XtX + reg_matrix
            B = X.T @ Y
            try:
                # Cholesky solve is ~2x faster and numerically superior
                L = torch.linalg.cholesky(A)
                W = torch.cholesky_solve(B, L)
            except Exception:
                W = torch.linalg.solve(A, B)
        else:
            # Dual formulation: Kernel Gram matrix (N, N)
            XXt = X @ X.T
            reg_matrix = reg_lambda * torch.eye(N, device=device, dtype=dtype)
            A = XXt + reg_matrix
            try:
                L = torch.linalg.cholesky(A)
                alpha = torch.cholesky_solve(Y, L)
            except Exception:
                alpha = torch.linalg.solve(A, Y)
            W = X.T @ alpha

        return W

    @staticmethod
    def orthonormal_basis(H: torch.Tensor, threshold: float = 1e-5) -> torch.Tensor:
        """
        Extracts an orthonormal basis for the column space of H^T using QR decomposition.
        
        Args:
            H: Representation matrix (N, D)
            threshold: Numerical zero tolerance
            
        Returns:
            Orthonormal basis Q of shape (D, r) where Q^T Q = I_r
        """
        # We want basis for the row space of H (subspace spanned by sample representations in R^D)
        # That is the column space of H^T (D, N)
        Q, R = torch.linalg.qr(H.T, mode="reduced")
        # Keep non-zero diagonal entries in R
        diag_R = torch.abs(torch.diag(R))
        valid_cols = diag_R > threshold
        if torch.sum(valid_cols) == 0:
            return Q[:, :1]
        return Q[:, valid_cols]

    @staticmethod
    def solve_nullspace_projected_ridge(
        X: torch.Tensor,
        Y: torch.Tensor,
        U_prior: Optional[torch.Tensor],
        reg_lambda: float = 1e-3
    ) -> torch.Tensor:
        """
        Finds optimal Delta_W that minimizes ||X Delta_W - Y||^2
        strictly subject to U_prior^T Delta_W = 0 (Nullspace constraint).
        
        Projector P_null = I - U_prior @ U_prior^T.
        Delta_W* = P_null ( (P_null X^T X P_null + lambda I)^(-1) P_null X^T Y )
        
        Args:
            X: Input matrix (N, D)
            Y: Target residuals (N, K)
            U_prior: Orthonormal basis of prior subspace (D, r), where U^T U = I
            reg_lambda: Regularization weight
            
        Returns:
            Delta_W of shape (D, K) with U_prior^T Delta_W == 0
        """
        D = X.shape[1]
        device = X.device
        dtype = X.dtype

        if U_prior is None or U_prior.shape[1] == 0:
            return AlgebraicEngine.solve_ridge(X, Y, reg_lambda=reg_lambda)

        # Compute P_null = I - U U^T
        # For efficiency with large D, we project X directly:
        # X_proj = X @ P_null = X - (X @ U) @ U^T
        X_proj = X - (X @ U_prior) @ U_prior.T

        # Solve ridge regression on projected features
        dW = AlgebraicEngine.solve_ridge(X_proj, Y, reg_lambda=reg_lambda)

        # Enforce exact nullspace projection on dW to eliminate floating-point drift
        dW_null = dW - U_prior @ (U_prior.T @ dW)
        return dW_null

    @staticmethod
    def woodbury_update(
        P: torch.Tensor,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Online rank-1 Recursive Least Squares (RLS) update via the Sherman-Morrison-Woodbury formula.
        Maintains inverse covariance matrix P = (X^T X + lambda I)^(-1) in O(D^2) per step.
        
        Args:
            P: Current inverse covariance matrix (D, D)
            x: New incoming feature vector (D,)
            
        Returns:
            (P_updated, gain_k)
        """
        # P_x = P @ x : shape (D,)
        Px = P @ x
        denom = 1.0 + torch.dot(x, Px)
        k = Px / denom
        P_next = P - torch.outer(k, Px)
        return P_next, k
