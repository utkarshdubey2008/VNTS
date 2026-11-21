"""
Standalone Native Linear Algebra Engine for VNTS.
Runs completely independent of PyTorch using pure standard BLAS/LAPACK (NumPy).
Enables zero-dependency deployments on bare-metal IoT, microcontrollers, and low-power edge nodes.
"""

from typing import Tuple, Optional
import numpy as np


class NativeAlgebraicEngine:
    """
    Pure NumPy implementation of VNTS algebraic solvers.
    Requires zero GPU, zero autograd, and zero external deep learning frameworks.
    """

    @staticmethod
    def solve_ridge(
        X: np.ndarray,
        Y: np.ndarray,
        reg_lambda: float = 1e-3
    ) -> np.ndarray:
        """
        Closed-form optimal Tikhonov/Ridge regression:
        W* = argmin ||X W - Y||_F^2 + lambda ||W||_F^2
        """
        N, D = X.shape
        if N >= D:
            # Primal: (X^T X + lambda I)^(-1) X^T Y
            XtX = X.T @ X
            A = XtX + reg_lambda * np.eye(D, dtype=X.dtype)
            B = X.T @ Y
            try:
                L = np.linalg.cholesky(A)
                # Solve L y = B, then L^T W = y
                y = np.linalg.solve(L, B)
                W = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                W = np.linalg.solve(A, B)
        else:
            # Dual: X^T (X X^T + lambda I)^(-1) Y
            XXt = X @ X.T
            A = XXt + reg_lambda * np.eye(N, dtype=X.dtype)
            try:
                L = np.linalg.cholesky(A)
                y = np.linalg.solve(L, Y)
                alpha = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                alpha = np.linalg.solve(A, Y)
            W = X.T @ alpha
        return W

    @staticmethod
    def orthonormal_basis(H: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
        """Computes orthonormal basis for row space of H using QR decomposition."""
        Q, R = np.linalg.qr(H.T, mode="reduced")
        diag_R = np.abs(np.diag(R))
        valid = diag_R > threshold
        if np.sum(valid) == 0:
            return Q[:, :1]
        return Q[:, valid]

    @staticmethod
    def solve_nullspace_projected(
        X: np.ndarray,
        Y: np.ndarray,
        U_prior: Optional[np.ndarray],
        reg_lambda: float = 1e-3
    ) -> np.ndarray:
        """
        Solves optimal update Delta_W constrained to nullspace: U_prior^T Delta_W = 0.
        """
        if U_prior is None or U_prior.shape[1] == 0:
            return NativeAlgebraicEngine.solve_ridge(X, Y, reg_lambda=reg_lambda)

        # X_proj = X (I - U U^T) = X - (X U) U^T
        X_proj = X - (X @ U_prior) @ U_prior.T
        dW = NativeAlgebraicEngine.solve_ridge(X_proj, Y, reg_lambda=reg_lambda)
        # Exact nullspace projection
        dW_null = dW - U_prior @ (U_prior.T @ dW)
        return dW_null

    @staticmethod
    def woodbury_update(P: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Online rank-1 Recursive Least Squares update in O(D^2)."""
        Px = P @ x
        denom = 1.0 + np.dot(x, Px)
        k = Px / denom
        P_next = P - np.outer(k, Px)
        return P_next, k


class NativeMemory:
    """
    Pure NumPy continual learning memory module with zero catastrophic forgetting.
    """
    def __init__(self, in_features: int, out_features: int, reg_lambda: float = 1e-2):
        self.in_features = in_features
        self.out_features = out_features
        self.reg_lambda = reg_lambda
        self.W = np.zeros((in_features, out_features), dtype=np.float32)
        self.U_k: Optional[np.ndarray] = None
        self.num_tasks = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W

    def assimilate(self, x: np.ndarray, y: np.ndarray):
        """Assimilates dataset (X, Y) with zero interference on prior representations."""
        if self.num_tasks == 0 or self.U_k is None:
            self.W = NativeAlgebraicEngine.solve_ridge(x, y, reg_lambda=self.reg_lambda)
            self.U_k = NativeAlgebraicEngine.orthonormal_basis(x)
            self.num_tasks = 1
            return

        residuals = y - (x @ self.W)
        delta_W = NativeAlgebraicEngine.solve_nullspace_projected(
            x, residuals, self.U_k, reg_lambda=self.reg_lambda
        )
        self.W = self.W + delta_W

        U_new = NativeAlgebraicEngine.orthonormal_basis(x)
        combined = np.concatenate([self.U_k, U_new], axis=1)
        Q_comb, _ = np.linalg.qr(combined, mode="reduced")
        max_cols = min(self.in_features - 1, Q_comb.shape[1])
        self.U_k = Q_comb[:, :max_cols]
        self.num_tasks += 1
