//! Nullspace Projectors and Zero-Forgetting Parameter Modulations in pure Rust.

use crate::cholesky::Cholesky;
use crate::matrix::Matrix;
use crate::qr::QR;

pub struct NullspaceEngine;

impl NullspaceEngine {
    /// Computes X_projected = X @ (I - U @ U^T) = X - (X @ U) @ U^T
    /// Avoids forming large D x D matrix explicitly.
    pub fn project_features(x: &Matrix, u: &Matrix) -> Matrix {
        // x: (N x D), u: (D x r) where U^T U = I_r
        let xu = x.matmul(u); // (N x r)
        let ut = u.transpose(); // (r x D)
        let recon = xu.matmul(&ut); // (N x D)
        x.sub(&recon)
    }

    /// Solves optimal parameter update Delta_W constrained strictly to the nullspace:
    /// U^T @ Delta_W = 0 (Zero Interference with prior representations).
    pub fn solve_nullspace_projected(
        x: &Matrix,
        y: &Matrix,
        u_prior: Option<&Matrix>,
        reg_lambda: f32,
    ) -> Matrix {
        match u_prior {
            None => Cholesky::solve_ridge(x, y, reg_lambda),
            Some(u) if u.cols == 0 => Cholesky::solve_ridge(x, y, reg_lambda),
            Some(u) => {
                // 1. Project features into nullspace
                let x_proj = Self::project_features(x, u);

                // 2. Closed-form Ridge solve on projected features
                let dw = Cholesky::solve_ridge(&x_proj, y, reg_lambda);

                // 3. Exact re-projection of dW to eliminate floating point numerical drift
                // dW_null = dW - U @ (U^T @ dW)
                let ut = u.transpose();
                let ut_dw = ut.matmul(&dw); // (r x K)
                let drift = u.matmul(&ut_dw); // (D x K)
                dw.sub(&drift)
            }
        }
    }

    /// Merges an existing basis U_old (D x r1) with new representations H_new (N x D)
    /// into a unified orthonormal basis Q_merged (D x r_total).
    pub fn merge_basis(u_old: Option<&Matrix>, h_new: &Matrix, tol: f32) -> Matrix {
        let u_new = QR::row_space_basis(h_new, tol);

        match u_old {
            None => u_new,
            Some(u) => {
                let d = u.rows;
                let r1 = u.cols;
                let r2 = u_new.cols;

                // Stack columns: (D x (r1 + r2))
                let mut combined = Matrix::zeros(d, r1 + r2);
                for r in 0..d {
                    for c in 0..r1 {
                        combined.set(r, c, u.get(r, c));
                    }
                    for c in 0..r2 {
                        combined.set(r, r1 + c, u_new.get(r, c));
                    }
                }

                // QR orthonormalization
                let (q, _) = QR::decompose(&combined, tol);
                q
            }
        }
    }
}
