//! Cholesky Factorization and Closed-Form Ridge Solvers in pure Rust.

use crate::matrix::Matrix;

pub struct Cholesky;

impl Cholesky {
    /// Computes lower-triangular factor L such that A = L @ L^T.
    /// Returns None if matrix is not positive-definite.
    pub fn decompose(a: &Matrix) -> Option<Matrix> {
        assert_eq!(a.rows, a.cols, "Matrix must be square for Cholesky");
        let n = a.rows;
        let mut l = Matrix::zeros(n, n);

        for j in 0..n {
            let mut d = 0.0f32;
            for k in 0..j {
                let ljk = l.get(j, k);
                d += ljk * ljk;
            }
            let diag_val = a.get(j, j) - d;
            if diag_val <= 0.0 {
                return None; // Not positive definite
            }
            let ljj = diag_val.sqrt();
            l.set(j, j, ljj);

            for i in (j + 1)..n {
                let mut s = 0.0f32;
                for k in 0..j {
                    s += l.get(i, k) * l.get(j, k);
                }
                let lij = (a.get(i, j) - s) / ljj;
                l.set(i, j, lij);
            }
        }

        Some(l)
    }

    /// Solves L @ y = B for lower triangular matrix L.
    pub fn forward_substitute(l: &Matrix, b: &Matrix) -> Matrix {
        let n = l.rows;
        let k = b.cols;
        let mut y = Matrix::zeros(n, k);

        for col in 0..k {
            for i in 0..n {
                let mut sum = 0.0f32;
                for j in 0..i {
                    sum += l.get(i, j) * y.get(j, col);
                }
                let val = (b.get(i, col) - sum) / l.get(i, i);
                y.set(i, col, val);
            }
        }
        y
    }

    /// Solves L^T @ X = Y for lower triangular matrix L.
    pub fn backward_substitute(l: &Matrix, y: &Matrix) -> Matrix {
        let n = l.rows;
        let k = y.cols;
        let mut x = Matrix::zeros(n, k);

        for col in 0..k {
            for i in (0..n).rev() {
                let mut sum = 0.0f32;
                for j in (i + 1)..n {
                    sum += l.get(j, i) * x.get(j, col); // Note: L^T(i, j) = L(j, i)
                }
                let val = (y.get(i, col) - sum) / l.get(i, i);
                x.set(i, col, val);
            }
        }
        x
    }

    /// Solves A @ X = B where A is symmetric positive-definite via Cholesky.
    pub fn solve(a: &Matrix, b: &Matrix) -> Option<Matrix> {
        let l = Self::decompose(a)?;
        let y = Self::forward_substitute(&l, b);
        let x = Self::backward_substitute(&l, &y);
        Some(x)
    }

    /// Solves regularized linear inverse: W* = argmin ||X W - Y||^2 + lambda ||W||^2
    /// Automatically selects Primal (N >= D) or Dual (N < D) for O(min(N, D)^2 max(N, D)).
    pub fn solve_ridge(x: &Matrix, y: &Matrix, reg_lambda: f32) -> Matrix {
        let n = x.rows;
        let d = x.cols;

        if n >= d {
            // Primal formulation: A = X^T X + lambda I, B = X^T Y
            let xt = x.transpose();
            let mut xtx = xt.matmul(x);
            for i in 0..d {
                xtx.set(i, i, xtx.get(i, i) + reg_lambda);
            }
            let xty = xt.matmul(y);

            match Self::solve(&xtx, &xty) {
                Some(w) => w,
                None => {
                    // Fallback with boosted regularization if ill-conditioned
                    for i in 0..d {
                        xtx.set(i, i, xtx.get(i, i) + reg_lambda * 10.0);
                    }
                    Self::solve(&xtx, &xty).unwrap_or_else(|| Matrix::zeros(d, y.cols))
                }
            }
        } else {
            // Dual formulation: A = X X^T + lambda I, alpha = A^-1 Y, W = X^T alpha
            let xt = x.transpose();
            let mut xxt = x.matmul(&xt);
            for i in 0..n {
                xxt.set(i, i, xxt.get(i, i) + reg_lambda);
            }

            let alpha = match Self::solve(&xxt, y) {
                Some(a) => a,
                None => {
                    for i in 0..n {
                        xxt.set(i, i, xxt.get(i, i) + reg_lambda * 10.0);
                    }
                    Self::solve(&xxt, y).unwrap_or_else(|| Matrix::zeros(n, y.cols))
                }
            };

            xt.matmul(&alpha)
        }
    }
}
