//! Modified Gram-Schmidt QR Decomposition for Orthonormal Subspace Basis Extraction.

use crate::matrix::Matrix;

pub struct QR;

impl QR {
    /// Computes thin/reduced QR decomposition of A (M x N, M >= N): A = Q @ R
    /// where Q is M x r with Q^T Q = I_r, and R is r x N upper triangular.
    pub fn decompose(a: &Matrix, tol: f32) -> (Matrix, Matrix) {
        let m = a.rows;
        let n = a.cols;

        // Columns of A
        let mut v = Vec::with_capacity(n);
        for j in 0..n {
            let mut col = Vec::with_capacity(m);
            for i in 0..m {
                col.push(a.get(i, j));
            }
            v.push(col);
        }

        let mut q_cols: Vec<Vec<f32>> = Vec::new();
        let mut r_entries: Vec<(usize, usize, f32)> = Vec::new();

        for j in 0..n {
            let mut w = v[j].clone();

            for (k, q_k) in q_cols.iter().enumerate() {
                // Projection of w onto q_k
                let mut dot = 0.0f32;
                for i in 0..m {
                    dot += q_k[i] * w[i];
                }
                r_entries.push((k, j, dot));
                for i in 0..m {
                    w[i] -= dot * q_k[i];
                }
            }

            // Norm of orthogonalized vector
            let norm = w.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > tol {
                r_entries.push((q_cols.len(), j, norm));
                let q_col: Vec<f32> = w.iter().map(|&x| x / norm).collect();
                q_cols.push(q_col);
            }
        }

        let r_rank = q_cols.len().max(1);
        let mut q_mat = Matrix::zeros(m, r_rank);
        for (col_idx, col_data) in q_cols.iter().enumerate() {
            for row_idx in 0..m {
                q_mat.set(row_idx, col_idx, col_data[row_idx]);
            }
        }

        let mut r_mat = Matrix::zeros(r_rank, n);
        for (k, j, val) in r_entries {
            if k < r_rank {
                r_mat.set(k, j, val);
            }
        }

        (q_mat, r_mat)
    }

    /// Extracts orthonormal basis for the row-space of H (N x D) -> returns Q (D x r).
    pub fn row_space_basis(h: &Matrix, tol: f32) -> Matrix {
        // The row space of H is the column space of H^T (D x N)
        let ht = h.transpose();
        let (q, _) = Self::decompose(&ht, tol);
        q
    }
}
