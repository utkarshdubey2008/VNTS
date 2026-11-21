//! Streaming Sherman-Morrison-Woodbury Inverse Covariance Tracker in pure Rust.

use crate::matrix::Matrix;

pub struct Woodbury;

impl Woodbury {
    /// Performs an online rank-1 Recursive Least Squares (RLS) update:
    /// P_next = P - (P @ x @ x^T @ P) / (1 + x^T @ P @ x)
    /// Runs in O(D^2) operations per step.
    ///
    /// Returns: (P_next, gain_vector_k)
    pub fn update(p: &Matrix, x_vec: &[f32]) -> (Matrix, Vec<f32>) {
        let d = p.rows;
        assert_eq!(d, p.cols, "P matrix must be square");
        assert_eq!(d, x_vec.len(), "Vector dimension mismatch in Woodbury");

        // Compute px = P @ x : (D,)
        let mut px = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d {
                sum += p.get(i, j) * x_vec[j];
            }
            px[i] = sum;
        }

        // Compute denom = 1.0 + x^T @ px
        let mut xt_px = 0.0f32;
        for i in 0..d {
            xt_px += x_vec[i] * px[i];
        }
        let denom = 1.0f32 + xt_px;

        // Gain vector k = px / denom : (D,)
        let k: Vec<f32> = px.iter().map(|&val| val / denom).collect();

        // P_next = P - k @ px^T
        let mut p_next = p.clone();
        for i in 0..d {
            let ki = k[i];
            for j in 0..d {
                let delta = ki * px[j];
                p_next.set(i, j, p_next.get(i, j) - delta);
            }
        }

        (p_next, k)
    }

    /// Adapts weight matrix W online given incoming feature x_vec (D,) and target y_vec (K,):
    /// err = y - x @ W
    /// W_next = W + k @ err^T
    pub fn adapt_weights(
        p: &mut Matrix,
        w: &mut Matrix,
        x_vec: &[f32],
        y_vec: &[f32],
    ) {
        let (p_next, k) = Self::update(p, x_vec);
        *p = p_next;

        let d = w.rows;
        let num_targets = w.cols;

        // Current prediction: y_hat = x @ W
        let mut y_hat = vec![0.0f32; num_targets];
        for c in 0..num_targets {
            let mut sum = 0.0f32;
            for r in 0..d {
                sum += x_vec[r] * w.get(r, c);
            }
            y_hat[c] = sum;
        }

        // Error: err = y - y_hat
        let mut err = vec![0.0f32; num_targets];
        for c in 0..num_targets {
            err[c] = y_vec[c] - y_hat[c];
        }

        // Update: W[r, c] += k[r] * err[c]
        for r in 0..d {
            let kr = k[r];
            for c in 0..num_targets {
                w.set(r, c, w.get(r, c) + kr * err[c]);
            }
        }
    }
}
