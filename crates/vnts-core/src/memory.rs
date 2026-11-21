//! Native Continual Lifelong Memory Engine in pure Rust.

use crate::matrix::Matrix;
use crate::nullspace::NullspaceEngine;

#[derive(Debug, Clone)]
pub struct NativeOrthogonalMemory {
    pub in_features: usize,
    pub out_features: usize,
    pub reg_lambda: f32,
    pub weight: Matrix,
    pub u_basis: Option<Matrix>,
    pub num_tasks_learned: usize,
}

impl NativeOrthogonalMemory {
    pub fn new(in_features: usize, out_features: usize, reg_lambda: f32) -> Self {
        Self {
            in_features,
            out_features,
            reg_lambda,
            weight: Matrix::zeros(in_features, out_features),
            u_basis: None,
            num_tasks_learned: 0,
        }
    }

    pub fn forward(&self, x: &Matrix) -> Matrix {
        x.matmul(&self.weight)
    }

    /// Assimilates a task dataset (X, Y) without catastrophic forgetting:
    /// Delta_W is constrained strictly to the orthonormal nullspace P_null = I - Q Q^T.
    pub fn learn_task(&mut self, x: &Matrix, y: &Matrix) -> usize {
        assert_eq!(x.cols, self.in_features);
        assert_eq!(y.cols, self.out_features);

        if self.num_tasks_learned == 0 || self.u_basis.is_none() {
            // Task 1: Direct closed-form Cholesky Ridge solve
            let new_w = crate::cholesky::Cholesky::solve_ridge(x, y, self.reg_lambda);
            self.weight = new_w;
            self.u_basis = Some(crate::qr::QR::row_space_basis(x, 1e-4));
            self.num_tasks_learned = 1;
            return self.u_basis.as_ref().unwrap().cols;
        }

        // Sequential Task: Residual solve in orthogonal nullspace
        let current_preds = self.forward(x);
        let residuals = y.sub(&current_preds);

        let delta_w = NullspaceEngine::solve_nullspace_projected(
            x,
            &residuals,
            self.u_basis.as_ref(),
            self.reg_lambda,
        );

        self.weight = self.weight.add(&delta_w);

        // Update basis of occupied subspaces
        let updated_u = NullspaceEngine::merge_basis(self.u_basis.as_ref(), x, 1e-4);
        let current_rank = updated_u.cols;
        self.u_basis = Some(updated_u);
        self.num_tasks_learned += 1;

        current_rank
    }
}
