pub mod attention;
pub mod c_api;
pub mod cholesky;
pub mod llm;
pub mod matrix;
pub mod memory;
pub mod nullspace;
pub mod qr;
pub mod woodbury;

pub use attention::NativeMORA;
pub use cholesky::Cholesky;
pub use llm::NativeMicroLLM;
pub use matrix::Matrix;
pub use memory::NativeOrthogonalMemory;
pub use nullspace::NullspaceEngine;
pub use qr::QR;
pub use woodbury::Woodbury;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_matmul() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_cholesky_solve() {
        let a = Matrix::new(2, 2, vec![4.0, 2.0, 2.0, 3.0]);
        let b = Matrix::new(2, 1, vec![6.0, 5.0]);
        let x = Cholesky::solve(&a, &b).expect("Cholesky failed");
        assert!((x.get(0, 0) - 1.0).abs() < 1e-4);
        assert!((x.get(1, 0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_qr_orthonormality() {
        let a = Matrix::random(10, 4, 1.0);
        let (q, _) = QR::decompose(&a, 1e-4);
        let qtq = q.transpose().matmul(&q);
        for i in 0..qtq.rows {
            for j in 0..qtq.cols {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((qtq.get(i, j) - expected).abs() < 1e-3, "QR orthonormality error");
            }
        }
    }

    #[test]
    fn test_nullspace_invariance() {
        let d = 32;
        let mut mem = NativeOrthogonalMemory::new(d, 2, 1e-3);

        let x1 = Matrix::random(30, d, 1.0);
        let y1 = Matrix::random(30, 2, 1.0);
        mem.learn_task(&x1, &y1);
        let p1_before = mem.forward(&x1);

        let x2 = Matrix::random(30, d, 1.0);
        let y2 = Matrix::random(30, 2, 1.0);
        mem.learn_task(&x2, &y2);
        let p1_after = mem.forward(&x1);

        let diff = p1_after.sub(&p1_before).norm_frobenius() / p1_before.norm_frobenius();
        assert!(diff < 1e-4, "Task 1 representation drift: {}", diff);
    }

    #[test]
    fn test_native_llm_training_and_generation() {
        let text = "vector nullspace transformation system in pure rust. ";
        let mut llm = NativeMicroLLM::from_corpus(text, 32, 2, 1e-3);
        let (acc, tokens) = llm.train(text, 8);
        assert!(tokens > 0);
        assert!(acc > 0.3);

        let gen = llm.generate("vector", 15, 0.7);
        assert!(gen.starts_with("vector"));
    }

    #[test]
    fn test_native_mora_attention() {
        let t_len = 16;
        let embed_dim = 32;
        let num_heads = 2;
        let mora = NativeMORA::new(embed_dim, num_heads);
        let x = Matrix::random(t_len, embed_dim, 1.0);
        let out = mora.forward(&x);
        assert_eq!((out.rows, out.cols), (t_len, embed_dim));
    }
}
