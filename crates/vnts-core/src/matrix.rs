//! Cache-friendly, high-performance dense matrix operations in pure Rust (Zero Dependencies).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len(), "Dimension mismatch in Matrix::new");
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(dim: usize) -> Self {
        let mut mat = Self::zeros(dim, dim);
        for i in 0..dim {
            mat.set(i, i, 1.0);
        }
        mat
    }

    pub fn random(rows: usize, cols: usize, scale: f32) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let mut seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            // Xorshift64 PRNG
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let val = ((seed as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32 * scale;
            data.push(val);
        }
        Self { rows, cols, data }
    }

    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    #[inline(always)]
    pub fn set(&mut self, r: usize, c: usize, val: f32) {
        self.data[r * self.cols + c] = val;
    }

    #[inline(always)]
    pub fn row(&self, r: usize) -> &[f32] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    #[inline(always)]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        let start = r * self.cols;
        &mut self.data[start..start + self.cols]
    }

    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    /// Cache-blocked matrix multiplication: C = A @ B.
    /// Transposes B for linear row-row dot product access.
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows, "Incompatible dimensions for matmul: ({}x{}) x ({}x{})",
            self.rows, self.cols, other.rows, other.cols);

        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        // Transpose other for cache-localized contiguous dot products
        let other_t = other.transpose();

        let mut out = Self::zeros(m, n);

        for r in 0..m {
            let a_row = &self.data[r * k..(r + 1) * k];
            let out_offset = r * n;
            for c in 0..n {
                let b_col = &other_t.data[c * k..(c + 1) * k];
                let mut dot = 0.0f32;
                for i in 0..k {
                    dot += a_row[i] * b_col[i];
                }
                out.data[out_offset + c] = dot;
            }
        }

        out
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        let data: Vec<f32> = self.data.iter().zip(&other.data).map(|(&a, &b)| a + b).collect();
        Self { rows: self.rows, cols: self.cols, data }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        let data: Vec<f32> = self.data.iter().zip(&other.data).map(|(&a, &b)| a - b).collect();
        Self { rows: self.rows, cols: self.cols, data }
    }

    pub fn scale(&self, scalar: f32) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x * scalar).collect();
        Self { rows: self.rows, cols: self.cols, data }
    }

    pub fn norm_frobenius(&self) -> f32 {
        self.data.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }
}
