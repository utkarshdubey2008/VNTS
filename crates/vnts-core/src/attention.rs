//! Multi-Scale Orthogonal Recurrent Attention (MORA) in Native Rust.
//! Computes causal context evolution in O(1) memory without backpropagation through time.

use crate::matrix::Matrix;

pub struct NativeMORA {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_proj: Matrix,
    pub k_proj: Matrix,
    pub v_proj: Matrix,
    pub out_proj: Matrix,
    pub r_heads: Vec<Matrix>,
    pub alphas: Vec<f32>,
}

impl NativeMORA {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert_eq!(embed_dim % num_heads, 0, "embed_dim must be divisible by num_heads");
        let head_dim = embed_dim / num_heads;
        let total_dim = num_heads * head_dim;

        let scale = 1.0 / (embed_dim as f32).sqrt();
        let q_proj = Matrix::random(embed_dim, total_dim, scale);
        let k_proj = Matrix::random(embed_dim, total_dim, scale);
        let v_proj = Matrix::random(embed_dim, total_dim, scale);
        let out_proj = Matrix::random(total_dim, embed_dim, 1.0 / (total_dim as f32).sqrt());

        // Multi-head unitary rotation matrices in SO(head_dim)
        let mut r_heads = Vec::with_capacity(num_heads);
        for _ in 0..num_heads {
            let rand_mat = Matrix::random(head_dim, head_dim, 1.0);
            let (q, _) = crate::qr::QR::decompose(&rand_mat, 1e-4);
            r_heads.push(q);
        }

        // Multi-scale temporal decay spectrum
        let mut alphas = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let frac = if num_heads > 1 {
                h as f32 / (num_heads - 1) as f32
            } else {
                0.5
            };
            alphas.push(0.40 + frac * 0.55);
        }

        Self {
            embed_dim,
            num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            r_heads,
            alphas,
        }
    }

    /// Forward pass through MORA for a sequence of tokens X: (T x embed_dim).
    /// Returns Attention Output: (T x embed_dim).
    pub fn forward(&self, x: &Matrix) -> Matrix {
        let t_len = x.rows;
        let q_all = x.matmul(&self.q_proj);
        let k_all = x.matmul(&self.k_proj);
        let v_all = x.matmul(&self.v_proj);

        // Memory states for each head: (num_heads x head_dim x head_dim)
        let mut s_heads = vec![Matrix::zeros(self.head_dim, self.head_dim); self.num_heads];
        let mut output_flat = Matrix::zeros(t_len, self.num_heads * self.head_dim);

        for t in 0..t_len {
            let q_row = q_all.row(t);
            let k_row = k_all.row(t);
            let v_row = v_all.row(t);

            for h in 0..self.num_heads {
                let alpha = self.alphas[h];
                let r_mat = &self.r_heads[h];
                let s_mat = &mut s_heads[h];

                let h_offset = h * self.head_dim;
                let k_h = &k_row[h_offset..h_offset + self.head_dim];
                let v_h = &v_row[h_offset..h_offset + self.head_dim];
                let q_h = &q_row[h_offset..h_offset + self.head_dim];

                // Rotate state: S_rot = S @ R
                let s_rot = s_mat.matmul(r_mat);

                // Outer product KV = k^T @ v : (head_dim x head_dim)
                // S_next = alpha * S_rot + (1 - alpha) * KV
                for r in 0..self.head_dim {
                    let kr = k_h[r];
                    for c in 0..self.head_dim {
                        let kv_val = kr * v_h[c];
                        let new_s = alpha * s_rot.get(r, c) + (1.0 - alpha) * kv_val;
                        s_mat.set(r, c, new_s);
                    }
                }

                // Query output: O_h = q_h @ S : (head_dim,)
                for c in 0..self.head_dim {
                    let mut sum = 0.0f32;
                    for r in 0..self.head_dim {
                        sum += q_h[r] * s_mat.get(r, c);
                    }
                    output_flat.set(t, h_offset + c, sum);
                }
            }
        }

        // Final output projection
        output_flat.matmul(&self.out_proj)
    }
}
