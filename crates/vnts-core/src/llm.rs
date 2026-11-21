//! High-Performance Native Language Model Engine in pure Rust.
//! Uses Multi-Head Causal Orthogonal Manifold Tracking (MH-COMT).
//! Memory complexity: O(1) per token, zero BPTT, zero GPUs.

use crate::cholesky::Cholesky;
use crate::matrix::Matrix;
use std::collections::HashMap;

pub struct NativeMicroLLM {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub reg_lambda: f32,
    pub embeddings: Matrix,
    pub r_heads: Vec<Matrix>,
    pub alphas: Vec<f32>,
    pub w_out: Matrix,
    pub char2idx: HashMap<char, usize>,
    pub idx2char: HashMap<usize, char>,
}

impl NativeMicroLLM {
    pub fn from_corpus(
        corpus: &str,
        embed_dim: usize,
        num_heads: usize,
        reg_lambda: f32,
    ) -> Self {
        assert_eq!(embed_dim % num_heads, 0, "embed_dim must be divisible by num_heads");
        let head_dim = embed_dim / num_heads;

        // Distinct character vocabulary
        let mut chars: Vec<char> = corpus.chars().collect();
        chars.sort_unstable();
        chars.dedup();

        let vocab_size = chars.len().max(2);
        let mut char2idx = HashMap::new();
        let mut idx2char = HashMap::new();
        for (idx, &ch) in chars.iter().enumerate() {
            char2idx.insert(ch, idx);
            idx2char.insert(idx, ch);
        }

        // 1. Normalized Random Token Embeddings: (vocab_size x embed_dim)
        let scale = 1.0 / (embed_dim as f32).sqrt();
        let embeddings = Matrix::random(vocab_size, embed_dim, scale);

        // 2. Unitary Rotation Matrices R_h in SO(head_dim) via QR
        let mut r_heads = Vec::with_capacity(num_heads);
        for _ in 0..num_heads {
            let rand_mat = Matrix::random(head_dim, head_dim, 1.0);
            let (q, _) = crate::qr::QR::decompose(&rand_mat, 1e-4);
            r_heads.push(q);
        }

        // Multi-scale temporal retention factors from 0.40 to 0.95
        let mut alphas = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let frac = if num_heads > 1 {
                h as f32 / (num_heads - 1) as f32
            } else {
                0.5
            };
            alphas.push(0.40 + frac * 0.55);
        }

        // Readout weights: (embed_dim x vocab_size)
        let w_out = Matrix::zeros(embed_dim, vocab_size);

        Self {
            vocab_size,
            embed_dim,
            num_heads,
            head_dim,
            reg_lambda,
            embeddings,
            r_heads,
            alphas,
            w_out,
            char2idx,
            idx2char,
        }
    }

    pub fn encode_text(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|ch| *self.char2idx.get(&ch).unwrap_or(&0))
            .collect()
    }

    pub fn decode_tokens(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|tok| self.idx2char.get(tok).copied().unwrap_or('?'))
            .collect()
    }

    /// Autoregressively tracks multi-head orthogonal context for a sequence of tokens.
    /// Returns Context Matrix (T x embed_dim).
    pub fn track_context(&self, tokens: &[usize]) -> Matrix {
        let t_len = tokens.len();
        let mut contexts = Matrix::zeros(t_len, self.embed_dim);

        // Head states: (num_heads x head_dim)
        let mut c_heads = vec![vec![0.0f32; self.head_dim]; self.num_heads];

        for t in 0..t_len {
            let tok_idx = tokens[t];
            let emb_row = self.embeddings.row(tok_idx);

            let mut unified_context = Vec::with_capacity(self.embed_dim);

            for h in 0..self.num_heads {
                let alpha = self.alphas[h];
                let r_mat = &self.r_heads[h];
                let emb_slice = &emb_row[h * self.head_dim..(h + 1) * self.head_dim];

                // Unitary evolution: c_rot = c @ R
                let mut c_rot = vec![0.0f32; self.head_dim];
                for c in 0..self.head_dim {
                    let mut sum = 0.0f32;
                    for r in 0..self.head_dim {
                        sum += c_heads[h][r] * r_mat.get(r, c);
                    }
                    c_rot[c] = sum;
                }

                // Update head state: c_new = alpha * c_rot + (1 - alpha) * emb
                for d in 0..self.head_dim {
                    let new_val = alpha * c_rot[d] + (1.0 - alpha) * emb_slice[d];
                    c_heads[h][d] = new_val;
                    unified_context.push(new_val);
                }
            }

            // Write unified context row for timestep t
            for (col, &val) in unified_context.iter().enumerate() {
                contexts.set(t, col, val);
            }
        }

        contexts
    }

    /// Trains the language model on a text corpus in a single forward pass without backpropagation.
    pub fn train(&mut self, text: &str, seq_len: usize) -> (f32, usize) {
        let tokens = self.encode_text(text);
        let total_tokens = tokens.len();
        assert!(total_tokens > seq_len, "Corpus must be longer than seq_len");

        let num_chunks = (total_tokens - 1) / seq_len;
        let total_pairs = num_chunks * seq_len;

        let mut x_rows = Vec::with_capacity(total_pairs * self.embed_dim);
        let mut y_rows = Vec::with_capacity(total_pairs * self.vocab_size);

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * seq_len;
            let end = start + seq_len + 1;
            let chunk = &tokens[start..end];

            // Context for chunk[:-1]
            let chunk_inputs = &chunk[0..seq_len];
            let contexts = self.track_context(chunk_inputs);

            // Targets chunk[1:]
            let chunk_targets = &chunk[1..=seq_len];

            for t in 0..seq_len {
                // Feature row
                let ctx_row = contexts.row(t);
                x_rows.extend_from_slice(ctx_row);

                // One-hot target row
                let mut target_row = vec![0.0f32; self.vocab_size];
                target_row[chunk_targets[t]] = 1.0;
                y_rows.extend_from_slice(&target_row);
            }
        }

        let x_mat = Matrix::new(total_pairs, self.embed_dim, x_rows);
        let y_mat = Matrix::new(total_pairs, self.vocab_size, y_rows);

        // Optimal closed-form Cholesky solve
        let new_w = Cholesky::solve_ridge(&x_mat, &y_mat, self.reg_lambda);
        self.w_out = new_w;

        // Compute training accuracy
        let preds = x_mat.matmul(&self.w_out);
        let mut correct = 0usize;
        for i in 0..total_pairs {
            let row = preds.row(i);
            let mut max_val = f32::NEG_INFINITY;
            let mut argmax = 0usize;
            for (c, &val) in row.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    argmax = c;
                }
            }

            let true_target = tokens[i % seq_len + 1 + (i / seq_len) * seq_len];
            if argmax == true_target {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / total_pairs as f32;
        (accuracy, total_pairs)
    }

    /// Generates autoregressive text continuation from prompt.
    pub fn generate(&self, prompt: &str, max_tokens: usize, temperature: f32) -> String {
        let mut tokens = self.encode_text(prompt);
        if tokens.is_empty() {
            tokens.push(0);
        }

        for _ in 0..max_tokens {
            let window_start = if tokens.len() > 64 { tokens.len() - 64 } else { 0 };
            let active_window = &tokens[window_start..];

            let context_mat = self.track_context(active_window);
            let last_ctx = context_mat.row(active_window.len() - 1);

            // Logits = last_ctx @ W_out
            let mut logits = vec![0.0f32; self.vocab_size];
            for c in 0..self.vocab_size {
                let mut sum = 0.0f32;
                for r in 0..self.embed_dim {
                    sum += last_ctx[r] * self.w_out.get(r, c);
                }
                logits[c] = sum / temperature.max(1e-3);
            }

            // Softmax sampling
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
            let probs: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp() / exp_sum).collect();

            // Simple random sample
            use std::time::{SystemTime, UNIX_EPOCH};
            let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
            let mut rand_val = (nanos as f32 / 1_000_000_000.0) % 1.0;

            let mut next_token = 0usize;
            for (idx, &p) in probs.iter().enumerate() {
                if rand_val <= p {
                    next_token = idx;
                    break;
                }
                rand_val -= p;
                next_token = idx;
            }

            tokens.push(next_token);
        }

        self.decode_tokens(&tokens)
    }
}
