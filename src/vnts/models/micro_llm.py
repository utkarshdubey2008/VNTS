"""
Micro-LLM: Lightweight Causal Language Model powered by Multi-Head COMT.
Runs autoregressive text learning and generation on commodity CPUs without BPTT or GPUs.
"""

from typing import List, Dict, Optional, Tuple
import time
import torch
import torch.nn.functional as F

from ..base import VNTSModule
from ..engine.solver import AlgebraicEngine


class MicroLLM(VNTSModule):
    """
    Zero-autograd Micro-LLM using Multi-Head Causal Orthogonal Manifold Tracking (MH-COMT).
    Each head operates at a distinct temporal decay rate alpha_h with unitary rotation R_h in SO(d_h).
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        reg_lambda: float = 1e-3
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.reg_lambda = reg_lambda

        # Character/Token vocabulary mapping
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}

        # 1. Token Embeddings: (vocab_size, embed_dim)
        embeddings = torch.randn(vocab_size, embed_dim) / (embed_dim ** 0.5)
        self.register_parameter("embeddings", embeddings)

        # 2. Multi-Head Unitary Rotation Matrices: (num_heads, head_dim, head_dim)
        R_heads = torch.zeros(num_heads, self.head_dim, self.head_dim)
        for h in range(num_heads):
            M = torch.randn(self.head_dim, self.head_dim)
            Q, _ = torch.linalg.qr(M)
            R_heads[h] = Q
        self.register_parameter("R_heads", R_heads)

        # Multi-scale temporal retention factors alpha_h (e.g. 0.5 to 0.95)
        alphas = torch.linspace(0.5, 0.95, num_heads)
        self.register_buffer("alphas", alphas)

        # 3. Readout Projection: (embed_dim, vocab_size)
        W_out = torch.zeros(embed_dim, vocab_size)
        self.register_parameter("W_out", W_out)

    @property
    def embeddings(self) -> torch.Tensor:
        return self._parameters["embeddings"]

    @property
    def R_heads(self) -> torch.Tensor:
        return self._parameters["R_heads"]

    @property
    def alphas(self) -> torch.Tensor:
        return self._buffers["alphas"]

    @property
    def W_out(self) -> torch.Tensor:
        return self._parameters["W_out"]

    @classmethod
    def from_text(
        cls,
        text_corpus: str,
        embed_dim: int = 128,
        num_heads: int = 4,
        reg_lambda: float = 1e-3
    ) -> 'MicroLLM':
        """Constructs and initializes vocabulary from text corpus."""
        chars = sorted(list(set(text_corpus)))
        vocab_size = len(chars)
        model = cls(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, reg_lambda=reg_lambda)
        model.char2idx = {ch: i for i, ch in enumerate(chars)}
        model.idx2char = {i: ch for i, ch in enumerate(chars)}
        return model

    def encode_text(self, text: str) -> torch.Tensor:
        """Converts string into token tensor."""
        indices = [self.char2idx.get(ch, 0) for ch in text]
        return torch.tensor(indices, dtype=torch.long)

    def decode_tokens(self, tokens: torch.Tensor) -> str:
        """Converts token tensor back to string."""
        return "".join([self.idx2char.get(int(idx.item()), "") for idx in tokens])

    def track_context(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Autoregressively updates context states across time without BPTT.
        Args:
            token_ids: (B, T) LongTensor
        Returns:
            context_states: (B, T, embed_dim)
        """
        B, T = token_ids.shape
        device = token_ids.device
        embeds = self.embeddings[token_ids]  # (B, T, embed_dim)

        # Context per head: (B, num_heads, head_dim)
        c_h = torch.zeros(B, self.num_heads, self.head_dim, device=device)
        context_states = torch.zeros(B, T, self.embed_dim, device=device)

        for t in range(T):
            e_t = embeds[:, t, :].view(B, self.num_heads, self.head_dim)
            # Parallel multi-head unitary evolution
            for h in range(self.num_heads):
                alpha = self.alphas[h]
                R = self.R_heads[h]
                # Unitary rotation preserves head energy: ||c @ R|| == ||c||
                c_h[:, h, :] = alpha * (c_h[:, h, :] @ R) + (1.0 - alpha) * e_t[:, h, :]

            # Concatenate heads into unified context representation
            context_states[:, t, :] = c_h.view(B, self.embed_dim)

        return context_states

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes next-token prediction logits for sequence: (B, T, vocab_size).
        """
        contexts = self.track_context(token_ids)
        logits = contexts @ self.W_out
        return logits

    def train_corpus(
        self,
        text: str,
        seq_len: int = 32,
        batch_size: int = 128
    ) -> Dict[str, float]:
        """
        Trains language model on text corpus in closed form in a single forward pass.
        Zero backprop, zero gradients, zero GPUs.
        """
        t0 = time.perf_counter()
        token_ids = self.encode_text(text)
        total_tokens = len(token_ids)
        assert total_tokens > seq_len, "Corpus must be longer than seq_len"

        # Chunk text into overlapping or sequential training windows
        num_samples = (total_tokens - 1) // seq_len
        if num_samples == 0:
            num_samples = 1
            seq_len = total_tokens - 1

        chunks = []
        for i in range(num_samples):
            start = i * seq_len
            end = start + seq_len + 1
            chunks.append(token_ids[start:end])

        batch = torch.stack(chunks)  # (num_samples, seq_len + 1)
        contexts = self.track_context(batch)  # (num_samples, seq_len + 1, embed_dim)

        # Context at t predicts token at t + 1
        X = contexts[:, :-1, :].reshape(-1, self.embed_dim)   # (N, embed_dim)
        Y_tokens = batch[:, 1:].reshape(-1)                   # (N,)

        # One-hot target matrix
        Y_onehot = torch.zeros(X.shape[0], self.vocab_size, device=X.device, dtype=torch.float32)
        Y_onehot.scatter_(1, Y_tokens.unsqueeze(1), 1.0)

        # Closed-form optimal Ridge solve
        new_W = AlgebraicEngine.solve_ridge(X, Y_onehot, reg_lambda=self.reg_lambda)
        self._parameters["W_out"] = new_W

        elapsed = time.perf_counter() - t0

        # Compute next-token accuracy
        with torch.no_grad():
            preds = torch.argmax(X @ new_W, dim=1)
            accuracy = (preds == Y_tokens).float().mean().item()

        return {
            "tokens_processed": len(Y_tokens),
            "training_time_s": elapsed,
            "next_token_accuracy": accuracy,
            "throughput_tokens_per_s": len(Y_tokens) / max(1e-6, elapsed)
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 5
    ) -> str:
        """
        Autoregressively generates continuation text starting from prompt.
        """
        if not prompt:
            prompt = list(self.char2idx.keys())[0]

        tokens = [self.char2idx.get(ch, 0) for ch in prompt]
        curr_tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # (1, T)

        generated = list(tokens)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(curr_tokens)  # (1, T, vocab_size)
                next_logits = logits[0, -1, :] / max(1e-4, temperature)

                # Top-k filtering
                if top_k > 0:
                    val, _ = torch.topk(next_logits, min(top_k, self.vocab_size))
                    min_val = val[-1]
                    next_logits[next_logits < min_val] = -float("Inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)

                # Append to current sequence
                curr_tokens = torch.cat(
                    [curr_tokens, torch.tensor([[next_token]], dtype=torch.long)], dim=1
                )
                # Keep sliding window if sequence gets too long
                if curr_tokens.shape[1] > 64:
                    curr_tokens = curr_tokens[:, -64:]

        return self.decode_tokens(torch.tensor(generated, dtype=torch.long))
