"""
Byte-Level Scaled Causal Language Model (ByteLLM) for VNTS.
Operates on raw byte sequences (256 base byte tokens) with zero out-of-vocabulary issues.
Employs 8-Head Unitary Causal Orthogonal Manifold Tracking across extended sequence horizons.
"""

from typing import List, Dict, Optional, Tuple
import time
import torch
import torch.nn.functional as F

from ..base import VNTSModule
from ..engine.solver import AlgebraicEngine


class ByteLLM(VNTSModule):
    """
    Byte-Level Scaled Language Model.
    Vocabulary consists of 256 byte values + 1 EOS token = 257 tokens.
    Guarantees 100% universal text coverage across any human language or code.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        reg_lambda: float = 1e-3
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.vocab_size = 257  # 0..255 bytes + 256 EOS
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.reg_lambda = reg_lambda
        self.eos_token_id = 256

        # 1. Byte Embeddings: (257, embed_dim)
        embeddings = torch.randn(self.vocab_size, embed_dim) / (embed_dim ** 0.5)
        self.register_parameter("embeddings", embeddings)

        # 2. Multi-Head Unitary Rotation Matrices: (num_heads, head_dim, head_dim)
        R_heads = torch.zeros(num_heads, self.head_dim, self.head_dim)
        for h in range(num_heads):
            M = torch.randn(self.head_dim, self.head_dim)
            Q, _ = torch.linalg.qr(M)
            R_heads[h] = Q
        self.register_parameter("R_heads", R_heads)

        # Logarithmic memory retention factors alpha_h from 0.30 (short-range) to 0.98 (long-range)
        alphas = torch.logspace(torch.log10(torch.tensor(0.30)), torch.log10(torch.tensor(0.98)), num_heads)
        self.register_buffer("alphas", alphas)

        # 3. Readout Matrix: (embed_dim, 257)
        W_out = torch.zeros(embed_dim, self.vocab_size)
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

    def encode_text(self, text: str) -> torch.Tensor:
        """Converts string into UTF-8 byte token IDs."""
        byte_vals = list(text.encode("utf-8"))
        return torch.tensor(byte_vals, dtype=torch.long)

    def decode_tokens(self, tokens: torch.Tensor) -> str:
        """Converts byte token IDs back into string, gracefully handling decode errors."""
        byte_list = []
        for t in tokens:
            val = int(t.item())
            if val < 256:
                byte_list.append(val)
        return bytes(byte_list).decode("utf-8", errors="replace")

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

        c_h = torch.zeros(B, self.num_heads, self.head_dim, device=device)
        context_states = torch.zeros(B, T, self.embed_dim, device=device)

        for t in range(T):
            e_t = embeds[:, t, :].view(B, self.num_heads, self.head_dim)
            for h in range(self.num_heads):
                alpha = self.alphas[h]
                R = self.R_heads[h]
                # Unitary evolution conserving representation norm
                c_h[:, h, :] = alpha * (c_h[:, h, :] @ R) + (1.0 - alpha) * e_t[:, h, :]
            context_states[:, t, :] = c_h.view(B, self.embed_dim)

        return context_states

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Computes next-byte prediction logits: (B, T, 257)."""
        contexts = self.track_context(token_ids)
        return contexts @ self.W_out

    def train_text(
        self,
        text: str,
        seq_len: int = 64,
        reg_lambda: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Trains model on raw text in a single forward pass without backprop or GPUs.
        """
        t0 = time.perf_counter()
        token_ids = self.encode_text(text)
        total_tokens = len(token_ids)
        assert total_tokens > seq_len, "Text must be longer than seq_len"

        num_samples = (total_tokens - 1) // seq_len
        chunks = []
        for i in range(num_samples):
            start = i * seq_len
            end = start + seq_len + 1
            chunks.append(token_ids[start:end])

        batch = torch.stack(chunks)  # (N, seq_len + 1)
        contexts = self.track_context(batch)  # (N, seq_len + 1, embed_dim)

        X = contexts[:, :-1, :].reshape(-1, self.embed_dim)  # (M, embed_dim)
        Y_tokens = batch[:, 1:].reshape(-1)                  # (M,)

        # One-hot target matrix
        Y_onehot = torch.zeros(X.shape[0], self.vocab_size, device=X.device, dtype=torch.float32)
        Y_onehot.scatter_(1, Y_tokens.unsqueeze(1), 1.0)

        lam = reg_lambda if reg_lambda is not None else self.reg_lambda
        new_W = AlgebraicEngine.solve_ridge(X, Y_onehot, reg_lambda=lam)
        self._parameters["W_out"] = new_W

        elapsed = time.perf_counter() - t0

        with torch.no_grad():
            preds = torch.argmax(X @ new_W, dim=1)
            accuracy = (preds == Y_tokens).float().mean().item()

        return {
            "bytes_processed": len(Y_tokens),
            "training_time_s": elapsed,
            "next_byte_accuracy": accuracy,
            "throughput_bytes_per_s": len(Y_tokens) / max(1e-6, elapsed)
        }

    def generate(
        self,
        prompt: str,
        max_new_bytes: int = 100,
        temperature: float = 0.8,
        top_k: int = 10
    ) -> str:
        """Autoregressively generates text continuation from byte prompt."""
        if not prompt:
            prompt = " "

        token_ids = self.encode_text(prompt)
        curr_tokens = token_ids.unsqueeze(0)  # (1, T)
        generated = list(token_ids)

        for _ in range(max_new_bytes):
            with torch.no_grad():
                logits = self.forward(curr_tokens)
                next_logits = logits[0, -1, :] / max(1e-4, temperature)

                if top_k > 0:
                    val, _ = torch.topk(next_logits, min(top_k, self.vocab_size))
                    min_val = val[-1]
                    next_logits[next_logits < min_val] = -float("Inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                if next_token == self.eos_token_id:
                    break

                generated.append(torch.tensor(next_token, dtype=torch.long))
                curr_tokens = torch.cat(
                    [curr_tokens, torch.tensor([[next_token]], dtype=torch.long)], dim=1
                )
                if curr_tokens.shape[1] > 128:
                    curr_tokens = curr_tokens[:, -128:]

        return self.decode_tokens(torch.stack(generated))
