"""
VNTS-LM: Production-Scale Autoregressive Language Model powered by MORA.
Replaces quadratic Transformer self-attention with O(1) memory orthogonal recurrent attention.
"""

from typing import List, Dict, Optional, Tuple
import time
import torch
import torch.nn.functional as F

from ..base import VNTSModule
from ..nn.mora import MultiScaleOrthogonalAttention
from ..engine.solver import AlgebraicEngine


class VNTSLanguageModel(VNTSModule):
    """
    VNTS Autoregressive Language Model.
    Architecture:
    Token Embeddings -> MultiScaleOrthogonalAttention (MORA) -> Non-Linear Gated Feedforward -> Closed-form Output Logits.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        reg_lambda: float = 1e-3
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.reg_lambda = reg_lambda

        # 1. Token Embeddings: (vocab_size, embed_dim)
        embeddings = torch.randn(vocab_size, embed_dim) / (embed_dim ** 0.5)
        self.register_parameter("embeddings", embeddings)

        # 2. MultiScaleOrthogonalAttention (MORA)
        self.attention = MultiScaleOrthogonalAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.register_module("attention", self.attention)

        # 3. Feedforward Projection: (embed_dim, ffn_dim)
        W_ffn1 = torch.randn(embed_dim, ffn_dim) / (embed_dim ** 0.5)
        self.register_parameter("W_ffn1", W_ffn1)
        b_ffn1 = torch.zeros(ffn_dim)
        self.register_parameter("b_ffn1", b_ffn1)

        # 4. Readout Matrix: (ffn_dim, vocab_size)
        W_out = torch.zeros(ffn_dim, vocab_size)
        self.register_parameter("W_out", W_out)

        # Tokenizer vocabulary lookup
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}

    @property
    def embeddings(self) -> torch.Tensor:
        return self._parameters["embeddings"]

    @property
    def W_ffn1(self) -> torch.Tensor:
        return self._parameters["W_ffn1"]

    @property
    def b_ffn1(self) -> torch.Tensor:
        return self._parameters["b_ffn1"]

    @property
    def W_out(self) -> torch.Tensor:
        return self._parameters["W_out"]

    @classmethod
    def from_corpus(
        cls,
        corpus_text: str,
        embed_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256
    ) -> 'VNTSLanguageModel':
        chars = sorted(list(set(corpus_text)))
        vocab_size = len(chars)
        lm = cls(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim)
        lm.token2idx = {ch: i for i, ch in enumerate(chars)}
        lm.idx2token = {i: ch for i, ch in enumerate(chars)}
        return lm

    def encode(self, text: str) -> torch.Tensor:
        indices = [self.token2idx.get(ch, 0) for ch in text]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join([self.idx2token.get(int(idx.item()), "") for idx in tokens])

    def forward(
        self,
        token_ids: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes next-token logits: (B, T, vocab_size).
        """
        B, T = token_ids.shape
        # Embedding lookup
        x = self.embeddings[token_ids]  # (B, T, embed_dim)

        # MORA attention with residual routing
        attn_out, next_state = self.attention(x, state=state)
        h = x + attn_out

        # Feedforward non-linear activation
        ffn_out = F.leaky_relu(h @ self.W_ffn1 + self.b_ffn1, negative_slope=0.05)

        # Logits projection
        logits = ffn_out @ self.W_out
        return logits, next_state

    def train_text(
        self,
        corpus_text: str,
        seq_len: int = 32
    ) -> Dict[str, float]:
        """
        Trains language model on text in closed form. Zero backpropagation.
        """
        t0 = time.perf_counter()
        token_ids = self.encode(corpus_text)
        total_len = len(token_ids)
        assert total_len > seq_len, "Corpus length must exceed seq_len"

        num_samples = (total_len - 1) // seq_len
        chunks = [token_ids[i * seq_len : i * seq_len + seq_len + 1] for i in range(num_samples)]
        batch = torch.stack(chunks)  # (N, seq_len + 1)

        input_tokens = batch[:, :-1]
        target_tokens = batch[:, 1:].reshape(-1)

        # Forward pass to extract FFN features
        x = self.embeddings[input_tokens]
        attn_out, _ = self.attention(x)
        h = x + attn_out
        ffn_feats = F.leaky_relu(h @ self.W_ffn1 + self.b_ffn1, negative_slope=0.05)
        X_flat = ffn_feats.reshape(-1, self.ffn_dim)

        # Target one-hot matrix
        Y_onehot = torch.zeros(X_flat.shape[0], self.vocab_size, device=X_flat.device, dtype=torch.float32)
        Y_onehot.scatter_(1, target_tokens.unsqueeze(1), 1.0)

        # Solve readout in closed form
        new_W = AlgebraicEngine.solve_ridge(X_flat, Y_onehot, reg_lambda=self.reg_lambda)
        self._parameters["W_out"] = new_W

        elapsed = time.perf_counter() - t0
        with torch.no_grad():
            preds = torch.argmax(X_flat @ new_W, dim=1)
            accuracy = (preds == target_tokens).float().mean().item()

        return {
            "tokens_processed": len(target_tokens),
            "training_time_s": elapsed,
            "next_token_accuracy": accuracy,
            "throughput_tokens_per_s": len(target_tokens) / max(1e-6, elapsed)
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 80,
        temperature: float = 0.8,
        top_k: int = 5
    ) -> str:
        """Autoregressively generates text continuation from prompt."""
        if not prompt:
            prompt = list(self.token2idx.keys())[0]

        tokens = [self.token2idx.get(ch, 0) for ch in prompt]
        curr_tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        generated = list(tokens)

        for _ in range(max_tokens):
            with torch.no_grad():
                logits, _ = self.forward(curr_tokens)
                next_logits = logits[0, -1, :] / max(1e-4, temperature)

                if top_k > 0:
                    val, _ = torch.topk(next_logits, min(top_k, self.vocab_size))
                    next_logits[next_logits < val[-1]] = -float("Inf")

                probs = F.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_tok)

                curr_tokens = torch.cat(
                    [curr_tokens, torch.tensor([[next_tok]], dtype=torch.long)], dim=1
                )
                if curr_tokens.shape[1] > 64:
                    curr_tokens = curr_tokens[:, -64:]

        return self.decode(torch.tensor(generated, dtype=torch.long))
