"""
Production MoE Language Model for VNTS (VNTS-MoE-LM).
Combines Multi-Scale Orthogonal Recurrent Attention (MORA) with
Orthogonal Subspace Mixture-of-Experts (OSR-MoE) feedforward routing.
"""

from typing import List, Dict, Optional, Tuple
import math
import time
import torch
import torch.nn.functional as F

from ..base import VNTSModule
from ..nn.mora import MultiScaleOrthogonalAttention
from ..nn.moe import OrthogonalSubspaceMoE
from ..engine.solver import AlgebraicEngine


class VNTSMoELanguageModel(VNTSModule):
    """
    VNTS MoE Language Model.
    Architecture:
    Token Embeddings -> MORA Attention -> OrthogonalSubspaceMoE -> Closed-Form Readout.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_experts: int = 4,
        top_k: int = 2,
        expert_dim: int = 256,
        reg_lambda: float = 1e-3
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dim = expert_dim
        self.reg_lambda = reg_lambda

        # 1. Embeddings: (vocab_size x embed_dim)
        embeddings = torch.randn(vocab_size, embed_dim) / (embed_dim ** 0.5)
        self.register_parameter("embeddings", embeddings)

        # 2. MORA Attention
        self.attention = MultiScaleOrthogonalAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.register_module("attention", self.attention)

        # 3. MoE Feedforward Block
        self.moe = OrthogonalSubspaceMoE(
            in_features=embed_dim,
            out_features=expert_dim,
            num_experts=num_experts,
            top_k=top_k,
            subspace_dim=16,
            reg_lambda=reg_lambda
        )
        self.register_module("moe", self.moe)

        # 4. Readout Matrix: (expert_dim x vocab_size)
        W_out = torch.zeros(expert_dim, vocab_size)
        self.register_parameter("W_out", W_out)

        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}

    @property
    def embeddings(self) -> torch.Tensor:
        return self._parameters["embeddings"]

    @property
    def W_out(self) -> torch.Tensor:
        return self._parameters["W_out"]

    @classmethod
    def from_corpus(
        cls,
        corpus_text: str,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_experts: int = 4,
        top_k: int = 2,
        expert_dim: int = 256
    ) -> 'VNTSMoELanguageModel':
        chars = sorted(list(set(corpus_text)))
        vocab_size = len(chars)
        lm = cls(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_experts=num_experts,
            top_k=top_k,
            expert_dim=expert_dim
        )
        lm.char2idx = {ch: i for i, ch in enumerate(chars)}
        lm.idx2char = {i: ch for i, ch in enumerate(chars)}
        return lm

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char2idx.get(ch, 0) for ch in text], dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join([self.idx2char.get(int(idx.item()), "") for idx in tokens])

    def forward(
        self,
        token_ids: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MORA -> Residual -> MoE -> Readout.
        Returns: (logits, next_state)
        """
        B, T = token_ids.shape
        x = self.embeddings[token_ids]  # (B, T, embed_dim)

        attn_out, next_state = self.attention(x, state=state)
        h = x + attn_out  # Residual connection

        moe_out = self.moe(h)  # (B, T, expert_dim)
        logits = moe_out @ self.W_out  # (B, T, vocab_size)
        return logits, next_state

    def fit_and_evaluate(
        self,
        train_text: str,
        val_text: str,
        seq_len: int = 32
    ) -> Dict[str, float]:
        """
        Trains model on train_text in closed form, then rigorously evaluates
        out-of-sample perplexity and cross-entropy on unseen val_text.
        """
        t0 = time.perf_counter()
        train_tokens = self.encode(train_text)
        assert len(train_tokens) > seq_len, "Train text must exceed seq_len"

        num_samples = (len(train_tokens) - 1) // seq_len
        chunks = [train_tokens[i * seq_len : i * seq_len + seq_len + 1] for i in range(num_samples)]
        batch = torch.stack(chunks)  # (N, seq_len + 1)

        input_tokens = batch[:, :-1]
        target_tokens = batch[:, 1:].reshape(-1)

        # 1. Forward pass through MORA & MoE to collect expert features
        x = self.embeddings[input_tokens]
        attn_out, _ = self.attention(x)
        h = x + attn_out
        moe_feats = self.moe(h).reshape(-1, self.expert_dim)

        # 2. Closed-form readout solve
        Y_onehot = torch.zeros(moe_feats.shape[0], self.vocab_size, device=x.device, dtype=torch.float32)
        Y_onehot.scatter_(1, target_tokens.unsqueeze(1), 1.0)

        new_W = AlgebraicEngine.solve_ridge(moe_feats, Y_onehot, reg_lambda=self.reg_lambda)
        self._parameters["W_out"] = new_W

        train_time = time.perf_counter() - t0

        # In-sample metrics
        train_preds = torch.argmax(moe_feats @ new_W, dim=1)
        train_acc = (train_preds == target_tokens).float().mean().item()

        # 3. Honest out-of-sample evaluation on held-out validation set
        val_tokens = self.encode(val_text)
        val_samples = (len(val_tokens) - 1) // seq_len
        if val_samples > 0:
            val_chunks = [val_tokens[i * seq_len : i * seq_len + seq_len + 1] for i in range(val_samples)]
            val_batch = torch.stack(val_chunks)
            val_inputs = val_batch[:, :-1]
            val_targets = val_batch[:, 1:].reshape(-1)

            with torch.no_grad():
                val_logits, _ = self.forward(val_inputs)
                val_logits_flat = val_logits.reshape(-1, self.vocab_size)
                val_loss = F.cross_entropy(val_logits_flat, val_targets).item()
                val_perplexity = math.exp(min(val_loss, 20.0))  # Prevent numerical overflow
                val_acc = (torch.argmax(val_logits_flat, dim=1) == val_targets).float().mean().item()
        else:
            val_loss = 0.0
            val_perplexity = 0.0
            val_acc = 0.0

        return {
            "tokens_trained": len(target_tokens),
            "training_time_s": train_time,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
            "val_accuracy": val_acc,
            "throughput_tokens_per_s": len(target_tokens) / max(1e-6, train_time)
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 5
    ) -> str:
        """Autoregressively generates text continuation from prompt."""
        if not prompt:
            prompt = list(self.char2idx.keys())[0]

        tokens = [self.char2idx.get(ch, 0) for ch in prompt]
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
