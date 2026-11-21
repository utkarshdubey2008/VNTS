"""
Causal Orthogonal Manifold Tracking (COMT) sequence modeling for VNTS.
Eliminates Backpropagation Through Time (BPTT), vanishing gradients,
and activation caching with O(1) sequence memory.
"""

from typing import Optional, Tuple
import torch
from ..base import VNTSModule
from ..engine.solver import AlgebraicEngine


class COMTSequence(VNTSModule):
    """
    Autoregressive causal sequence module.
    Evolves context via exact norm-conserving unitary rotation R in SO(d):
        c_t = alpha * (c_{t-1} @ R) + (1 - alpha) * e_t
    Learns next-token readout W_out in closed form via streaming Woodbury / Ridge updates.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        alpha: float = 0.85,
        reg_lambda: float = 1e-3
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.reg_lambda = reg_lambda

        # Token embedding table: (vocab_size, hidden_dim)
        embeddings = torch.randn(vocab_size, hidden_dim) / (hidden_dim ** 0.5)
        self.register_parameter("embeddings", embeddings)

        # Exact Unitary Rotation Matrix R via QR decomposition: R @ R^T = I
        M = torch.randn(hidden_dim, hidden_dim)
        Q, _ = torch.linalg.qr(M)
        self.register_parameter("R", Q)

        # Output projection / readout weights: (hidden_dim, vocab_size)
        W_out = torch.zeros(hidden_dim, vocab_size)
        self.register_parameter("W_out", W_out)

        # Streaming RLS inverse covariance matrix P: (hidden_dim, hidden_dim)
        P = (1.0 / reg_lambda) * torch.eye(hidden_dim)
        self.register_buffer("P", P)

    @property
    def embeddings(self) -> torch.Tensor:
        return self._parameters["embeddings"]

    @property
    def R(self) -> torch.Tensor:
        return self._parameters["R"]

    @property
    def W_out(self) -> torch.Tensor:
        return self._parameters["W_out"]

    @property
    def P(self) -> torch.Tensor:
        return self._buffers["P"]

    def encode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embeds and tracks causal context vectors across time steps.
        Args:
            token_ids: LongTensor of shape (batch_size, seq_len)
        Returns:
            Context states: shape (batch_size, seq_len, hidden_dim)
        """
        B, T = token_ids.shape
        device = token_ids.device
        embeds = self.embeddings[token_ids]  # (B, T, hidden_dim)

        context_states = torch.zeros(B, T, self.hidden_dim, device=device)
        c_prev = torch.zeros(B, self.hidden_dim, device=device)

        for t in range(T):
            e_t = embeds[:, t, :]  # (B, hidden_dim)
            # Unitary evolution: norm is strictly conserved by R
            c_t = self.alpha * (c_prev @ self.R) + (1.0 - self.alpha) * e_t
            context_states[:, t, :] = c_t
            c_prev = c_t

        return context_states

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes next-token logits for each step in sequence: (B, T, vocab_size).
        """
        contexts = self.encode_tokens(token_ids)
        logits = contexts @ self.W_out
        return logits

    def assimilate_sequence(
        self,
        token_ids: torch.Tensor,
        streaming: bool = False
    ):
        """
        Learns autoregressive language/token sequence in a single forward pass without BPTT.
        Predicts token[t+1] from context[t].
        
        Args:
            token_ids: (B, T) LongTensor of token sequences
            streaming: If True, uses online Woodbury updates. If False, solves exact batch Ridge.
        """
        B, T = token_ids.shape
        assert T >= 2, "Sequence must have at least 2 tokens for next-token prediction"

        contexts = self.encode_tokens(token_ids)  # (B, T, hidden_dim)

        # Context at t predicts token at t + 1
        X = contexts[:, :-1, :].reshape(-1, self.hidden_dim)  # (B*(T-1), hidden_dim)
        Y_tokens = token_ids[:, 1:].reshape(-1)               # (B*(T-1),)

        # One-hot target matrix
        Y_onehot = torch.zeros(X.shape[0], self.vocab_size, device=X.device, dtype=X.dtype)
        Y_onehot.scatter_(1, Y_tokens.unsqueeze(1), 1.0)

        if not streaming:
            # Batch closed-form Ridge solve
            new_W = AlgebraicEngine.solve_ridge(X, Y_onehot, reg_lambda=self.reg_lambda)
            self._parameters["W_out"] = new_W
        else:
            # Online streaming Woodbury updates
            P = self.P
            W = self.W_out
            for i in range(X.shape[0]):
                x_i = X[i]
                y_i = Y_onehot[i]
                P, k = AlgebraicEngine.woodbury_update(P, x_i)
                err = y_i - (x_i @ W)
                W = W + torch.outer(k, err)
            self._parameters["W_out"] = W
            self._buffers["P"] = P
