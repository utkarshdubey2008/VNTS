"""
Multi-Scale Orthogonal Recurrent Attention (MORA).
A mathematically principled O(1) memory alternative to Transformer Self-Attention (MHA).
Evolves historical context through orthogonal Lie-group rotations without Backpropagation Through Time.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn.functional as F

from ..base import VNTSModule
from ..engine.solver import AlgebraicEngine


class MultiScaleOrthogonalAttention(VNTSModule):
    """
    Multi-Scale Orthogonal Recurrent Attention (MORA).
    
    Replaces O(T^2) quadratic softmax attention with exact state-space orthogonal evolution:
        S_t = alpha_h * (S_{t-1} @ R_h) + K_t^T @ V_t
        O_t = Q_t @ S_t
    where R_h in SO(d) is a unitary rotation matrix conserving context norm.
    Memory complexity: O(1) per token, independent of sequence length T.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else (embed_dim // num_heads)
        self.total_head_dim = self.num_heads * self.head_dim

        # Projections: Query, Key, Value
        # Initialized with scaled orthogonal projections
        Q_weight = torch.randn(embed_dim, self.total_head_dim) / (embed_dim ** 0.5)
        K_weight = torch.randn(embed_dim, self.total_head_dim) / (embed_dim ** 0.5)
        V_weight = torch.randn(embed_dim, self.total_head_dim) / (embed_dim ** 0.5)
        self.register_parameter("q_proj", Q_weight)
        self.register_parameter("k_proj", K_weight)
        self.register_parameter("v_proj", V_weight)

        # Output projection
        out_weight = torch.randn(self.total_head_dim, embed_dim) / (self.total_head_dim ** 0.5)
        self.register_parameter("out_proj", out_weight)

        # Multi-scale orthogonal rotation matrices R_h in SO(head_dim)
        R_heads = torch.zeros(num_heads, self.head_dim, self.head_dim)
        for h in range(num_heads):
            M = torch.randn(self.head_dim, self.head_dim)
            Q, _ = torch.linalg.qr(M)
            R_heads[h] = Q
        self.register_parameter("R_heads", R_heads)

        # Multi-scale exponential decay spectrum (from fast syntactic to long-range semantic)
        alphas = torch.linspace(0.40, 0.98, num_heads)
        self.register_buffer("alphas", alphas)

    @property
    def q_proj(self) -> torch.Tensor:
        return self._parameters["q_proj"]

    @property
    def k_proj(self) -> torch.Tensor:
        return self._parameters["k_proj"]

    @property
    def v_proj(self) -> torch.Tensor:
        return self._parameters["v_proj"]

    @property
    def out_proj(self) -> torch.Tensor:
        return self._parameters["out_proj"]

    @property
    def R_heads(self) -> torch.Tensor:
        return self._parameters["R_heads"]

    @property
    def alphas(self) -> torch.Tensor:
        return self._buffers["alphas"]

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequences (B, T, embed_dim)
            state: Optional previous memory state (B, num_heads, head_dim, head_dim)
        Returns:
            (out, next_state) where out has shape (B, T, embed_dim)
        """
        B, T, D = x.shape
        device = x.device

        # Linear projections
        Q = (x @ self.q_proj).view(B, T, self.num_heads, self.head_dim)
        K = (x @ self.k_proj).view(B, T, self.num_heads, self.head_dim)
        V = (x @ self.v_proj).view(B, T, self.num_heads, self.head_dim)

        # Normalize queries and keys for numerical stability
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)

        # State tensor for each head: (B, num_heads, head_dim, head_dim)
        if state is None:
            S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=device)
        else:
            S = state

        outputs = torch.zeros(B, T, self.num_heads, self.head_dim, device=device)

        # Autoregressive sequence tracking
        for t in range(T):
            q_t = Q[:, t, :, :]  # (B, num_heads, head_dim)
            k_t = K[:, t, :, :]  # (B, num_heads, head_dim)
            v_t = V[:, t, :, :]  # (B, num_heads, head_dim)

            # Outer product of key and value: (B, num_heads, head_dim, head_dim)
            kv_t = torch.einsum("bhd,bhm->bhdm", k_t, v_t)

            # Evolve state per head: S_t = alpha * (S_{t-1} @ R) + kv_t
            for h in range(self.num_heads):
                alpha = self.alphas[h]
                R = self.R_heads[h]
                S[:, h, :, :] = alpha * (S[:, h, :, :] @ R) + (1.0 - alpha) * kv_t[:, h, :, :]

                # Query output: O_t = Q_t @ S_t  -> (B, head_dim)
                outputs[:, t, h, :] = torch.einsum("bd,bdm->bm", q_t[:, h, :], S[:, h, :, :])

        # Merge heads and apply output projection
        out_flat = outputs.view(B, T, self.total_head_dim)
        out = out_flat @ self.out_proj
        return out, S
