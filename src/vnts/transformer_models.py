"""
VNTS: Causal Transformer Architectures
Supports:
1. Standard Causal Transformer (End-to-End Backpropagation)
2. Decoupled Forward Transformer (DFT / FRP-Transformer) with layer-wise local token prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device), diagonal=1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class StandardTransformerLM(nn.Module):
    """Monolithic Causal Transformer trained with standard End-to-End Backpropagation."""
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, S = idx.shape
        pos = torch.arange(0, S, device=idx.device).unsqueeze(0)
        h = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits


class DecoupledForwardTransformerLM(nn.Module):
    """
    Decoupled Forward Transformer (DFT):
    Each block updates immediately using a local token prediction head.
    Zero depth-wise autograd graph across transformer layers.
    """
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # Local prediction heads for each block
        self.local_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False) for _ in range(n_layers)
        ])
        self.local_lns = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Final head
        self.ln_f = nn.LayerNorm(d_model)
        self.final_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        """Inference forward pass."""
        B, S = idx.shape
        pos = torch.arange(0, S, device=idx.device).unsqueeze(0)
        h = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.final_head(h)
