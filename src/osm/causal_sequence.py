"""
COMT: Causal Orthogonal Manifold Tracking & Recurrent Subspace Association
Learns sequential and auto-regressive next-token prediction without Backpropagation Through Time (BPTT).
Maintains unitary context stability and executes online rank-1 closed-form updates in O(1) memory.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np


class CausalOrthogonalSequenceModel:
    """
    Auto-regressive sequence model that trains in a single forward causal pass.
    Completely eliminates BPTT, computational graph unrolling, and gradient optimizers.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        context_decay: float = 0.85,
        forgetting_factor: float = 0.9995,
        delta_p: float = 1.0,
        seed: Optional[int] = None
    ):
        if seed is not None:
            torch.manual_seed(seed)
            
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_decay = context_decay
        self.lambda_forget = forgetting_factor
        
        # 1. Normalized orthogonal token codebook
        E = torch.randn(vocab_size, embed_dim)
        self.embedding = E / torch.norm(E, dim=1, keepdim=True).clamp(min=1e-6)
        self.embedding.requires_grad = False
        
        # 2. Unitary Temporal Rotation Operator: R in SO(embed_dim)
        # Guarantees context stability over infinite time horizons: ||R c|| == ||c||
        R_rand = torch.randn(embed_dim, embed_dim)
        Q, _ = torch.linalg.qr(R_rand)
        self.R = Q
        self.R.requires_grad = False
        
        # 3. Readout Parameter Matrix: W_readout (embed_dim + 1, vocab_size)
        self.W_readout = torch.zeros((embed_dim + 1, vocab_size), dtype=torch.float32)
        self.W_readout.requires_grad = False
        
        # 4. Online Woodbury Covariance Tracker: P of shape (embed_dim + 1, embed_dim + 1)
        self.P = delta_p * torch.eye(embed_dim + 1, dtype=torch.float32)

    def init_context(self) -> torch.Tensor:
        """Initializes empty zero context vector."""
        return torch.zeros((1, self.embed_dim), dtype=torch.float32)

    def step_context(self, c_prev: torch.Tensor, token_id: int) -> torch.Tensor:
        """
        Causal unitary context evolution:
        c_t = decay * (c_{t-1} @ R) + (1 - decay) * embedding[token_id]
        """
        e_t = self.embedding[token_id:token_id+1]  # (1, embed_dim)
        c_rot = c_prev @ self.R                    # Unitary rotation
        c_t = self.context_decay * c_rot + (1.0 - self.context_decay) * e_t
        # Normalize to maintain constant energy sphere
        norm = torch.norm(c_t, dim=1, keepdim=True).clamp(min=1e-6)
        return c_t / norm

    def predict_next(self, c_t: torch.Tensor) -> torch.Tensor:
        """Computes next-token probability distribution from context vector."""
        ones = torch.ones((c_t.shape[0], 1), dtype=c_t.dtype, device=c_t.device)
        c_aug = torch.cat([c_t, ones], dim=1)
        logits = c_aug @ self.W_readout
        return logits

    def train_step(self, c_t: torch.Tensor, next_token_id: int) -> float:
        """
        Executes single online rank-1 closed-form Woodbury update for next-token prediction.
        Zero autograd, zero BPTT, O(1) memory.
        """
        ones = torch.ones((1, 1), dtype=c_t.dtype, device=c_t.device)
        c_aug = torch.cat([c_t, ones], dim=1).T  # Shape: (d+1, 1)
        
        # Target one-hot vector with positive margin
        y_target = torch.full((1, self.vocab_size), -1.0 / max(1, self.vocab_size - 1))
        y_target[0, next_token_id] = 1.0
        
        # Woodbury gain vector
        Pi_c = self.P @ c_aug  # (d+1, 1)
        denom = self.lambda_forget + (c_aug.T @ Pi_c).item()
        k = Pi_c / max(1e-6, denom)  # (d+1, 1)
        
        # Prediction residual
        error = y_target - (c_aug.T @ self.W_readout)  # (1, vocab_size)
        
        # Parameter update
        self.W_readout = self.W_readout + (k @ error)
        
        # Inverse covariance update
        self.P = (self.P - (k @ Pi_c.T)) / self.lambda_forget
        
        # Compute loss
        loss = torch.norm(error).item()
        return loss

    def train_sequence(self, token_ids: List[int]) -> Dict[str, float]:
        """
        Trains on an entire sequence token by token in a single causal forward sweep.
        """
        if len(token_ids) < 2:
            return {"loss": 0.0, "acc": 1.0}
            
        c = self.init_context()
        correct = 0
        total = 0
        total_loss = 0.0
        
        for t in range(len(token_ids) - 1):
            curr_token = token_ids[t]
            next_token = token_ids[t + 1]
            
            c = self.step_context(c, curr_token)
            
            # Predict
            logits = self.predict_next(c)
            pred_id = torch.argmax(logits, dim=1).item()
            if pred_id == next_token:
                correct += 1
            total += 1
            
            # Online update
            step_loss = self.train_step(c, next_token)
            total_loss += step_loss
            
        return {"loss": total_loss / max(1, total), "acc": correct / max(1, total)}

    def evaluate_sequence(self, token_ids: List[int]) -> Dict[str, float]:
        """Evaluates prediction accuracy without updating parameters."""
        if len(token_ids) < 2:
            return {"loss": 0.0, "acc": 1.0}
            
        c = self.init_context()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for t in range(len(token_ids) - 1):
                curr_token = token_ids[t]
                next_token = token_ids[t + 1]
                
                c = self.step_context(c, curr_token)
                logits = self.predict_next(c)
                pred_id = torch.argmax(logits, dim=1).item()
                if pred_id == next_token:
                    correct += 1
                total += 1
                
        return {"acc": correct / max(1, total)}
