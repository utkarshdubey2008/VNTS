"""
Deep-OSM: Deep Hierarchical Orthogonal Subspace Memory Networks
Extends OSM-RME to deep hierarchical architectures with multi-scale readouts
and bounded spectral memory compaction.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

from .model import OrthogonalSubspaceMemory


class DeepOSMLayer:
    """
    A single hierarchical layer in Deep-OSM:
    - Forward projection with residual manifold connection.
    - Local Orthogonal Subspace Memory module.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        target_dim: int,
        variance_threshold: float = 1.0,
        reg_lambda: float = 1e-2,
        seed: Optional[int] = None
    ):
        if seed is not None:
            torch.manual_seed(seed)
            
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Projection weights (non-gradient)
        self.W_proj = torch.randn(in_dim, out_dim) * (2.0 / in_dim)**0.5
        self.b_proj = torch.randn(out_dim) * 0.05
        self.W_proj.requires_grad = False
        self.b_proj.requires_grad = False
        
        # Optional linear residual projection if dimensions differ
        if in_dim == out_dim:
            self.W_res = None
        else:
            self.W_res = torch.randn(in_dim, out_dim) * (1.0 / in_dim)**0.5
            self.W_res.requires_grad = False
            
        # Local Orthogonal Subspace Memory Readout
        self.memory = OrthogonalSubspaceMemory(
            in_features=out_dim + 1,  # +1 for bias
            out_features=target_dim,
            variance_threshold=variance_threshold,
            reg_lambda=reg_lambda
        )

    def forward_repr(self, h_prev: torch.Tensor) -> torch.Tensor:
        """Computes next-layer non-linear representation."""
        z = h_prev @ self.W_proj + self.b_proj
        h = torch.relu(z)
        if self.W_res is not None:
            h = h + 0.5 * (h_prev @ self.W_res)
        elif h_prev.shape[1] == self.out_dim:
            h = h + 0.5 * h_prev
        return h

    def augment(self, h: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((h.shape[0], 1), dtype=h.dtype, device=h.device)
        return torch.cat([h, ones], dim=1)

    def learn_task(self, h_aug: torch.Tensor, Y_residual: torch.Tensor) -> int:
        return self.memory.learn_task(h_aug, Y_residual)


class DeepOSMNetwork:
    """
    Multi-Layer Deep Orthogonal Subspace Memory Network.
    Supports arbitrary depth, hierarchical feature extraction,
    multi-scale residual memory routing, and continual learning without forgetting.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        variance_threshold: float = 1.0,
        reg_lambda: float = 1e-2,
        seed: Optional[int] = None
    ):
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.num_layers = len(hidden_dims)
        
        self.layers: List[DeepOSMLayer] = []
        prev_dim = in_dim
        
        for i, h_dim in enumerate(hidden_dims):
            layer_seed = (seed + i * 100) if seed is not None else None
            layer = DeepOSMLayer(
                in_dim=prev_dim,
                out_dim=h_dim,
                target_dim=out_dim,
                variance_threshold=variance_threshold,
                reg_lambda=reg_lambda,
                seed=layer_seed
            )
            self.layers.append(layer)
            prev_dim = h_dim

    def forward_hierarchy(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of representations at each layer [h_1, ..., h_L]."""
        representations = []
        h = x
        for layer in self.layers:
            h = layer.forward_repr(h)
            representations.append(h)
        return representations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale resonant forward inference:
        Aggregates predictions from all hierarchical memory layers.
        """
        reps = self.forward_hierarchy(x)
        # Uniform or confidence-weighted ensemble across layers
        total_logits = torch.zeros((x.shape[0], self.out_dim), dtype=torch.float32)
        for i, (layer, h) in enumerate(zip(self.layers, reps)):
            h_aug = layer.augment(h)
            layer_pred = layer.memory.forward(h_aug)
            # Weighted toward deeper layers
            weight = (i + 1) / self.num_layers
            total_logits = total_logits + weight * layer_pred
        return total_logits

    def learn_task(self, x: torch.Tensor, y: torch.Tensor, num_classes: int):
        """
        Learns a task hierarchically across all layers.
        Each layer fits the residual error left by prior layers!
        """
        N = y.shape[0]
        y_target = torch.full((N, num_classes), -1.0)
        y_target.scatter_(1, y.unsqueeze(1), 1.0)
        
        reps = self.forward_hierarchy(x)
        current_residual = y_target.clone()
        
        for i, (layer, h) in enumerate(zip(self.layers, reps)):
            h_aug = layer.augment(h)
            # Learn current residual at this layer
            layer.learn_task(h_aug, current_residual)
            # Update residual for the next layer
            layer_pred = layer.memory.forward(h_aug)
            current_residual = current_residual - 0.5 * layer_pred

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            preds = self.forward(x)
            pred_classes = torch.argmax(preds, dim=1)
            acc = (pred_classes == y).float().mean().item()
            loss = torch.nn.functional.cross_entropy(preds, y).item()
            return {"accuracy": acc, "loss": loss}
