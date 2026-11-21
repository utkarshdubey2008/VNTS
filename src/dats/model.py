"""
DATSNetwork: Multi-layer Neural Network trained via Dual-Stream Algebraic Target Synthesis.
Zero gradients, zero autograd, zero backward graph.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn

from .activations import InvertibleActivation, InvertibleLeakyReLU, InvertibleLinear
from .algebraic_solver import regularized_pseudoinverse, solve_ridge_subspace


class DATSNetwork:
    """
    Feedforward Neural Network executing the Dual-Stream Algebraic Target Synthesis paradigm.
    All parameters are explicitly non-gradient tensors (requires_grad=False).
    """
    def __init__(
        self,
        layer_dims: List[int],
        activation: Optional[InvertibleActivation] = None,
        output_activation: Optional[InvertibleActivation] = None,
        seed: Optional[int] = None
    ):
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must have at least 2 dimensions [input_dim, output_dim].")
            
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        
        if seed is not None:
            torch.manual_seed(seed)
            
        self.activations: List[InvertibleActivation] = []
        default_act = activation if activation is not None else InvertibleLeakyReLU(negative_slope=0.1)
        out_act = output_activation if output_activation is not None else InvertibleLinear()
        
        for l in range(self.num_layers - 1):
            self.activations.append(default_act)
        self.activations.append(out_act)
        
        # Initialize weights and biases using normalized orthogonal / He initialization
        self.weights: List[torch.Tensor] = []
        self.biases: List[torch.Tensor] = []
        
        for l in range(self.num_layers):
            d_in = layer_dims[l]
            d_out = layer_dims[l + 1]
            
            # Orthogonal initialization scaled by gain
            gain = 1.0 if l == self.num_layers - 1 else 1.414
            W = torch.empty((d_in, d_out), dtype=torch.float32)
            nn.init.orthogonal_(W, gain=gain)
            W.requires_grad = False
            
            b = torch.zeros((d_out,), dtype=torch.float32)
            b.requires_grad = False
            
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward inference."""
        h = x
        for l in range(self.num_layers):
            z = h @ self.weights[l] + self.biases[l]
            h = self.activations[l].forward(z)
        return h

    def forward_intermediates(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Computes forward pass while recording all intermediate (Z_l, H_l).
        Returns:
            zs: list of pre-activations [Z_1, ..., Z_L]
            hs: list of activations [H_0, H_1, ..., H_L], where H_0 = x
        """
        zs = []
        hs = [x]
        h = x
        for l in range(self.num_layers):
            z = h @ self.weights[l] + self.biases[l]
            h = self.activations[l].forward(z)
            zs.append(z)
            hs.append(h)
        return zs, hs

    def synthesize_targets(
        self,
        zs: List[torch.Tensor],
        hs: List[torch.Tensor],
        y_target: torch.Tensor,
        beta: float = 0.5,
        gamma: float = 1e-4,
        normalize_targets: bool = True
    ) -> List[torch.Tensor]:
        """
        Backward Algebraic Target Synthesis Stream.
        Synthesizes target pre-activations Z_tilde_l for all layers l = 1...L.
        
        Args:
            zs: list of forward pre-activations [Z_1, ..., Z_L]
            hs: list of forward activations [H_0, H_1, ..., H_L]
            y_target: Target output values/logits (N, d_L)
            beta: Target relaxation rate in (0, 1]
            gamma: Regularizer for pseudoinverse
            normalize_targets: If True, standardizes synthesized targets to match forward activation scale,
                               preventing vanishing/exploding targets across deep layers.
            
        Returns:
            z_targets: list of target pre-activations [Z_tilde_1, ..., Z_tilde_L]
        """
        L = self.num_layers
        z_targets: List[Optional[torch.Tensor]] = [None] * L
        
        # 1. Output layer boundary condition:
        # Preimage of target through output activation
        z_targets[L - 1] = self.activations[L - 1].inverse(y_target)
        
        # 2. Algebraic backward target inversion for hidden layers
        for l in range(L - 2, -1, -1):
            next_z_target = z_targets[l + 1]
            W_next = self.weights[l + 1]
            b_next = self.biases[l + 1]
            
            # Center target by subtracting next layer bias
            Z_centered = next_z_target - b_next.unsqueeze(0)
            
            # Pseudoinverse projection: Z_centered is (N, d_next), W_pinv is (d_next, d_curr)
            W_pinv = regularized_pseudoinverse(W_next, gamma=gamma)
            T_l_raw = Z_centered @ W_pinv
            
            H_curr = hs[l + 1]
            if normalize_targets:
                # Subspace scale matching: prevents norm explosion/decay across deep layers
                std_h = H_curr.std(dim=0, keepdim=True).clamp(min=1e-3)
                mean_h = H_curr.mean(dim=0, keepdim=True)
                std_t = T_l_raw.std(dim=0, keepdim=True).clamp(min=1e-3)
                mean_t = T_l_raw.mean(dim=0, keepdim=True)
                T_l_aligned = (T_l_raw - mean_t) / std_t * std_h + mean_h
            else:
                T_l_aligned = T_l_raw
            
            # Residual target contraction: blends current state with ideal inverted target
            T_l = (1.0 - beta) * H_curr + beta * T_l_aligned
            
            # Bijective pre-activation preimage
            z_targets[l] = self.activations[l].inverse(T_l)
            
        return z_targets

    def update_subspaces(
        self,
        hs: List[torch.Tensor],
        z_targets: List[torch.Tensor],
        reg_lambda: float = 1e-4
    ):
        """
        Computes closed-form Tikhonov/Ridge subspace projection at each layer.
        Replaces gradient updates with exact analytic solutions.
        """
        for l in range(self.num_layers):
            H_prev = hs[l]
            N = H_prev.shape[0]
            
            # Construct augmented matrix [H_prev, 1]
            ones = torch.ones((N, 1), dtype=H_prev.dtype, device=H_prev.device)
            H_aug = torch.cat([H_prev, ones], dim=1)
            
            # Target pre-activations for this layer
            Z_target = z_targets[l]
            
            # Solve normal equations in closed form
            W_aug_opt = solve_ridge_subspace(H_aug, Z_target, reg_lambda=reg_lambda)
            
            # Extract weights and biases
            self.weights[l] = W_aug_opt[:-1, :].clone()
            self.biases[l] = W_aug_opt[-1, :].clone()
