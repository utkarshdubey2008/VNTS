"""
Dense algebraic feedforward layer for VNTS.
Supports closed-form Ridge parameter updates and exact target inversion.
"""

from typing import Optional, Union
import torch
from ..base import VNTSModule
from ..engine.solver import AlgebraicEngine
from .activations import Activation, InvertibleLeakyReLU, Identity


class Dense(VNTSModule):
    """
    Algebraic Dense Layer.
    Computes Y = activation(X @ W + b) with zero autograd.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[Activation] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.activation = activation if activation is not None else Identity()

        # He / Kaiming normal initialization scaled for linear algebra stability
        std = (2.0 / in_features) ** 0.5
        W = torch.randn(in_features, out_features) * std
        self.register_parameter("weight", W)

        if bias:
            b = torch.zeros(out_features)
            self.register_parameter("bias", b)
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        return self._parameters["weight"]

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self._parameters.get("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bias and self.bias is not None:
            pre_act = x @ self.weight + self.bias
        else:
            pre_act = x @ self.weight
        return self.activation(pre_act)

    def invert_target(self, target_output: torch.Tensor) -> torch.Tensor:
        """
        Computes exact preimage of output targets:
        z = activation.inverse(target_output) - bias
        target_input = z @ pinv(W)
        """
        # Step 1: Invert non-linearity
        z = self.activation.inverse(target_output)
        if self.use_bias and self.bias is not None:
            z = z - self.bias

        # Step 2: Linear pseudo-inverse
        # W has shape (in_features, out_features)
        # pinv(W) has shape (out_features, in_features)
        pinv_W = torch.linalg.pinv(self.weight)
        target_input = z @ pinv_W
        return target_input

    def adapt(
        self,
        x: torch.Tensor,
        target_output: torch.Tensor,
        reg_lambda: float = 1e-3,
        damping: float = 1.0
    ):
        """
        Closed-form optimal algebraic update for W and b:
        Solves: argmin ||(X W + b) - z||^2 + lambda ||W||^2
        """
        # Invert activation to get pre-activation targets
        z = self.activation.inverse(target_output)

        if self.use_bias:
            # Prepend bias column of 1s
            ones = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
            X_aug = torch.cat([x, ones], dim=1)
            W_aug = AlgebraicEngine.solve_ridge(X_aug, z, reg_lambda=reg_lambda)
            new_W = W_aug[:-1, :]
            new_b = W_aug[-1, :]

            # Apply damped update
            self._parameters["weight"] = (1.0 - damping) * self.weight + damping * new_W
            self._parameters["bias"] = (1.0 - damping) * self.bias + damping * new_b
        else:
            new_W = AlgebraicEngine.solve_ridge(x, z, reg_lambda=reg_lambda)
            self._parameters["weight"] = (1.0 - damping) * self.weight + damping * new_W
