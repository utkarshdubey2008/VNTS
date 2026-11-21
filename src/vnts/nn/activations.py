"""
Invertible non-linear activations for VNTS architectures.
Guarantees analytical forward and exact inverse mapping.
"""

from typing import Optional
import torch
from ..base import VNTSModule


class Activation(VNTSModule):
    """Abstract activation module."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Identity(Activation):
    """Identity pass-through activation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y


class InvertibleLeakyReLU(Activation):
    """
    Piecewise linear bijective activation:
        phi(x) = x if x >= 0 else negative_slope * x
        phi^-1(y) = y if y >= 0 else y / negative_slope
    """
    def __init__(self, negative_slope: float = 0.05):
        super().__init__()
        assert negative_slope > 0, "negative_slope must be strictly positive for bijectivity"
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, self.negative_slope * x)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.where(y >= 0, y, y / self.negative_slope)


class Softmax(VNTSModule):
    """Softmax probability layer (output only)."""
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=self.dim)
