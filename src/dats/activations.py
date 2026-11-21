"""
Activation functions with exact analytical pre-image (inverse) mappings for DATS-SI.
"""

import torch


class InvertibleActivation:
    """Base class for invertible activation functions."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class InvertibleLeakyReLU(InvertibleActivation):
    """
    LeakyReLU activation: phi(z) = z if z >= 0 else negative_slope * z
    Exact inverse: psi(h) = h if h >= 0 else h / negative_slope
    Since negative_slope > 0, this is a strictly monotonic continuous bijection on R.
    """
    def __init__(self, negative_slope: float = 0.1):
        if negative_slope <= 0.0:
            raise ValueError("negative_slope must be strictly positive for invertibility.")
        self.negative_slope = negative_slope

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.where(z >= 0, z, self.negative_slope * z)

    def inverse(self, h: torch.Tensor) -> torch.Tensor:
        return torch.where(h >= 0, h, h / self.negative_slope)


class InvertibleLinear(InvertibleActivation):
    """Identity activation: phi(z) = z, psi(h) = h."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def inverse(self, h: torch.Tensor) -> torch.Tensor:
        return h


class InvertibleTanh(InvertibleActivation):
    """
    Hyperbolic tangent activation with clipping to avoid numerical singularity:
    phi(z) = tanh(z)
    psi(h) = atanh(clip(h, -1 + eps, 1 - eps))
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(z)

    def inverse(self, h: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(h, -1.0 + self.eps, 1.0 - self.eps)
        return torch.atanh(clipped)
