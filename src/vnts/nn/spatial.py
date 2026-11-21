"""
Spatial Patch projection for 2D visual inputs in VNTS.
Extracts localized receptive field manifolds without convolutional backpropagation.
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from ..base import VNTSModule
from ..engine.solver import AlgebraicEngine


class SpatialPatch(VNTSModule):
    """
    2D Spatial Patch Feature Extractor.
    Extracts sliding local patches from images (B, C, H, W) and projects them
    into a structured feature manifold with spatial pooling.
    """
    def __init__(
        self,
        in_channels: int,
        patch_size: int = 4,
        stride: int = 2,
        out_features: int = 64
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.stride = stride
        self.patch_dim = in_channels * patch_size * patch_size
        self.out_features = out_features

        # Normalized random projection matrix: (patch_dim, out_features)
        P = torch.randn(self.patch_dim, out_features) / (self.patch_dim ** 0.5)
        self.register_parameter("projection", P)

    @property
    def projection(self) -> torch.Tensor:
        return self._parameters["projection"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W) or (B, H*W)
        Returns:
            Spatial pooled representations: (B, out_features * 2) [mean + max pool]
        """
        if x.dim() == 2:
            # Assume square image
            B, L = x.shape
            side = int(L ** 0.5)
            x = x.view(B, self.in_channels, side, side)

        B, C, H, W = x.shape
        # Unfold 2D patches: (B, C * patch_size * patch_size, num_patches)
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        # Transpose to (B, num_patches, patch_dim)
        patches = patches.transpose(1, 2)

        # Linear projection into manifold: (B, num_patches, out_features)
        projected = patches @ self.projection
        projected = F.leaky_relu(projected, negative_slope=0.05)

        # Dual pooling: global average and max pooling across patches
        mean_pooled = torch.mean(projected, dim=1)  # (B, out_features)
        max_pooled, _ = torch.max(projected, dim=1)  # (B, out_features)

        out = torch.cat([mean_pooled, max_pooled], dim=1)  # (B, out_features * 2)
        return out
