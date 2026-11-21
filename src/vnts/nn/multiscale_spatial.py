"""
Multi-Scale Spatial Receptive Field Manifold for Color Vision (RGB).
Extracts fine and coarse 3D color receptive fields and pools representations
into high-dimensional visual manifolds without convolutional backpropagation.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn.functional as F
from ..base import VNTSModule
from ..engine.solver import AlgebraicEngine


class MultiScaleSpatialOSM(VNTSModule):
    """
    Multi-Scale Spatial Patch Vision Extractor for multi-channel color images (B, C, H, W).
    Extracts multi-resolution sliding patches (e.g. 3x3 and 5x5), projects into orthogonal
    receptive manifolds, and performs spatial pyramid pooling.
    """
    def __init__(
        self,
        in_channels: int = 3,
        patch_sizes: Tuple[int, ...] = (3, 5),
        strides: Tuple[int, ...] = (2, 2),
        features_per_scale: int = 64
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.features_per_scale = features_per_scale
        self.num_scales = len(patch_sizes)

        # Projection weights per scale: (C * P_s * P_s, features_per_scale)
        for s, (p_size, stride) in enumerate(zip(patch_sizes, strides)):
            patch_dim = in_channels * p_size * p_size
            proj = torch.randn(patch_dim, features_per_scale) / (patch_dim ** 0.5)
            self.register_parameter(f"proj_scale_{s}", proj)

        # Output feature dimension: for each scale, we extract [mean, max, std] = 3 * features_per_scale
        self.out_dim = self.num_scales * (features_per_scale * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Color image tensor of shape (B, C, H, W)
        Returns:
            Multi-scale visual manifold representations: (B, out_dim)
        """
        B, C, H, W = x.shape
        scale_features: List[torch.Tensor] = []

        for s, (p_size, stride) in enumerate(zip(self.patch_sizes, self.strides)):
            proj = self._parameters[f"proj_scale_{s}"]  # (patch_dim, features_per_scale)

            # Unfold patches: (B, C * p_size * p_size, num_patches)
            patches = F.unfold(x, kernel_size=p_size, stride=stride)
            patches = patches.transpose(1, 2)  # (B, num_patches, patch_dim)

            # Manifold projection with piecewise non-linearity
            projected = patches @ proj  # (B, num_patches, features_per_scale)
            projected = F.leaky_relu(projected, negative_slope=0.05)

            # Spatial pooling across patches
            mean_pool = torch.mean(projected, dim=1)  # (B, features_per_scale)
            max_pool, _ = torch.max(projected, dim=1)  # (B, features_per_scale)
            std_pool = torch.std(projected, dim=1, unbiased=False) + 1e-6  # (B, features_per_scale)

            scale_rep = torch.cat([mean_pool, max_pool, std_pool], dim=1)
            scale_features.append(scale_rep)

        out = torch.cat(scale_features, dim=1)  # (B, out_dim)
        return out
