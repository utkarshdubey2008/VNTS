"""
Spatial-OSM: 2D Spatial Subspace Convolution & Patch Projection
Scales Orthogonal Subspace Memory to multi-channel color vision (CIFAR-10, ImageNet).
Provides translation equivariance and local receptive field feature extraction
completely without autograd or backpropagation.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import OrthogonalSubspaceMemory


class SpatialPatchProjector:
    """
    Extracts local 2D receptive fields (patches) from multi-channel images
    and projects them into a high-dimensional non-linear spatial feature manifold.
    """
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 4,
        stride: int = 4,
        patch_embed_dim: int = 128,
        seed: Optional[int] = None
    ):
        if seed is not None:
            torch.manual_seed(seed)
            
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.stride = stride
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_embed_dim = patch_embed_dim
        
        # Spatial projection filter (orthogonal initialization)
        W = torch.randn(self.patch_dim, patch_embed_dim) * (2.0 / self.patch_dim)**0.5
        self.W_patch = W
        self.W_patch.requires_grad = False
        self.b_patch = torch.randn(patch_embed_dim) * 0.05
        self.b_patch.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, C, H, W)
        Output: Pooled spatial representation of shape (B, patch_embed_dim * 2)
        """
        B, C, H, W = x.shape
        
        # 1. Unfold image into spatial patches: shape (B, patch_dim, num_patches)
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        patches = patches.transpose(1, 2)  # (B, num_patches, patch_dim)
        
        # 2. Non-linear patch projection:
        # (B, num_patches, patch_dim) @ (patch_dim, embed_dim) -> (B, num_patches, embed_dim)
        z = patches @ self.W_patch + self.b_patch
        h_patches = F.relu(z)
        
        # 3. Dual spatial pooling: max-pooling + average-pooling across patches
        h_max, _ = torch.max(h_patches, dim=1)  # (B, embed_dim)
        h_mean = torch.mean(h_patches, dim=1)   # (B, embed_dim)
        
        # Concatenate dual pooled features
        h_spatial = torch.cat([h_max, h_mean], dim=1)  # (B, embed_dim * 2)
        return h_spatial


class SpatialOSMNetwork:
    """
    End-to-end Spatial Vision Network with Orthogonal Subspace Memory.
    Operates on raw 2D images (B, C, H, W) with zero gradients and zero backpropagation.
    """
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 4,
        patch_embed_dim: int = 256,
        out_dim: int = 10,
        variance_threshold: float = 0.98,
        reg_lambda: float = 1e-2,
        seed: Optional[int] = None
    ):
        self.in_channels = in_channels
        self.image_size = image_size
        self.out_dim = out_dim
        
        self.projector = SpatialPatchProjector(
            in_channels=in_channels,
            patch_size=patch_size,
            stride=patch_size,
            patch_embed_dim=patch_embed_dim,
            seed=seed
        )
        
        feature_dim = patch_embed_dim * 2
        
        # Orthogonal Subspace Memory Readout
        self.memory = OrthogonalSubspaceMemory(
            in_features=feature_dim + 1,  # +1 for bias
            out_features=out_dim,
            variance_threshold=variance_threshold,
            reg_lambda=reg_lambda,
            enable_compaction=True
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.projector.forward(x)
        ones = torch.ones((h.shape[0], 1), dtype=h.dtype, device=h.device)
        return torch.cat([h, ones], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.extract_features(x)
        return self.memory.forward(phi)

    def learn_task(self, x: torch.Tensor, y: torch.Tensor, num_classes: int) -> int:
        phi = self.extract_features(x)
        N = y.shape[0]
        y_target = torch.full((N, num_classes), -1.0)
        y_target.scatter_(1, y.unsqueeze(1), 1.0)
        return self.memory.learn_task(phi, y_target)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            preds = self.forward(x)
            pred_classes = torch.argmax(preds, dim=1)
            acc = (pred_classes == y).float().mean().item()
            loss = F.cross_entropy(preds, y).item()
            return {"accuracy": acc, "loss": loss}
