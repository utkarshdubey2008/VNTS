"""
Deep Residual Vision Backbone for VNTS (Res-VNTS).
Employs identity residual skips and multi-scale receptive projections
to scale vision depth to 16+ layers without gradient degradation or rank contraction.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F

from ..base import VNTSModule
from ..engine.solver import AlgebraicEngine
from .multiscale_spatial import MultiScaleSpatialOSM


class ResidualBlock(VNTSModule):
    """
    Identity residual block with closed-form non-linear transformation:
    h_out = h_in + phi(h_in @ W + b)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        std = (2.0 / dim) ** 0.5
        W = torch.randn(dim, dim) * std * 0.5
        self.register_parameter("weight", W)
        b = torch.zeros(dim)
        self.register_parameter("bias", b)

    @property
    def weight(self) -> torch.Tensor:
        return self._parameters["weight"]

    @property
    def bias(self) -> torch.Tensor:
        return self._parameters["bias"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = F.leaky_relu(x @ self.weight + self.bias, negative_slope=0.05)
        return x + res


class ResVNTSVision(VNTSModule):
    """
    Deep Residual Vision Network for multi-class RGB image classification.
    MultiScaleSpatialOSM -> Linear Embedding -> N Residual Blocks -> Terminal Classifier.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        depth: int = 4,
        features_dim: int = 128,
        patch_sizes: Tuple[int, ...] = (3, 5),
        strides: Tuple[int, ...] = (2, 2)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        self.features_dim = features_dim

        # 1. Multi-scale color patch extractor
        self.spatial = MultiScaleSpatialOSM(
            in_channels=in_channels,
            patch_sizes=patch_sizes,
            strides=strides,
            features_per_scale=32
        )
        self.register_module("spatial", self.spatial)

        # 2. Embedding projection into features_dim
        embed_proj = torch.randn(self.spatial.out_dim, features_dim) / (self.spatial.out_dim ** 0.5)
        self.register_parameter("embed_proj", embed_proj)

        # 3. Stack of N Residual Blocks
        self.res_blocks: List[ResidualBlock] = []
        for i in range(depth):
            block = ResidualBlock(features_dim)
            self.res_blocks.append(block)
            self.register_module(f"res_block_{i}", block)

        # 4. Terminal Classification Readout
        W_cls = torch.zeros(features_dim, num_classes)
        self.register_parameter("W_cls", W_cls)

    @property
    def embed_proj(self) -> torch.Tensor:
        return self._parameters["embed_proj"]

    @property
    def W_cls(self) -> torch.Tensor:
        return self._parameters["W_cls"]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates through multi-scale spatial patches and residual stack."""
        sp_feats = self.spatial(x)
        h = sp_feats @ self.embed_proj
        for block in self.res_blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return feats @ self.W_cls

    def assimilate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Assimilates image batch (X, Y) in closed form in a single sweep."""
        t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        import time
        start = time.perf_counter()

        feats = self.extract_features(x)
        N = x.shape[0]

        if y.dim() == 1:
            Y_onehot = torch.zeros(N, self.num_classes, device=x.device, dtype=torch.float32)
            Y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
        else:
            Y_onehot = y.float()

        new_W = AlgebraicEngine.solve_ridge(feats, Y_onehot, reg_lambda=1e-3)
        self._parameters["W_cls"] = new_W

        elapsed = time.perf_counter() - start
        return elapsed
