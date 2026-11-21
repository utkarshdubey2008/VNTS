"""
High-level VNTS Model and Sequential Pipeline.
Provides simple, intuitive API for zero-autograd closed-form AI training and inference.
"""

from typing import List, Dict, Any, Optional, Union
import os
import time
import torch
from .base import VNTSModule
from .nn.linear import Dense
from .nn.memory import OrthogonalMemory
from .nn.sequence import COMTSequence
from .nn.spatial import SpatialPatch
from .nn.activations import InvertibleLeakyReLU, Softmax


class Sequential(VNTSModule):
    """
    Sequential container of VNTS modules.
    Executes layers in sequence.
    """
    def __init__(self, *layers: VNTSModule):
        super().__init__()
        self.layers: List[VNTSModule] = list(layers)
        for i, layer in enumerate(self.layers):
            self.register_module(f"layer_{i}", layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Model(VNTSModule):
    """
    High-level VNTS Unified Model.
    Supports classification, regression, continual lifelong learning, and sequence tracking.
    """
    def __init__(
        self,
        architecture: Optional[List[VNTSModule]] = None,
        task_type: str = "classification"
    ):
        super().__init__()
        self.task_type = task_type
        if architecture is not None:
            self.pipeline = Sequential(*architecture)
            self.register_module("pipeline", self.pipeline)
        else:
            self.pipeline = None

    @classmethod
    def create_classifier(
        cls,
        in_features: int,
        hidden_dims: List[int],
        num_classes: int,
        use_orthogonal_memory: bool = True
    ) -> 'Model':
        """
        Factory to create a production-grade classification model.
        Uses Dense projection layers with terminal Orthogonal Memory for zero-forgetting.
        """
        layers: List[VNTSModule] = []
        curr_dim = in_features

        for h_dim in hidden_dims:
            layers.append(Dense(curr_dim, h_dim, activation=InvertibleLeakyReLU(0.05)))
            curr_dim = h_dim

        if use_orthogonal_memory:
            layers.append(OrthogonalMemory(curr_dim, num_classes))
        else:
            layers.append(Dense(curr_dim, num_classes, bias=True))

        model = cls(layers, task_type="classification")
        return model

    @classmethod
    def create_vision_classifier(
        cls,
        in_channels: int,
        num_classes: int,
        patch_size: int = 4,
        stride: int = 2,
        patch_features: int = 64,
        hidden_dims: Optional[List[int]] = None
    ) -> 'Model':
        """
        Factory to create a spatial vision classification model.
        """
        layers: List[VNTSModule] = []
        spatial = SpatialPatch(in_channels=in_channels, patch_size=patch_size, stride=stride, out_features=patch_features)
        layers.append(spatial)

        curr_dim = patch_features * 2
        hidden_dims = hidden_dims or [128]
        for h_dim in hidden_dims:
            layers.append(Dense(curr_dim, h_dim, activation=InvertibleLeakyReLU(0.05)))
            curr_dim = h_dim

        layers.append(OrthogonalMemory(curr_dim, num_classes))
        return cls(layers, task_type="classification")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pipeline is None:
            raise RuntimeError("Model pipeline has not been configured.")
        return self.pipeline(x)

    def assimilate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Assimilates training data (X, Y) in closed form in a single sweep without gradients.
        
        Args:
            x: Input feature tensor (N, in_features)
            y: Target tensor: class indices (N,) or continuous targets (N, out_features)
            task_id: Optional identifier for continual multi-task tracking
            
        Returns:
            Dictionary with training telemetry: execution time, loss, sample count.
        """
        t0 = time.perf_counter()
        N = x.shape[0]

        # Convert class indices to one-hot targets if classification
        if self.task_type == "classification" and y.dim() == 1:
            # Determine num_classes from last layer
            last_layer = self.pipeline.layers[-1]
            num_classes = getattr(last_layer, "out_features", int(y.max().item()) + 1)
            y_targets = torch.zeros(N, num_classes, device=x.device, dtype=torch.float32)
            y_targets.scatter_(1, y.unsqueeze(1), 1.0)
        else:
            y_targets = y.float()

        # Forward pass through intermediate layers to compute features entering the output layer
        feat = x
        for layer in self.pipeline.layers[:-1]:
            feat = layer(feat)

        # Assimilate into final layer
        last_layer = self.pipeline.layers[-1]
        if isinstance(last_layer, OrthogonalMemory):
            last_layer.assimilate(feat, y_targets)
        elif isinstance(last_layer, Dense):
            last_layer.adapt(feat, y_targets)
        else:
            raise NotImplementedError(f"Assimilation not supported for layer type {type(last_layer)}")

        elapsed = time.perf_counter() - t0
        return {
            "samples": N,
            "time_s": elapsed,
            "status": "assimilated"
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs inference. Returns predicted class indices (classification) or raw outputs (regression).
        """
        with torch.no_grad():
            logits = self.forward(x)
            if self.task_type == "classification":
                return torch.argmax(logits, dim=1)
            return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns softmax probabilities for classification.
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Evaluates model on validation/test set.
        """
        with torch.no_grad():
            preds = self.predict(x)
            if self.task_type == "classification":
                if y.dim() > 1:
                    y_indices = torch.argmax(y, dim=1)
                else:
                    y_indices = y
                accuracy = (preds == y_indices).float().mean().item()
                return {"accuracy": accuracy}
            else:
                mse = torch.mean((preds - y) ** 2).item()
                return {"mse": mse}

    def save(self, filepath: str):
        """Saves model weights and buffers to file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            "task_type": self.task_type,
            "state_dict": self.state_dict()
        }, filepath)

    def load(self, filepath: str):
        """Loads model weights and buffers from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.task_type = checkpoint.get("task_type", self.task_type)
        self.load_state_dict(checkpoint["state_dict"])
