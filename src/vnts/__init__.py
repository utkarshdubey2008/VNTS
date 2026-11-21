"""
VNTS: Vector Nullspace Transformation System
A fundamentally new neural network learning paradigm without backpropagation.
"""

from .model import Model, Sequential
from .base import VNTSModule
from .engine.solver import AlgebraicEngine
from .models.micro_llm import MicroLLM
from .models.bpe_llm import ByteLLM
from .nn.multiscale_spatial import MultiScaleSpatialOSM
from . import nn
from . import engine
from . import models
from . import native

__version__ = "0.7.0"

__all__ = [
    "Model",
    "Sequential",
    "VNTSModule",
    "AlgebraicEngine",
    "MicroLLM",
    "ByteLLM",
    "MultiScaleSpatialOSM",
    "nn",
    "engine",
    "models",
    "native",
    "__version__"
]
