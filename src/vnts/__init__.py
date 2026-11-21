"""
VNTS: Vector Nullspace Transformation System
A fundamentally new neural network learning paradigm without backpropagation.
"""

from .model import Model, Sequential
from .base import VNTSModule
from .engine.solver import AlgebraicEngine
from .models.micro_llm import MicroLLM
from . import nn
from . import engine
from . import models

__version__ = "0.6.0"

__all__ = [
    "Model",
    "Sequential",
    "VNTSModule",
    "AlgebraicEngine",
    "MicroLLM",
    "nn",
    "engine",
    "models",
    "__version__"
]
