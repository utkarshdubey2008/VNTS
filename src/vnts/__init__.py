"""
VNTS: Vector Nullspace Transformation System
A fundamentally new neural network learning paradigm without backpropagation.
"""

from .model import Model, Sequential
from .base import VNTSModule
from .engine.solver import AlgebraicEngine
from . import nn
from . import engine

__version__ = "0.5.0"

__all__ = [
    "Model",
    "Sequential",
    "VNTSModule",
    "AlgebraicEngine",
    "nn",
    "engine",
    "__version__"
]
