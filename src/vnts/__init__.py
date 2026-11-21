"""
VNTS: Vector Nullspace Transformation System
A fundamentally new neural network learning paradigm without backpropagation.
"""

from .model import Model, Sequential
from .base import VNTSModule
from .engine.solver import AlgebraicEngine
from .models.micro_llm import MicroLLM
from .models.bpe_llm import ByteLLM
from .models.vnts_lm import VNTSLanguageModel
from .models.moe_lm import VNTSMoELanguageModel
from .nn.multiscale_spatial import MultiScaleSpatialOSM
from .nn.mora import MultiScaleOrthogonalAttention
from .nn.time_series import ContinuousStateSpaceTracker
from .nn.residual_vision import ResVNTSVision
from .nn.moe import OrthogonalSubspaceMoE
from . import nn
from . import engine
from . import models
from . import native

__version__ = "0.9.0"

__all__ = [
    "Model",
    "Sequential",
    "VNTSModule",
    "AlgebraicEngine",
    "MicroLLM",
    "ByteLLM",
    "VNTSLanguageModel",
    "VNTSMoELanguageModel",
    "MultiScaleSpatialOSM",
    "MultiScaleOrthogonalAttention",
    "ContinuousStateSpaceTracker",
    "ResVNTSVision",
    "OrthogonalSubspaceMoE",
    "nn",
    "engine",
    "models",
    "native",
    "__version__"
]
