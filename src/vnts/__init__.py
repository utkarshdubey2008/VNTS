"""
VNTS: Vector Nullspace Transformation System
A fundamentally new neural network learning paradigm without backpropagation.
"""

# ── CPU-usage guard ──────────────────────────────────────────────────────────
# Applied at module import time so VNTS never spawns runaway BLAS thread pools
# on shared / commodity CPUs.  Users who want more parallelism can override
# via the VNTS_NUM_THREADS environment variable.
import os as _os

_num_threads = int(_os.environ.get("VNTS_NUM_THREADS", "1"))
for _var in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_var, str(_num_threads))

try:
    import torch as _torch
    _torch.set_num_threads(_num_threads)
    try:
        _torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already started — thread cap still applies to intra-op threads
except ImportError:
    pass
# ────────────────────────────────────────────────────────────────────────────

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
from .nn.orma import ORMALayer, ORMANetwork
from .apps.doc_engine import DocumentIntelligenceEngine
from . import nn
from . import engine
from . import models
from . import native
from . import apps

__version__ = "1.0.0"

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
    "ORMALayer",
    "ORMANetwork",
    "DocumentIntelligenceEngine",
    "nn",
    "engine",
    "models",
    "native",
    "apps",
    "__version__"
]
