"""
VNTS: VYOMA Neural Training Systems
Research codebase for alternative neural network training paradigms.
"""

from .benchmarks import (
    generate_multimanifold_dataset,
    generate_nonlinear_regression_dataset,
    generate_structured_patterns
)
from .models import CleanMLP
from .metrics import (
    ExperimentTracker,
    compute_classification_accuracy,
    compute_r2_score
)
from .trainers import (
    BackpropTrainer,
    DFATrainer,
    FRPTrainer,
    SPSABaselineTrainer,
    LSFGTrainer,
    CLATrainer
)

__version__ = "0.1.0"
