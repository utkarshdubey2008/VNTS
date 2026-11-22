"""
Global pytest configuration for VNTS test suite.

Caps CPU usage to prevent overloading shared / throttled hosts:
  - PyTorch/BLAS: 1 thread
  - OMP / MKL / BLIS: 1 thread each
  - NumPy (via OpenBLAS): 1 thread

This does NOT limit total test wall-clock time — it limits *CPU intensity*
so the process never consumes more than roughly one full core (≤25 % on a
4-core host), keeping load average in check.
"""
import os

import pytest


def _cap_threads():
    # Must happen before any import of torch / numpy BLAS routines
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")

    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass

    try:
        import numpy as np
        # numpy >= 2 exposes this; harmless if absent
        if hasattr(np, "set_num_threads"):
            np.set_num_threads(1)  # type: ignore[attr-defined]
    except ImportError:
        pass


# Cap threads as early as possible — before any tests or fixtures run.
_cap_threads()


@pytest.fixture(autouse=True)
def _torch_single_thread():
    """Re-apply the thread cap per test in case any import reset it."""
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass
    yield
