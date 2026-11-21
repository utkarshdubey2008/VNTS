"""
Python C-Types FFI Bridge to native libvnts_core.so compiled from Rust.
Enables running native bare-metal linear algebra and nullspace projections directly from Python.
"""

import os
import ctypes
from typing import Optional
import numpy as np


def _find_native_library() -> Optional[str]:
    # Look in cargo release or debug build directories
    candidates = [
        os.path.join(os.path.dirname(__file__), "../../../crates/vnts-core/target/release/libvnts_core.so"),
        os.path.join(os.path.dirname(__file__), "../../../crates/vnts-core/target/debug/libvnts_core.so"),
        "/usr/local/lib/libvnts_core.so",
    ]
    for p in candidates:
        abs_p = os.path.abspath(p)
        if os.path.isfile(abs_p):
            return abs_p
    return None


class RustNativeEngine:
    """
    Direct C-FFI wrapper to the native compiled Rust engine (libvnts_core.so).
    """
    _lib = None

    @classmethod
    def load_library(cls) -> bool:
        if cls._lib is not None:
            return True
        lib_path = _find_native_library()
        if lib_path is None:
            return False
        try:
            lib = ctypes.CDLL(lib_path)

            # Define C argument types
            lib.vnts_native_solve_ridge.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float)
            ]
            lib.vnts_native_solve_ridge.restype = ctypes.c_int

            lib.vnts_native_solve_nullspace.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float)
            ]
            lib.vnts_native_solve_nullspace.restype = ctypes.c_int

            cls._lib = lib
            return True
        except Exception:
            return False

    @classmethod
    def is_available(cls) -> bool:
        return cls.load_library()

    @classmethod
    def solve_ridge(cls, X: np.ndarray, Y: np.ndarray, reg_lambda: float = 1e-3) -> np.ndarray:
        if not cls.load_library():
            raise RuntimeError("libvnts_core.so native Rust library is not available.")

        X = np.ascontiguousarray(X, dtype=np.float32)
        Y = np.ascontiguousarray(Y, dtype=np.float32)
        N, D = X.shape
        K = Y.shape[1]

        W = np.empty((D, K), dtype=np.float32)

        x_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w_ptr = W.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        status = cls._lib.vnts_native_solve_ridge(
            x_ptr, y_ptr, N, D, K, ctypes.c_float(reg_lambda), w_ptr
        )
        if status != 0:
            raise RuntimeError(f"Native Rust solve_ridge failed with error code: {status}")

        return W

    @classmethod
    def solve_nullspace(
        cls,
        X: np.ndarray,
        Y: np.ndarray,
        U_prior: Optional[np.ndarray],
        reg_lambda: float = 1e-3
    ) -> np.ndarray:
        if not cls.load_library():
            raise RuntimeError("libvnts_core.so native Rust library is not available.")

        X = np.ascontiguousarray(X, dtype=np.float32)
        Y = np.ascontiguousarray(Y, dtype=np.float32)
        N, D = X.shape
        K = Y.shape[1]

        W = np.empty((D, K), dtype=np.float32)

        x_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w_ptr = W.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if U_prior is not None and U_prior.shape[1] > 0:
            U = np.ascontiguousarray(U_prior, dtype=np.float32)
            r = U.shape[1]
            u_ptr = U.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            u_ptr = None
            r = 0

        status = cls._lib.vnts_native_solve_nullspace(
            x_ptr, y_ptr, u_ptr, N, D, K, r, ctypes.c_float(reg_lambda), w_ptr
        )
        if status != 0:
            raise RuntimeError(f"Native Rust solve_nullspace failed with error code: {status}")

        return W
