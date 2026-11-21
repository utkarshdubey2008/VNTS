"""
Multivariate Time-Series State-Space Tracker (VNTS-TS).
Performs online continuous signal forecasting and regime adaptation
without backpropagation, gradient descent, or retraining loops.
"""

from typing import Tuple, Optional, Dict
import time
import torch
import torch.nn.functional as F

from ..base import VNTSModule
from ..engine.solver import AlgebraicEngine


class ContinuousStateSpaceTracker(VNTSModule):
    """
    Multivariate Continuous State-Space Tracker for multi-step ahead time-series forecasting.
    Tracks dynamical state transitions and updates forecast readouts online via closed-form RLS.
    """
    def __init__(
        self,
        in_features: int,
        state_dim: int = 64,
        forecast_horizon: int = 1,
        out_features: Optional[int] = None,
        decay_factor: float = 0.95,
        reg_lambda: float = 1e-3
    ):
        super().__init__()
        self.in_features = in_features
        self.state_dim = state_dim
        self.forecast_horizon = forecast_horizon
        self.out_features = out_features if out_features is not None else in_features
        self.decay_factor = decay_factor
        self.reg_lambda = reg_lambda

        # 1. Input transition matrix B: (in_features, state_dim)
        B_mat = torch.randn(in_features, state_dim) / (in_features ** 0.5)
        self.register_parameter("B", B_mat)

        # 2. State transition matrix A in SO(state_dim): exact norm conservation
        M = torch.randn(state_dim, state_dim)
        Q, _ = torch.linalg.qr(M)
        self.register_parameter("A", Q)

        # 3. Readout Forecast Weights: (state_dim, out_features)
        W_forecast = torch.zeros(state_dim, self.out_features)
        self.register_parameter("W_forecast", W_forecast)

        # 4. Streaming Woodbury inverse covariance matrix P: (state_dim, state_dim)
        P = (1.0 / reg_lambda) * torch.eye(state_dim)
        self.register_buffer("P", P)

        # Internal persistent state
        self.register_buffer("current_state", torch.zeros(1, state_dim))

    @property
    def B(self) -> torch.Tensor:
        return self._parameters["B"]

    @property
    def A(self) -> torch.Tensor:
        return self._parameters["A"]

    @property
    def W_forecast(self) -> torch.Tensor:
        return self._parameters["W_forecast"]

    @property
    def P(self) -> torch.Tensor:
        return self._buffers["P"]

    def step(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes single-step incoming observation x_t: (B, in_features).
        Updates state: s_t = decay * (s_{t-1} @ A) + (1 - decay) * (x_t @ B)
        Returns: (forecast_next, next_state)
        """
        # Linear injection
        injection = x_t @ self.B
        state = self.decay_factor * (self._buffers["current_state"] @ self.A) + (1.0 - self.decay_factor) * injection
        self._buffers["current_state"] = state
        forecast = state @ self.W_forecast
        return forecast, state

    def forward_sequence(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes batch sequence: (B, T, in_features).
        Returns:
            predictions: (B, T, out_features)
            states: (B, T, state_dim)
        """
        B, T, D = x_seq.shape
        device = x_seq.device

        states = torch.zeros(B, T, self.state_dim, device=device)
        s_prev = torch.zeros(B, self.state_dim, device=device)

        for t in range(T):
            x_t = x_seq[:, t, :]
            s_t = self.decay_factor * (s_prev @ self.A) + (1.0 - self.decay_factor) * (x_t @ self.B)
            states[:, t, :] = s_t
            s_prev = s_t

        preds = states @ self.W_forecast
        return preds, states

    def fit_sequence(
        self,
        x_seq: torch.Tensor,
        target_seq: torch.Tensor,
        streaming: bool = False
    ) -> Dict[str, float]:
        """
        Fits forecast readout in a single forward pass without backpropagation.
        Args:
            x_seq: (B, T, in_features)
            target_seq: (B, T, out_features)
        """
        t0 = time.perf_counter()
        preds, states = self.forward_sequence(x_seq)

        X_flat = states.reshape(-1, self.state_dim)
        Y_flat = target_seq.reshape(-1, self.out_features)

        if not streaming:
            # Batch closed-form Cholesky Ridge solve
            new_W = AlgebraicEngine.solve_ridge(X_flat, Y_flat, reg_lambda=self.reg_lambda)
            self._parameters["W_forecast"] = new_W
        else:
            # Online streaming rank-1 Woodbury updates
            P = self.P
            W = self.W_forecast
            for i in range(X_flat.shape[0]):
                x_i = X_flat[i]
                y_i = Y_flat[i]
                P, k = AlgebraicEngine.woodbury_update(P, x_i)
                err = y_i - (x_i @ W)
                W = W + torch.outer(k, err)
            self._parameters["W_forecast"] = W
            self._buffers["P"] = P

        elapsed = time.perf_counter() - t0
        with torch.no_grad():
            final_preds = states @ self.W_forecast
            mse = F.mse_loss(final_preds, target_seq).item()

        return {
            "samples_processed": X_flat.shape[0],
            "fit_time_s": elapsed,
            "mse_loss": mse,
            "throughput_points_per_s": X_flat.shape[0] / max(1e-6, elapsed)
        }
