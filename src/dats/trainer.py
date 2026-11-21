"""
Trainer for Dual-Stream Algebraic Target Synthesis (DATS-SI).
Manages forward sweeps, target synthesis, closed-form subspace updates, and metrics.
"""

from typing import Dict, List, Optional
import time
import torch
import numpy as np

from .model import DATSNetwork


def encode_classification_targets(labels: torch.Tensor, num_classes: int, margin: float = 1.0) -> torch.Tensor:
    """
    Encodes discrete class labels into continuous target logits with positive margin.
    If label is k: target_k = +margin, target_{j != k} = -margin / (num_classes - 1)
    """
    N = labels.shape[0]
    targets = torch.full((N, num_classes), -margin / max(1, num_classes - 1), dtype=torch.float32)
    targets.scatter_(1, labels.unsqueeze(1), margin)
    return targets


class DATSTrainer:
    """
    Orchestrates DATS-SI training sweeps without autograd.
    """
    def __init__(
        self,
        model: DATSNetwork,
        beta: float = 0.3,
        alpha: float = 0.4,
        gamma: float = 1e-3,
        reg_lambda: float = 1e-3,
        margin: float = 1.0,
        normalize_targets: bool = True,
        task: str = "classification"
    ):
        self.model = model
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.margin = margin
        self.normalize_targets = normalize_targets
        self.task = task

    def train_epoch(self, x: torch.Tensor, y: torch.Tensor, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Executes one epoch of DATS-SI learning using forward-cascaded subspace updates.
        """
        N = x.shape[0]
        
        # Prepare target representation
        if self.task == "classification":
            num_classes = self.model.layer_dims[-1]
            y_encoded = encode_classification_targets(y, num_classes, margin=self.margin)
        else:
            y_encoded = y.float()

        if batch_size is None or batch_size >= N:
            # Full batch algebraic sweep
            self._update_sweep(x, y_encoded)
        else:
            # Mini-batch sweeps with permutation
            perm = torch.randperm(N)
            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                x_b = x[idx]
                y_b = y_encoded[idx]
                self._update_sweep(x_b, y_b)

        # Compute post-update metrics
        with torch.no_grad():
            preds = self.model.forward(x)
            if self.task == "classification":
                pred_classes = torch.argmax(preds, dim=1)
                acc = (pred_classes == y).float().mean().item()
                loss = torch.nn.functional.cross_entropy(preds, y).item()
                return {"loss": loss, "accuracy": acc}
            else:
                mse = torch.nn.functional.mse_loss(preds, y_encoded).item()
                return {"loss": mse}

    def _update_sweep(self, x: torch.Tensor, y_encoded: torch.Tensor):
        from .algebraic_solver import solve_ridge_subspace
        
        # 1. Forward pass to collect intermediate activations
        zs, hs = self.model.forward_intermediates(x)
        
        # 2. Backward target synthesis with scale normalization
        z_targets = self.model.synthesize_targets(
            zs, hs, y_encoded,
            beta=self.beta,
            gamma=self.gamma,
            normalize_targets=self.normalize_targets
        )
        
        # 3. Forward-cascaded subspace update
        h_curr = x
        for l in range(self.model.num_layers):
            N = h_curr.shape[0]
            ones = torch.ones((N, 1), dtype=h_curr.dtype, device=h_curr.device)
            H_aug = torch.cat([h_curr, ones], dim=1)
            Z_target = z_targets[l]
            
            W_aug_opt = solve_ridge_subspace(H_aug, Z_target, reg_lambda=self.reg_lambda)
            W_new = W_aug_opt[:-1, :]
            b_new = W_aug_opt[-1, :]
            
            # Damped update for stability
            self.model.weights[l] = (1.0 - self.alpha) * self.model.weights[l] + self.alpha * W_new
            self.model.biases[l] = (1.0 - self.alpha) * self.model.biases[l] + self.alpha * b_new
            
            # Cascade forward with updated parameters to feed the next layer
            z_curr = h_curr @ self.model.weights[l] + self.model.biases[l]
            h_curr = self.model.activations[l].forward(z_curr)

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        epochs: int = 50,
        batch_size: Optional[int] = None,
        verbose: bool = False
    ) -> Dict[str, List[float]]:
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_times": []
        }
        
        start_time = time.time()
        for ep in range(epochs):
            ep_start = time.time()
            metrics = self.train_epoch(x_train, y_train, batch_size=batch_size)
            ep_time = time.time() - ep_start
            
            history["train_loss"].append(metrics["loss"])
            history["epoch_times"].append(ep_time)
            if self.task == "classification":
                history["train_acc"].append(metrics["accuracy"])
                
            if x_val is not None and y_val is not None:
                val_metrics = self.evaluate(x_val, y_val)
                history["val_loss"].append(val_metrics["loss"])
                if self.task == "classification":
                    history["val_acc"].append(val_metrics["accuracy"])
                    
            if verbose and (ep + 1) % max(1, epochs // 10) == 0:
                acc_str = f" | Train Acc: {metrics.get('accuracy', 0):.4f}" if self.task == "classification" else ""
                val_str = f" | Val Acc: {history['val_acc'][-1]:.4f}" if (x_val is not None and self.task == "classification") else ""
                print(f"Epoch {ep+1:3d}/{epochs} - Loss: {metrics['loss']:.4f}{acc_str}{val_str} ({ep_time*1000:.1f}ms)")
                
        history["total_time"] = time.time() - start_time
        return history

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            preds = self.model.forward(x)
            if self.task == "classification":
                pred_classes = torch.argmax(preds, dim=1)
                acc = (pred_classes == y).float().mean().item()
                loss = torch.nn.functional.cross_entropy(preds, y).item()
                return {"loss": loss, "accuracy": acc}
            else:
                loss = torch.nn.functional.mse_loss(preds, y).item()
                return {"loss": loss}
