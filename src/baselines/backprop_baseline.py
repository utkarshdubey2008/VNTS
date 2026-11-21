"""
Conventional Backpropagation Baseline Network and Trainer.
Uses PyTorch autograd, reverse-mode automatic differentiation, and standard optimizers (AdamW, SGD).
"""

from typing import List, Dict, Optional
import time
import torch
import torch.nn as nn
import torch.optim as optim


class StandardMLP(nn.Module):
    """
    Standard PyTorch MLP baseline with identical layer dimensions and LeakyReLU activations.
    """
    def __init__(self, layer_dims: List[int], negative_slope: float = 0.1, seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            linear = nn.Linear(layer_dims[i], layer_dims[i+1])
            gain = 1.0 if i == len(layer_dims) - 2 else 1.414
            nn.init.orthogonal_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)
            self.layers.append(linear)
            
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = nn.functional.leaky_relu(h, negative_slope=self.negative_slope)
        return h


class BackpropTrainer:
    """
    Conventional trainer using PyTorch Autograd, backward graph unrolling, and gradient descent.
    """
    def __init__(
        self,
        model: StandardMLP,
        optimizer_name: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        task: str = "classification"
    ):
        self.model = model
        self.task = task
        
        if optimizer_name.lower() == "adamw":
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
        self.criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    def train_epoch(self, x: torch.Tensor, y: torch.Tensor, batch_size: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        N = x.shape[0]
        
        if batch_size is None or batch_size >= N:
            self.optimizer.zero_grad()
            preds = self.model(x)
            loss = self.criterion(preds, y if self.task == "classification" else y.float())
            loss.backward()
            self.optimizer.step()
        else:
            perm = torch.randperm(N)
            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                x_b, y_b = x[idx], y[idx]
                
                self.optimizer.zero_grad()
                preds = self.model(x_b)
                loss = self.criterion(preds, y_b if self.task == "classification" else y_b.float())
                loss.backward()
                self.optimizer.step()
                
        # Metrics
        return self.evaluate(x, y)

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
        self.model.eval()
        with torch.no_grad():
            preds = self.model(x)
            if self.task == "classification":
                loss = self.criterion(preds, y).item()
                acc = (torch.argmax(preds, dim=1) == y).float().mean().item()
                return {"loss": loss, "accuracy": acc}
            else:
                loss = self.criterion(preds, y.float()).item()
                return {"loss": loss}
