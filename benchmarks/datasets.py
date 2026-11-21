"""
Synthetic and Real Benchmark Datasets for evaluating non-gradient learning paradigms.
"""

from typing import Tuple
import torch
import numpy as np
from sklearn.datasets import make_moons, make_circles, load_digits
from sklearn.model_selection import train_test_split


def generate_xor_dataset(n_samples: int = 1000, noise: float = 0.1, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Continuous 2D XOR benchmark with 4 clusters.
    """
    np.random.seed(seed)
    n_per_cluster = n_samples // 4
    
    # 4 quadrants: (+1, +1)->0, (-1, -1)->0, (+1, -1)->1, (-1, +1)->1
    centers = [
        ([1.0, 1.0], 0),
        ([-1.0, -1.0], 0),
        ([1.0, -1.0], 1),
        ([-1.0, 1.0], 1)
    ]
    
    xs, ys = [], []
    for center, label in centers:
        x_c = np.random.randn(n_per_cluster, 2) * noise + np.array(center)
        y_c = np.full((n_per_cluster,), label, dtype=np.int64)
        xs.append(x_c)
        ys.append(y_c)
        
    X = np.vstack(xs)
    y = np.concatenate(ys)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    return (
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )


def generate_two_spirals(n_samples: int = 1000, noise: float = 0.1, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Two interleaved spirals benchmark: highly non-linear topological classification.
    """
    np.random.seed(seed)
    n = n_samples // 2
    
    # Spiral 1
    theta = np.sqrt(np.random.rand(n)) * 3 * np.pi
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T + np.random.randn(n, 2) * noise
    
    # Spiral 2
    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T + np.random.randn(n, 2) * noise
    
    X = np.vstack([data_a, data_b])
    # Normalize features to [-1, 1] range
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    return (
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )


def generate_circles_dataset(n_samples: int = 1000, noise: float = 0.08, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concentric circles benchmark."""
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.4, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    return (
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )


def load_digits_dataset(seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Real-world handwritten digits dataset (8x8 = 64 dimensions, 10 classes, ~1797 samples).
    """
    digits = load_digits()
    X = digits.data / 16.0  # Normalize to [0, 1]
    y = digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    return (
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )
