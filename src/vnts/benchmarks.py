"""
VYOMA Research Initiative: Synthetic and Structural Benchmarks
Provides deterministic, controlled datasets for evaluating alternative training paradigms.
"""

import numpy as np
import torch

def generate_multimanifold_dataset(n_samples=5000, n_classes=4, noise=0.15, seed=42):
    """
    Generates an interleaved multi-manifold non-linear spiral classification dataset.
    Requires non-linear feature transformation to separate classes.
    """
    np.random.seed(seed)
    samples_per_class = n_samples // n_classes
    X = []
    y = []
    for c in range(n_classes):
        r = np.linspace(0.1, 1.0, samples_per_class)
        t = np.linspace(c * 4.0 * np.pi / n_classes, (c + 1.5) * 4.0 * np.pi / n_classes, samples_per_class)
        t += np.random.normal(0, noise, samples_per_class)
        x1 = r * np.sin(t)
        x2 = r * np.cos(t)
        # Add two nuisance non-linear dimensions to test subspace isolation
        x3 = 0.5 * np.sin(2 * np.pi * r) + np.random.normal(0, noise, samples_per_class)
        x4 = 0.5 * np.cos(2 * np.pi * r) + np.random.normal(0, noise, samples_per_class)
        X.append(np.stack([x1, x2, x3, x4], axis=1))
        y.append(np.full(samples_per_class, c))
    
    X = np.concatenate(X, axis=0).astype(np.float32)
    y = np.concatenate(y, axis=0).astype(np.int64)
    
    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    
    # Split 80/20
    split = int(0.8 * len(X))
    return (torch.from_numpy(X[:split]), torch.from_numpy(y[:split])), \
           (torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))


def generate_nonlinear_regression_dataset(n_samples=5000, n_features=16, noise=0.05, seed=42):
    """
    Generates a hierarchical multi-frequency non-linear regression dataset.
    y = sin(2*pi*x1) * cos(2*pi*x2) + tanh(sum(x_3..8)) + sum(x_9..16^2)
    """
    np.random.seed(seed)
    X = np.random.uniform(-1.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    
    term1 = np.sin(2.0 * np.pi * X[:, 0]) * np.cos(2.0 * np.pi * X[:, 1])
    term2 = np.tanh(np.sum(X[:, 2:8], axis=1))
    term3 = np.sum(X[:, 8:16]**2, axis=1) * 0.2
    
    y = (term1 + term2 + term3 + np.random.normal(0, noise, size=n_samples)).astype(np.float32)
    y = y[:, None]  # Shape (n_samples, 1)
    
    split = int(0.8 * len(X))
    return (torch.from_numpy(X[:split]), torch.from_numpy(y[:split])), \
           (torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))


def generate_structured_patterns(n_samples=5000, n_features=64, n_classes=5, seed=42):
    """
    High-dimensional structured pattern classification with latent low-rank subspace components.
    """
    np.random.seed(seed)
    # Latent true rank is 8
    latent_dim = 8
    mixing_matrix = np.random.randn(latent_dim, n_features).astype(np.float32) / np.sqrt(latent_dim)
    
    samples_per_class = n_samples // n_classes
    X_list = []
    y_list = []
    
    for c in range(n_classes):
        center = np.random.randn(latent_dim).astype(np.float32) * 2.0
        z = center + np.random.randn(samples_per_class, latent_dim).astype(np.float32) * 0.5
        x = z @ mixing_matrix + np.random.randn(samples_per_class, n_features).astype(np.float32) * 0.1
        X_list.append(x)
        y_list.append(np.full(samples_per_class, c))
        
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    
    split = int(0.8 * len(X))
    return (torch.from_numpy(X[:split]), torch.from_numpy(y[:split])), \
           (torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
