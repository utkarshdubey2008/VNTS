"""
VYOMA Research Initiative: Neural Network Architectures
Modular MLP implementation supporting standard autograd, decoupled forward execution, and local projections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CleanMLP(nn.Module):
    def __init__(self, in_features, hidden_dims, out_features, task='classification'):
        super().__init__()
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.out_features = out_features
        self.task = task
        
        self.layers = nn.ModuleList()
        prev_dim = in_features
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.out_layer = nn.Linear(prev_dim, out_features)
        
        self.activation = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self, seed=42):
        torch.manual_seed(seed)
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5) if 'np' in globals() else 2.236)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = self.activation(layer(h))
        out = self.out_layer(h)
        return out

    def forward_intermediates(self, x):
        """Returns list of pre-activations and activations for analysis."""
        activations = [x]
        pre_acts = []
        h = x
        for layer in self.layers:
            z = layer(h)
            pre_acts.append(z)
            h = self.activation(z)
            activations.append(h)
        out = self.out_layer(h)
        pre_acts.append(out)
        activations.append(out)
        return activations, pre_acts
