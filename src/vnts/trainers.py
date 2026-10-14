"""
VYOMA Research Initiative: Implementations of Conventional Baselines & Candidate Training Paradigms
- Backpropagation (AdamW & SGD)
- Direct Feedback Alignment (DFA)
- Candidate 1 (H1): Forward Residual Plasticity (FRP) - Decoupled local forward updates
- Candidate 2 (H2): Low-Rank Structured Forward Gradients (L-SFG) vs. Isotropic SPSA
- Candidate 3 (H3): Closed-Form Latent Alignment (CLA) - Analytic regularized normal solves
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------------------------------------------
# 1. Backpropagation Baselines
# -------------------------------------------------------------

class BackpropTrainer:
    def __init__(self, model, optimizer_name='adamw', lr=1e-3, weight_decay=1e-4):
        self.model = model
        self.optimizer_name = optimizer_name
        if optimizer_name.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    def train_step(self, x, y, criterion):
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = criterion(out, y)
        loss.backward()
        self.optimizer.step()
        
        # Flops: ~6 * params * batch_size
        n_params = sum(p.numel() for p in self.model.parameters())
        flops = 6 * n_params * x.shape[0]
        return loss.item(), out.detach(), flops


# -------------------------------------------------------------
# 2. Direct Feedback Alignment (DFA) Baseline
# -------------------------------------------------------------

class DFATrainer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.lr = lr
        # Construct fixed random feedback matrices B_l from output dimension to each hidden layer
        self.B_matrices = []
        out_dim = model.out_features
        for layer in model.layers:
            h_dim = layer.out_features
            # Sample fixed random projection B ~ N(0, 1 / sqrt(out_dim))
            B = torch.randn(h_dim, out_dim) / np.sqrt(out_dim)
            self.B_matrices.append(B)

    def train_step(self, x, y, criterion):
        activations, pre_acts = self.model.forward_intermediates(x)
        out = activations[-1]
        loss = criterion(out, y)
        
        # Compute output gradient: delta_out = dL / d(out)
        out_for_grad = out.clone().detach().requires_grad_(True)
        loss_eval = criterion(out_for_grad, y)
        loss_eval.backward()
        delta_out = out_for_grad.grad.detach()  # Shape: (B, out_dim)

        batch_size = x.shape[0]
        n_params = sum(p.numel() for p in self.model.parameters())
        flops = 2 * n_params * batch_size  # Forward pass

        with torch.no_grad():
            # Output layer standard update: grad_W = delta_out^T @ h_{L-1}
            h_last = activations[-2]
            grad_W_out = (delta_out.t() @ h_last) / batch_size
            grad_b_out = delta_out.sum(dim=0) / batch_size
            self.model.out_layer.weight.data -= self.lr * grad_W_out
            self.model.out_layer.bias.data -= self.lr * grad_b_out
            flops += 2 * self.model.out_layer.weight.numel() * batch_size

            # DFA for hidden layers: delta_l = (delta_out @ B_l^T) * sigma'(pre_act_l)
            for l_idx, layer in enumerate(self.model.layers):
                z_l = pre_acts[l_idx]
                a_prev = activations[l_idx]
                B_l = self.B_matrices[l_idx].to(x.device)
                
                # Feedback projection: (B, out_dim) @ (out_dim, h_dim) -> (B, h_dim)
                proj_error = delta_out @ B_l.t()
                
                # GELU derivative approximation or tanh derivative
                with torch.enable_grad():
                    z_var = z_l.clone().requires_grad_(True)
                    act_z = self.model.activation(z_var)
                    act_grad = torch.autograd.grad(act_z.sum(), z_var)[0].detach()
                
                delta_l = proj_error * act_grad
                grad_W_l = (delta_l.t() @ a_prev) / batch_size
                grad_b_l = delta_l.sum(dim=0) / batch_size
                
                layer.weight.data -= self.lr * grad_W_l
                layer.bias.data -= self.lr * grad_b_l
                flops += 2 * (layer.weight.numel() + B_l.numel()) * batch_size

        return loss.item(), out.detach(), flops


# -------------------------------------------------------------
# 3. Candidate 1 (H1): Forward Residual Plasticity (FRP)
# -------------------------------------------------------------

class FRPTrainer:
    """
    Decoupled Forward Residual Plasticity:
    Each layer executes local forward adaptation with target residual broadcast
    and covariance regularizer. Completely eliminates depth-wise backward graph.
    """
    def __init__(self, model, lr=1e-3, alpha_rep=0.01):
        self.model = model
        self.lr = lr
        self.alpha_rep = alpha_rep
        
        # Local readout heads for each hidden layer: layer_dim -> out_features
        self.heads = nn.ModuleList([
            nn.Linear(layer.out_features, model.out_features)
            for layer in model.layers
        ])
        
        # Independent layer optimizers (no shared computation graph across depth)
        self.optimizers = [
            torch.optim.AdamW(list(layer.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
            for layer, head in zip(model.layers, self.heads)
        ]
        self.out_optimizer = torch.optim.AdamW(model.out_layer.parameters(), lr=lr, weight_decay=1e-4)

    def _cov_loss(self, z):
        # Decorrelation penalty to prevent representation collapse across channels
        B, D = z.shape
        if B <= 1 or D <= 1:
            return torch.tensor(0.0, device=z.device)
        z_cent = z - z.mean(dim=0, keepdim=True)
        cov = (z_cent.t() @ z_cent) / (B - 1)
        # Off-diagonal penalty
        diag = torch.diag(cov)
        off_diag = cov - torch.diag(diag)
        cov_penalty = (off_diag ** 2).sum() / D
        var_penalty = torch.mean(F.relu(1.0 - torch.sqrt(diag + 1e-4)))
        return cov_penalty + var_penalty

    def train_step(self, x, y, criterion):
        h = x
        total_loss = 0.0
        batch_size = x.shape[0]
        n_params = sum(p.numel() for p in self.model.parameters())
        flops = 0

        # Layer-wise forward update without accumulating graph across layers
        for layer, head, opt in zip(self.model.layers, self.heads, self.optimizers):
            opt.zero_grad()
            # Forward local computation: input h is treated as detached constant from previous layer
            h_in = h.detach()
            z = layer(h_in)
            h_out = self.model.activation(z)
            
            # Local auxiliary prediction and loss
            y_pred_local = head(h_out)
            l_task = criterion(y_pred_local, y)
            l_rep = self._cov_loss(h_out)
            local_loss = l_task + self.alpha_rep * l_rep
            
            local_loss.backward()
            opt.step()
            
            # Advance to next layer with detached activation (previous activation can be immediately freed)
            h = h_out.detach()
            flops += 4 * (layer.weight.numel() + head.weight.numel()) * batch_size

        # Final readout layer update
        self.out_optimizer.zero_grad()
        out = self.model.out_layer(h)
        final_loss = criterion(out, y)
        final_loss.backward()
        self.out_optimizer.step()
        
        flops += 4 * self.model.out_layer.weight.numel() * batch_size
        return final_loss.item(), out.detach(), flops


# -------------------------------------------------------------
# 4. Candidate 2 (H2): Low-Rank Structured Forward Gradients (L-SFG)
#    and Isotropic SPSA Baseline
# -------------------------------------------------------------

class SPSABaselineTrainer:
    """Isotropic Simultaneous Perturbation (MeZO/SPSA style)."""
    def __init__(self, model, lr=1e-3, epsilon=1e-3):
        self.model = model
        self.lr = lr
        self.epsilon = epsilon

    def train_step(self, x, y, criterion):
        params = [p for p in self.model.parameters() if p.requires_grad]
        # Sample isotropic perturbation z ~ N(0, 1)
        perturbs = [torch.randn_like(p) for p in params]
        batch_size = x.shape[0]
        n_params = sum(p.numel() for p in params)

        with torch.no_grad():
            # + epsilon
            for p, z in zip(params, perturbs):
                p.add_(z, alpha=self.epsilon)
            loss_pos = criterion(self.model(x), y).item()
            
            # - 2 * epsilon
            for p, z in zip(params, perturbs):
                p.add_(z, alpha=-2.0 * self.epsilon)
            loss_neg = criterion(self.model(x), y).item()
            
            # Reset parameters
            for p, z in zip(params, perturbs):
                p.add_(z, alpha=self.epsilon)
            
            # Estimated directional derivative: (L+ - L-) / (2 * eps)
            df = (loss_pos - loss_neg) / (2.0 * self.epsilon)
            
            # Parameter update
            for p, z in zip(params, perturbs):
                p.sub_(z, alpha=self.lr * df)

            # Center loss for reporting
            out = self.model(x)
            loss_center = criterion(out, y).item()
            
        # 3 forward passes: 3 * (2 * n_params * batch_size)
        flops = 6 * n_params * batch_size
        return loss_center, out.detach(), flops


class LSFGTrainer:
    """
    Low-Rank Structured Forward Gradients (Candidate 2):
    Perturbations are constructed as rank-1 outer products V_l = u v^T
    where v is aligned with input activations, reducing variance by restricting
    exploration to the active manifold of each layer.
    """
    def __init__(self, model, lr=1e-3, epsilon=1e-3):
        self.model = model
        self.lr = lr
        self.epsilon = epsilon

    def _sample_structured_perturbations(self, x):
        perturbs = []
        activations, _ = self.model.forward_intermediates(x)
        
        # Hidden layers
        for l_idx, layer in enumerate(self.model.layers):
            a_in = activations[l_idx].detach()  # (B, d_in)
            # Sample v from batch empirical distribution or active projection
            v = a_in.mean(dim=0) + torch.randn(layer.in_features, device=x.device) * 0.1
            v = v / (torch.norm(v) + 1e-8)
            u = torch.randn(layer.out_features, device=x.device)
            u = u / (torch.norm(u) + 1e-8)
            
            W_perturb = torch.outer(u, v)
            b_perturb = torch.randn_like(layer.bias) / np.sqrt(layer.out_features)
            perturbs.extend([W_perturb, b_perturb])
            
        # Out layer
        a_last = activations[-2].detach()
        v_out = a_last.mean(dim=0) + torch.randn(self.model.out_layer.in_features, device=x.device) * 0.1
        v_out = v_out / (torch.norm(v_out) + 1e-8)
        u_out = torch.randn(self.model.out_layer.out_features, device=x.device)
        u_out = u_out / (torch.norm(u_out) + 1e-8)
        
        W_out_pert = torch.outer(u_out, v_out)
        b_out_pert = torch.randn_like(self.model.out_layer.bias) / np.sqrt(self.model.out_layer.out_features)
        perturbs.extend([W_out_pert, b_out_pert])
        return perturbs

    def train_step(self, x, y, criterion):
        params = [p for p in self.model.parameters() if p.requires_grad]
        perturbs = self._sample_structured_perturbations(x)
        batch_size = x.shape[0]
        n_params = sum(p.numel() for p in params)

        with torch.no_grad():
            for p, v in zip(params, perturbs):
                p.add_(v, alpha=self.epsilon)
            loss_pos = criterion(self.model(x), y).item()
            
            for p, v in zip(params, perturbs):
                p.add_(v, alpha=-2.0 * self.epsilon)
            loss_neg = criterion(self.model(x), y).item()
            
            for p, v in zip(params, perturbs):
                p.add_(v, alpha=self.epsilon)
            
            df = (loss_pos - loss_neg) / (2.0 * self.epsilon)
            
            for p, v in zip(params, perturbs):
                p.sub_(v, alpha=self.lr * df)
                
            out = self.model(x)
            loss_center = criterion(out, y).item()

        flops = 6 * n_params * batch_size
        return loss_center, out.detach(), flops


# -------------------------------------------------------------
# 5. Candidate 3 (H3): Closed-Form Latent Alignment (CLA)
# -------------------------------------------------------------

class CLATrainer:
    """
    Closed-Form Latent Alignment (Candidate 3):
    Direct analytic regularized normal equations:
    W_l* = (Z_l* A_{l-1}^T) (A_{l-1} A_{l-1}^T + lambda I)^{-1}
    Propagates targets back through pseudo-inverses, executing exact linear solves.
    """
    def __init__(self, model, lr=0.1, ridge_lambda=1e-3):
        self.model = model
        self.lr = lr  # Interpolation rate: W = (1-lr)*W + lr*W_star
        self.ridge_lambda = ridge_lambda

    def _solve_ridge(self, A_in, Z_target):
        # A_in: (B, d_in), Z_target: (B, d_out)
        # W: (d_out, d_in), b: (d_out)
        B, d_in = A_in.shape
        # Add bias column of 1s
        A_ext = torch.cat([A_in, torch.ones(B, 1, device=A_in.device)], dim=1) # (B, d_in + 1)
        
        # Normal equation: (A_ext^T A_ext + lambda I) W_ext^T = A_ext^T Z_target
        Gram = A_ext.t() @ A_ext + self.ridge_lambda * torch.eye(d_in + 1, device=A_in.device)
        RHS = A_ext.t() @ Z_target
        
        # Solve Gram * W_ext^T = RHS -> W_ext^T = Gram^{-1} RHS
        W_ext_t = torch.linalg.solve(Gram, RHS) # (d_in + 1, d_out)
        W_ext = W_ext_t.t() # (d_out, d_in + 1)
        
        W = W_ext[:, :-1]
        b = W_ext[:, -1]
        return W, b

    def train_step(self, x, y, criterion):
        activations, pre_acts = self.model.forward_intermediates(x)
        out = activations[-1]
        loss = criterion(out, y)
        batch_size = x.shape[0]

        with torch.no_grad():
            # 1. Output layer target Z_L*
            if self.model.task == 'classification':
                # Convert class indices to one-hot targets scaled to margin
                num_classes = self.model.out_features
                Y_onehot = F.one_hot(y, num_classes=num_classes).float() * 2.0 - 1.0
                # Target logit: adjust current logits towards margin
                Z_target = out + 0.5 * (Y_onehot - out)
            else:
                Z_target = y

            # Solve output layer
            h_prev = activations[-2]
            W_out_star, b_out_star = self._solve_ridge(h_prev, Z_target)
            self.model.out_layer.weight.data = (1.0 - self.lr) * self.model.out_layer.weight.data + self.lr * W_out_star
            self.model.out_layer.bias.data = (1.0 - self.lr) * self.model.out_layer.bias.data + self.lr * b_out_star

            # 2. Back-propagate targets to hidden layers using pinv
            Z_curr_target = Z_target
            W_next = self.model.out_layer.weight.data

            for l_idx in reversed(range(len(self.model.layers))):
                layer = self.model.layers[l_idx]
                a_prev = activations[l_idx]
                z_curr = pre_acts[l_idx]
                
                # Invert linear step: Target activations H_l* ~ Z_{l+1}* @ pinv(W_{l+1})^T
                # Using pseudo-inverse
                pinv_W = torch.linalg.pinv(W_next) # (d_curr, d_next)
                H_target = Z_curr_target @ pinv_W.t()
                
                # Target pre-activations (approximate activation inversion for GELU: for positive, near identity)
                Z_layer_target = torch.clamp(H_target, -5.0, 5.0)
                
                W_star, b_star = self._solve_ridge(a_prev, Z_layer_target)
                layer.weight.data = (1.0 - self.lr) * layer.weight.data + self.lr * W_star
                layer.bias.data = (1.0 - self.lr) * layer.bias.data + self.lr * b_star
                
                Z_curr_target = Z_layer_target
                W_next = layer.weight.data

            # Evaluate updated output
            out_updated = self.model(x)
            loss_updated = criterion(out_updated, y).item()

        n_params = sum(p.numel() for p in self.model.parameters())
        # Matrix solve FLOPs: ~ 2 * B * d^2 + d^3
        flops = 4 * n_params * batch_size + 2 * (128**3)
        return loss_updated, out_updated.detach(), flops
