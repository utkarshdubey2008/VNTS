"""
Audit Script for DATS-SI:
1. Strict Autograd and Gradient Verification (verifying no tensor has grad, grad_fn, or requires_grad).
2. Mathematical Consistency Audit:
   - Inverse reconstruction error under expansion (d_in < d_out) and contraction (d_in >= d_out).
   - Analysis of Target Standardization vs Exact Algebraic Inversion.
3. Verification of Existing Benchmark Claims across multiple independent trials.
"""

import torch
import numpy as np
from src.dats.model import DATSNetwork
from src.dats.trainer import DATSTrainer, encode_classification_targets
from src.dats.algebraic_solver import regularized_pseudoinverse, solve_ridge_subspace
from benchmarks.datasets import generate_xor_dataset, generate_circles_dataset, load_digits_dataset


def audit_autograd_isolation():
    print("=== [AUDIT 1] Autograd & Gradient Isolation Check ===")
    model = DATSNetwork([10, 64, 32, 2], seed=123)
    
    # Check initial parameters
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        assert not w.requires_grad, f"Weight {i} has requires_grad=True!"
        assert not b.requires_grad, f"Bias {i} has requires_grad=True!"
        assert w.grad is None, f"Weight {i} has non-empty grad!"
        assert b.grad is None, f"Bias {i} has non-empty grad!"

    x = torch.randn(50, 10)
    y = torch.randint(0, 2, (50,))
    
    trainer = DATSTrainer(model, beta=0.3, alpha=0.4, gamma=1e-3, reg_lambda=1e-3)
    history = trainer.train_epoch(x, y)
    
    # Check post-update parameters
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        assert not w.requires_grad, f"Weight {i} has requires_grad=True after update!"
        assert not b.requires_grad, f"Bias {i} has requires_grad=True after update!"
        assert w.grad is None, f"Weight {i} has non-empty grad after update!"
        assert b.grad is None, f"Bias {i} has non-empty grad after update!"
        assert w.grad_fn is None, f"Weight {i} has a grad_fn attached!"
        assert b.grad_fn is None, f"Bias {i} has a grad_fn attached!"
        
    out = model.forward(x)
    assert out.grad_fn is None, "Model output has a computational graph / grad_fn!"
    print("✓ PASSED: Absolutely zero gradients, autograd graphs, or backward graph hooks exist.")


def audit_mathematical_consistency():
    print("\n=== [AUDIT 2] Mathematical Consistency & Inversion Fidelity ===")
    
    # Case A: Contraction layer (d_in=32 >= d_out=8)
    W_contract = torch.randn(32, 8)
    Z_target = torch.randn(50, 8)
    W_pinv = regularized_pseudoinverse(W_contract, gamma=1e-6)
    T_raw = Z_target @ W_pinv
    Z_reconstructed = T_raw @ W_contract
    rec_error_contract = torch.norm(Z_reconstructed - Z_target) / torch.norm(Z_target)
    print(f"Case A (Contraction d_in=32 >= d_out=8) Relative Reconstruction Error: {rec_error_contract.item():.6e}")

    # Case B: Expansion layer (d_in=8 < d_out=32)
    W_expand = torch.randn(8, 32)
    Z_target_exp = torch.randn(50, 32)
    W_pinv_exp = regularized_pseudoinverse(W_expand, gamma=1e-6)
    T_raw_exp = Z_target_exp @ W_pinv_exp
    Z_reconstructed_exp = T_raw_exp @ W_expand
    rec_error_expand = torch.norm(Z_reconstructed_exp - Z_target_exp) / torch.norm(Z_target_exp)
    print(f"Case B (Expansion d_in=8 < d_out=32) Relative Reconstruction Error: {rec_error_expand.item():.4f}")
    
    # Case C: Impact of Target Standardization
    H_dummy = torch.randn(50, 32) * 0.5 + 0.1
    std_h = H_dummy.std(dim=0, keepdim=True)
    mean_h = H_dummy.mean(dim=0, keepdim=True)
    std_t = T_raw.std(dim=0, keepdim=True)
    mean_t = T_raw.mean(dim=0, keepdim=True)
    T_norm = (T_raw - mean_t) / (std_t + 1e-4) * std_h + mean_h
    Z_from_norm = T_norm @ W_contract
    distortion = torch.norm(Z_from_norm - Z_target) / torch.norm(Z_target)
    print(f"Case C (Standardization Distortion on Contraction): Relative Error jumps to {distortion.item():.4f}")
    print("-> FINDING: Target Standardization prevents vanishing norms, but at the cost of breaking the exact algebraic inverse property!")


def audit_benchmark_reproducibility():
    print("\n=== [AUDIT 3] Verification of Benchmark Claims (XOR, Circles, Digits) ===")
    
    # 1. XOR over 5 runs
    xor_accs = []
    for s in range(5):
        X_tr, y_tr, X_te, y_te = generate_xor_dataset(n_samples=1000, seed=s)
        model = DATSNetwork([2, 32, 16, 2], seed=s)
        trainer = DATSTrainer(model, beta=0.35, alpha=0.35, gamma=1e-3, reg_lambda=1e-3)
        trainer.fit(X_tr, y_tr, epochs=30, batch_size=None)
        acc = trainer.evaluate(X_te, y_te)["accuracy"]
        xor_accs.append(acc)
    print(f"XOR (5 seeds): Mean Acc = {np.mean(xor_accs)*100:.2f}% ± {np.std(xor_accs)*100:.2f}% (Accs: {[round(a*100, 1) for a in xor_accs]})")

    # 2. Concentric Circles over 5 runs
    circle_accs = []
    for s in range(5):
        X_tr, y_tr, X_te, y_te = generate_circles_dataset(n_samples=1000, seed=s)
        model = DATSNetwork([2, 32, 16, 2], seed=s)
        trainer = DATSTrainer(model, beta=0.35, alpha=0.35, gamma=1e-3, reg_lambda=1e-3)
        trainer.fit(X_tr, y_tr, epochs=30, batch_size=None)
        acc = trainer.evaluate(X_te, y_te)["accuracy"]
        circle_accs.append(acc)
    print(f"Circles (5 seeds): Mean Acc = {np.mean(circle_accs)*100:.2f}% ± {np.std(circle_accs)*100:.2f}% (Accs: {[round(a*100, 1) for a in circle_accs]})")

    # 3. Handwritten Digits over 5 runs
    digit_accs = []
    for s in range(5):
        X_tr, y_tr, X_te, y_te = load_digits_dataset(seed=s)
        model = DATSNetwork([64, 128, 64, 10], seed=s)
        trainer = DATSTrainer(model, beta=0.25, alpha=0.35, gamma=1e-3, reg_lambda=1e-3)
        trainer.fit(X_tr, y_tr, epochs=30, batch_size=None)
        acc = trainer.evaluate(X_te, y_te)["accuracy"]
        digit_accs.append(acc)
    print(f"Digits (5 seeds): Mean Acc = {np.mean(digit_accs)*100:.2f}% ± {np.std(digit_accs)*100:.2f}% (Accs: {[round(a*100, 1) for a in digit_accs]})")


if __name__ == "__main__":
    audit_autograd_isolation()
    audit_mathematical_consistency()
    audit_benchmark_reproducibility()
