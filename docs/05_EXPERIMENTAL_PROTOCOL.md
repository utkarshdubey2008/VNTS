# Phase 5: Experimental Protocol & Test Design

**Initiative:** VYOMA — Independent Research Initiative  
**Date:** 2026-10-13  
**Status:** Rigorous Experimental Protocol  

---

## 1. Experimental Objectives

This protocol rigorously evaluates three candidate alternative training paradigms against conventional backpropagation baselines (AdamW and SGD) and established alternative methods (Direct Feedback Alignment - DFA, and Isotropic Forward Gradient - SPSA):

1. **Test Candidate 1 (H1: Forward Residual Plasticity - FRP):**
   - Can forward-local parameter updates with target-residual broadcast eliminate the backward activation graph while maintaining test generalization?
2. **Test Candidate 2 (H2: Low-Rank Structured Forward Gradients - L-SFG):**
   - Can structuring perturbation directions along activation subspaces reduce forward-gradient estimator variance and achieve convergence in a forward-only regime?
3. **Test Candidate 3 (H3: Closed-Form Latent Alignment - CLA):**
   - Can direct analytic linear solves on intermediate representations converge in $10\times\text{--}100\times$ fewer steps than gradient descent?

---

## 2. Experimental Controls & Fairness

To ensure absolute scientific integrity and avoid cherry-picking:
1. **Identical Model Architecture:**
   - Multi-Layer Perceptron: 4 layers, hidden dimensions `[128, 128, 64]`, non-linear activation (GELU / Tanh).
   - Fixed parameter initialization across all compared methods using identical pseudorandom seeds (`seed=42`).
2. **Identical Data Partitioning:**
   - 80% Training, 20% Held-out Test set.
   - Exact same batches, batch order, and data distributions fed to every training algorithm.
3. **Comprehensive Metric Suite:**
   - **Generalization:** Final Test Loss and Test Accuracy / $R^2$.
   - **Peak Memory:** Resident Set Size (RSS in MB) sampled during execution via `psutil`.
   - **Computational Cost:** Cumulative Wall-Clock Time (seconds) on identical 4-vCPU host.
   - **Convergence Rate:** Steps and FLOP estimates required to reach target performance thresholds.
4. **Resource Bounds:**
   - Peak RSS limit: $\le 1.5$ GiB.
   - Execution time limit: $\le 300$ seconds per run.

---

## 3. Benchmarks

1. **Benchmark 1: Multi-Manifold Non-Linear Classification (Spiral / Concentric Moons)**
   - Tests ability to warp coordinate space and learn non-linear representations across multiple layers.
   - 5,000 samples, 4 classes, non-linear topology.
2. **Benchmark 2: Hierarchical Non-Linear Regression**
   - Complex multi-frequency interaction: $y = \sin(2\pi x_1) \cos(2\pi x_2) + \sum_{i=3}^d x_i^2$.
   - 5,000 samples, 16 input dimensions.
3. **Benchmark 3: High-Dimensional Structured Pattern Discrimination**
   - 64-dimensional feature space with correlated latent clusters, testing scaling in higher dimensions.
