# Phase 2: First-Principles Reconstruction of the Neural Network Training Problem

**Initiative:** VYOMA — Independent Research Initiative  
**Date:** 2026-10-13  
**Status:** Theoretical Reconstruction & Cost Decomposition  

---

## 1. The Core Objective Function

Neural network training is framed as empirical risk minimization (ERM):
$$\min_{\theta \in \mathbb{R}^D} \mathcal{L}(\theta) := \frac{1}{N} \sum_{i=1}^N \ell(f(x_i; \theta), y_i)$$

Where:
- $f(\cdot; \theta)$ is an $L$-layer composite function: $f(x; \theta) = f_L(f_{L-1}(\dots f_1(x; W_1) \dots; W_L))$.
- $D$ is the parameter dimensionality (ranging from $10^5$ in toy models to $10^{11}$ in frontier LLMs).
- $\ell$ is a task loss (e.g., cross-entropy $\ell(\hat{y}, y) = - \sum_c y_c \log \hat{y}_c$).

---

## 2. Decomposition of Resource Demands in Contemporary Training

Modern training relies almost exclusively on **Backpropagation via Reverse-Mode Automatic Differentiation** combined with **Stochastic Gradient Descent variants (SGD, AdamW)**.

The computational and resource profile breaks down into five distinct dimensions:

### A. Memory Footprint Breakdown
For a model with $P$ parameters trained with batch size $B$, sequence length $S$, and layer dimension $d$:
1. **Model Weights:** $P \times \text{bytes per param}$ (e.g., 2 bytes in FP16/BF16, 4 bytes in FP32).
2. **Gradients:** Same size as weights ($P \times \text{bytes}$).
3. **Optimizer States:** For AdamW, stores 1st moment ($m_t$) and 2nd moment ($v_t$) plus master FP32 weights:
   $$\text{Memory}_{\text{AdamW}} = 4 \text{ (master)} + 4 \text{ (momentum)} + 4 \text{ (variance)} = 12\text{ to }16\text{ bytes per parameter}$$
4. **Activation Memory:** Backpropagation requires caching all intermediate layer activations:
   $$\text{Memory}_{\text{activations}} \propto B \times S \times L \times d$$
   In modern architectures (e.g., Transformers), activation memory quickly surpasses parameter memory as sequence length and batch size increase, necessitating activation checkpointing (which trades $33\%$ additional compute to save memory).

### B. FLOP Decomposition
Per token/sample:
- **Forward Pass:** $\approx 2P$ FLOPs (1 multiply + 1 add per parameter).
- **Backward Pass:** $\approx 4P$ FLOPs ($2P$ FLOPs to compute activation gradients $\delta_{l-1} = W_l^T \delta_l$, and $2P$ FLOPs to compute weight gradients $\nabla_{W_l} = \delta_l a_{l-1}^T$).
- **Total:** $\approx 6P$ FLOPs per token per step.

### C. The Memory Wall & Arithmetic Intensity
- Hardware FLOP counters (e.g., modern CPUs/GPUs) are severely bottlenecked by **memory bandwidth** ($\approx 50\text{--}100\text{ GB/s}$ on CPU DDR4/DDR5 vs. terabytes/s compute capability).
- Optimizer updates (e.g. Adam element-wise operations) have an arithmetic intensity of $\approx 1$ FLOP per byte transferred. They are completely memory-bandwidth bound.
- Standard training spends massive amounts of time reading weights and optimizer states from memory, writing them back, and shuffling activation tensors.

---

## 3. Fundamental Inherent Costs vs. Design Choice Costs

| Aspect | Fundamental (Information / Mathematical Limit) | Consequence of Current Design Choices (Potentially Avoidable) |
| :--- | :--- | :--- |
| **Information Capacity** | **Fundamental:** A model must absorb $I(X; Y)$ information to represent target mappings without underfitting. | **Design Choice:** Updating *every single weight* on *every single token* using dense floating-point operations. |
| **Activation Storage** | **Not Fundamental:** The brain does not store high-precision activations across 100 sequential layers to execute a reverse-time gradient pass. | **Design Choice:** Reverse-mode autodiff requires backwards-locking: layer $l$ cannot compute $\nabla W_l$ until layer $l+1$ propagates its error backward. |
| **Optimizer States** | **Not Fundamental:** Adam's second-moment tracking is a heuristic patch for ill-conditioned Hessian curvature. | **Design Choice:** Tracking rolling variances of first-order gradients consumes $2\times\text{--}3\times$ parameter memory. |
| **Optimization Steps** | **Partially Fundamental:** Non-convex optimization requires exploration of rugged landscapes. | **Design Choice:** Infinitesimal step sizes ($\eta \approx 10^{-4}$) are forced because first-order gradients carry no global curvature information. |
| **Dense GEMM Reliance** | **Not Fundamental:** Universal approximation does not mathematically require dense matrix multiplications. | **Design Choice:** Deep learning co-evolved with dense GPU matrix multiply accelerators (GEMM), biasing architectures away from sparse, discrete, or localized representations. |

---

## 4. Key Bottlenecks to Address

1. **The Backpropagation Memory Lock:** Can parameters be updated locally or forward-only, eliminating the need to store activations across the network depth?
2. **The First-Order Step Size Constraint:** Why does gradient descent require millions of micro-steps? Can directional or subspace updates capture higher curvature without $O(D^2)$ Hessian calculation?
3. **The Dense Update Inefficiency:** 99% of parameters in an overparameterized network may have negligible gradient contribution on a given sample. Can sparsity, adaptive routing, or localized plasticity yield identical convergence with a fraction of operations?
4. **The Loss Geometry & Precision:** Are 32-bit/16-bit continuous gradients necessary for learning representations, or can discrete/quantized feedback mechanisms achieve equivalent generalization?
