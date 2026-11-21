# Phase 2: Critical Audit and Empirical Validation of DATS-SI

## 1. Mathematical and Codebase Audit

### 1.1 Autograd and Gradient Absence Verification
Every tensor operation, weight matrix, and update in `src/dats/` was audited for computational graph leakage:
- All parameter matrices ($W_l, b_l$) are initialized with `requires_grad=False`.
- Forward passes during evaluation execute under `torch.no_grad()`.
- Model output tensors verify `out.grad_fn is None` and `out.grad is None`.
- Parameter updates are strictly algebraic assignments from closed-form Tikhonov normal equations:
  $$W_{aug}^* = \left( H_{aug}^T H_{aug} + \lambda I \right)^{-1} H_{aug}^T \tilde{Z}$$
**Audit Verdict:** Confirmed 100% free of gradients, autograd graphs, and reverse-mode automatic differentiation.

### 1.2 Mathematical Inconsistency: The Expansion Nullspace & Target Standardization Paradox
A rigorous mathematical audit of the backward target synthesis stream revealed two previously hidden theoretical flaws:

1. **The Expansion Nullspace Defect ($d_{in} < d_{out}$):**
   When projecting a target backward across an expanding layer:
   $$W^\dagger = W^T (W W^T + \gamma I)^{-1}$$
   If $d_{in} < d_{out}$, the matrix $W$ has rank at most $d_{in}$. The operator $W^\dagger$ maps from $\mathbb{R}^{d_{out}} \to \mathbb{R}^{d_{in}}$, but the reconstructed forward output:
   $$\hat{Z} = (Z W^\dagger) W$$
   is restricted to the $d_{in}$-dimensional row-space of $W$. The relative reconstruction error:
   $$\frac{\|\hat{Z} - Z\|_F}{\|Z\|_F} = 0.8728 \quad (87.3\% \text{ reconstruction error})$$
   Exact target reconstruction is mathematically impossible when layer width expands!

2. **The Target Standardization Distortion:**
   In Phase 1, target standardization was introduced to prevent vanishing/exploding norms across deep layers:
   $$T_l^{\text{aligned}} = \frac{T_l^{\text{raw}} - \mu(T_l)}{\sigma(T_l) + \epsilon} \cdot \sigma(H_l) + \mu(H_l)$$
   While this numerically stabilizes activations, it fundamentally scales and shifts the target points. When tested analytically, the reconstruction error of the standardized target jumps to:
   $$\frac{\|T_l^{\text{aligned}} W - Z_{target}\|_F}{\|Z_{target}\|_F} = 4.0095 \quad (400\% \text{ distortion})$$
   **Conclusion:** DATS-SI is NOT an exact inverse solver; it is a *pseudo-inverse directional transport* that functions as a normalized feedback alignment mechanism rather than an exact algebraic inversion.

---

## 2. Deep Empirical Experiments

### 2.1 Multi-Seed Statistical Robustness (20 Independent Seeds)
Evaluated across 20 distinct random seeds ($s \in [0, 19]$):

| Benchmark | Architecture | DATS-SI Mean Acc | DATS-SI Std | DATS-SI Min | Backprop (AdamW) Mean Acc | AdamW Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XOR** | `[2, 32, 16, 2]` | 99.28% | ±3.16% | 85.50% | **100.00%** | ±0.00% |
| **Circles** | `[2, 32, 16, 2]` | 99.85% | ±0.23% | 99.50% | **100.00%** | ±0.00% |
| **Digits** | `[64, 128, 64, 10]` | 94.46% | ±0.86% | 93.10% | **97.68%** | ±0.52% |

*Finding:* DATS-SI is stable across most seeds, but exhibits higher variance than AdamW and can occasionally settle into sub-optimal local projections (e.g. 85.5% min on XOR).

### 2.2 Sample Complexity Scaling ($N \in [200, 500, 1000, 2500, 5000]$)

| Sample Size $N$ | DATS-SI Test Acc | DATS-SI Time | Backprop (AdamW) Acc | AdamW Time | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$N = 200$** | **100.00%** | **0.029s** | 85.00% | 0.051s | **1.76x** |
| **$N = 500$** | **100.00%** | **0.036s** | 100.00% | 0.109s | **3.03x** |
| **$N = 1000$** | **100.00%** | **0.053s** | 100.00% | 0.190s | **3.58x** |
| **$N = 2500$** | **100.00%** | **0.095s** | 100.00% | 0.445s | **4.68x** |
| **$N = 5000$** | **99.90%** | **0.118s** | 100.00% | 0.898s | **7.61x** |

*Finding:* In low-data regimes ($N=200$), DATS-SI substantially outperforms AdamW (100.0% vs 85.0%) because closed-form Ridge projection immediately identifies the separating hyperplane without requiring hundreds of gradient steps. Furthermore, DATS-SI scales with exceptional wall-clock efficiency, training $7.6\times$ faster at $N=5000$.

### 2.3 Extreme Depth Scaling: The 128-Layer Breakdown ($L \in [4, 8, 16, 32, 64, 128]$)

| Depth $L$ | DATS-SI Acc | DATS-SI Loss | DATS-SI Time | Backprop (AdamW) Acc | AdamW Loss | AdamW Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$L = 4$** | **100.00%** | 0.1273 | **0.095s** | 100.00% | 0.0023 | 0.228s |
| **$L = 8$** | **100.00%** | 0.1114 | **0.207s** | 100.00% | 0.0002 | 0.376s |
| **$L = 16$** | 98.50% | 0.1511 | **0.481s** | **100.00%** | 0.0000 | 0.665s |
| **$L = 32$** | 86.00% | 0.3899 | **1.008s** | **100.00%** | 0.0000 | 1.448s |
| **$L = 64$** | 93.00% | 0.4351 | **2.113s** | **100.00%** | 0.0001 | 2.377s |
| **$L = 128$** | **50.00% (COLLAPSE)** | **0.6931** | **3.686s** | **100.00%** | **0.0000** | 4.938s |

*Critical Finding:* At **$L = 128$ layers, DATS-SI completely collapses to chance (50.00% accuracy, binary cross-entropy loss = 0.6931)**!
**Mechanism of Failure:** Chained linear pseudoinverses $\prod_{k=1}^{128} W_k^\dagger$ cause mathematical rank contraction. By the time targets reach the earliest layers, the effective rank of the target matrix is 1, obliterating all non-linear topological information.

### 2.4 Width Scaling and the Cubic Solve Bottleneck ($W \in [16, 32, 64, 128, 256, 512]$)

| Width $W$ | DATS-SI Acc | DATS-SI Time | Backprop (AdamW) Acc | AdamW Time | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$W = 16$** | 99.00% | **0.042s** | 100.00% | 0.194s | DATS-SI ($4.6\times$ faster) |
| **$W = 32$** | 100.00% | **0.064s** | 100.00% | 0.186s | DATS-SI ($2.9\times$ faster) |
| **$W = 64$** | 100.00% | **0.081s** | 100.00% | 0.201s | DATS-SI ($2.5\times$ faster) |
| **$W = 128$** | 100.00% | **0.161s** | 100.00% | 0.235s | DATS-SI ($1.5\times$ faster) |
| **$W = 256$** | 100.00% | 0.371s | 100.00% | **0.306s** | **AdamW ($1.2\times$ faster)** |
| **$W = 512$** | 100.00% | 1.316s | 100.00% | **0.637s** | **AdamW ($2.1\times$ faster)** |

*Critical Finding:* The exact linear solve $(H^T H + \lambda I)^{-1}$ scales as $\mathcal{O}(W^3)$. For wide networks ($W \ge 256$), the cubic solve becomes slower than AdamW's linear $\mathcal{O}(W)$ element-wise updates.

### 2.5 Continual Learning & Catastrophic Forgetting
Evaluated on Split-Digits (Task 1: classes 0–4; Task 2: classes 5–9):
- **DATS-SI:** Task 1 initial accuracy: **99.12%** $\to$ Retained after Task 2: **0.00%** (**99.12% forgetting drop**).
- **Backprop (AdamW):** Task 1 initial accuracy: **99.56%** $\to$ Retained after Task 2: **0.00%** (**99.56% forgetting drop**).
*Finding:* Both DATS-SI and conventional backprop suffer 100% catastrophic forgetting. DATS-SI's closed-form solve overfits to the active task's covariance matrix, completely erasing prior orthogonal subspaces.

### 2.6 Sequential & Online Streaming
- Full-batch DATS-SI: **100.00%**
- Naive Mini-batch DATS-SI ($B=32$): **50.00% (Complete failure)**
- Streaming Recursive Least Squares (RLS): **100.00% (Single streaming pass)**
*Finding:* Mini-batching in DATS causes destructive inter-batch interference unless recursive covariance memory ($P_l$) is maintained.
