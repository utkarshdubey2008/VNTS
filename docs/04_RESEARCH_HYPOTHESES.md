# Phase 4: Independently Formulated Research Hypotheses

**Initiative:** VYOMA — Independent Research Initiative  
**Date:** 2026-10-13  
**Status:** Formal Hypothesis Definition & Experimental Testability  

---

## Overview

To investigate whether a fundamentally new training paradigm can reduce dependence on GPUs, memory, and total compute, we formulate three distinct, non-trivial, and falsifiable hypotheses. Each hypothesis questions a core premise of contemporary deep learning.

---

## Hypothesis 1 (H1): Layer-Decoupled Forward Residual Plasticity (FRP)

### 1. Underlying Problem
Standard backpropagation imposes **backward locking**: layer $l$ cannot update its weights until layer $l+1 \dots L$ have completed their forward and backward passes. This mandates caching all intermediate activations across the entire depth and sequence length:
$$\text{Memory}_{\text{act}} = \mathcal{O}(L \cdot B \cdot S \cdot d)$$
This activation footprint is the primary cause of GPU Out-Of-Memory errors and forces expensive model parallelism across multiple accelerators.

### 2. Assumption Questioned
**Assumption:** Useful deep hierarchical representations can *only* emerge if weight updates are derived from the exact composite chain-rule product:
$$\delta_l = \left( \prod_{k=l+1}^L W_k^T \sigma'_k \right) \delta_L$$

### 3. Scientific Reasoning
In biological neural networks and distributed systems, local units adapt based on local input statistics and direct broadcast neuromodulatory signals (dopamine/serotonin), not reverse-time synchronous backpropagation.
If each layer $l$ receives:
1. Its local input $a_{l-1}$,
2. A direct low-rank broadcast projection of the task target/residual error $E_{\text{task}}$,
3. A local information-preserving regularizer (preventing dimensional collapse),
then each layer can compute its parameter update immediately during the forward pass. Once layer $l$ computes $a_l$, $a_{l-1}$ can be immediately deallocated from memory. Activation memory drops from $\mathcal{O}(L \cdot B \cdot d)$ to $\mathcal{O}(B \cdot d)$ (depth-independent).

### 4. Supporting vs. Falsifying Observations
- **Supporting Observations:**
  - Memory consumption remains flat as network depth $L$ increases.
  - Forward-local updates converge on non-linear benchmarks (e.g., hierarchical non-linear classification and multi-step regression) without representation collapse.
  - Wall-clock time on CPU is reduced due to immediate cache reuse and zero reverse-graph overhead.
- **Falsifying Observations:**
  - Layers suffer from representational collapse (activations collapse to identical subspaces).
  - Test generalization is strictly bounded at shallow-network performance, failing to benefit from depth.

### 5. Minimal Experiment
Construct a 4-layer non-linear network on a controlled benchmark (e.g., non-linear multi-class classification and non-linear regression). Compare standard Backpropagation vs. Forward Residual Plasticity. Measure peak RSS memory, convergence loss, and test generalization accuracy.

---

## Hypothesis 2 (H2): Low-Rank Structured Forward Gradients (L-SFG)

### 1. Underlying Problem
Backpropagation requires storing computational graphs to compute exact reverse-mode derivatives. Zeroth-order and forward-gradient methods compute directional derivatives in a single forward pass without storing any backward graph, but isotropic perturbation $z \sim \mathcal{N}(0, I_D)$ suffers from gradient estimation variance proportional to parameter dimension $D$:
$$\mathbb{V}[\hat{g}] \propto D \cdot \|\nabla \mathcal{L}\|^2$$
In models with $D > 10^5$, this variance requires millions of forward passes to achieve a stable descent step.

### 2. Assumption Questioned
**Assumption:** To achieve low-variance gradient estimation without reverse-mode autodiff, one must either accept high dimensional variance or perform full reverse-mode backpropagation.

### 3. Scientific Reasoning
The true gradients of neural network weights do not span an isotropic $D$-dimensional space; they lie in a tiny intrinsic subspace governed by the activation covariance $C_x = \mathbb{E}[x x^T]$.
If random perturbation directions $V \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ are structured as low-rank outer products $u v^T$ sampled along the dominant singular vectors of input activations, the forward directional derivative:
$$g_{\text{struct}} = \frac{\mathcal{L}(\theta + \epsilon V) - \mathcal{L}(\theta)}{\epsilon} V$$
confines exploration to the active manifold of the layer, suppressing noise from orthogonal directions.

### 4. Supporting vs. Falsifying Observations
- **Supporting Observations:**
  - Gradient variance drops by orders of magnitude compared to standard SPSA / isotropic forward gradients.
  - Training converges using forward-only evaluations without constructing PyTorch autograd graphs or caching activations.
- **Falsifying Observations:**
  - Structured perturbations fail to capture off-diagonal cross-layer gradient interactions, requiring more forward passes per unit loss reduction than standard backprop compute.

### 5. Minimal Experiment
Direct variance and convergence comparison on a controlled multi-layer network: Isotropic Forward Gradient vs. Low-Rank Structured Forward Gradient vs. Exact Backpropagation. Measure variance $\mathbb{E}[\|\hat{g} - g^*\|^2]$, forward FLOPs to target loss, and peak memory.

---

## Hypothesis 3 (H3): Closed-Form Latent Alignment (CLA) via Exact Normal Solves

### 1. Underlying Problem
First-order gradient descent requires tens of thousands of micro-steps with tiny learning rates ($\eta \sim 10^{-4}$) because it is blind to loss curvature and ill-conditioned Hessian eigenvalues.

### 2. Assumption Questioned
**Assumption:** Deep neural networks must be trained via iterative micro-gradient descent steps on continuous weights.

### 3. Scientific Reasoning
For any layer $l$ with linear mapping $Z_l = W_l A_{l-1}$, if target activations $Z_l^*$ are specified, the optimal weights $W_l^*$ can be solved **analytically and globally** in closed form via regularized ridge regression:
$$W_l^* = (Z_l^* A_{l-1}^T) (A_{l-1} A_{l-1}^T + \lambda I)^{-1}$$
By propagating targets backwards (using Moore-Penrose pseudo-inverses or dual projections) and updating weights via closed-form linear solves on batches of data, each layer takes a maximal jump to its conditional optimum. This replaces thousands of gradient descent steps with a handful of closed-form algebraic updates.

### 4. Supporting vs. Falsifying Observations
- **Supporting Observations:**
  - Drastic reduction in optimization iterations (e.g. 10–50 solves vs. 1,000–5,000 SGD steps).
  - Exact progress without learning rate tuning or momentum tuning.
- **Falsifying Observations:**
  - Target propagation generates infeasible target activations $Z_l^*$ across non-linearities, resulting in divergence or trivial solutions where pseudo-inverses blow up.

### 5. Minimal Experiment
Implement a 3-layer MLP on a non-linear benchmark. Compare convergence speed (iterations and CPU wall-clock seconds) of Closed-Form Latent Alignment vs. AdamW and SGD baselines.
