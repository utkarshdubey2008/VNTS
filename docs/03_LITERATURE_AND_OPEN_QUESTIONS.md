# Phase 3: Relevant Literature & Unresolved Scientific Questions

**Initiative:** VYOMA — Independent Research Initiative  
**Date:** 2026-10-13  
**Status:** Literature Synthesis & Critical Question Identification  

---

## 1. Survey of Alternative Training Paradigms

To rethink neural network training without presupposition, we categorize existing alternative paradigms, their proven capabilities, and their breakdown points:

| Paradigm | Primary Mechanism | Demonstrated Advantages | Known Breakdown / Failure Modes |
| :--- | :--- | :--- | :--- |
| **Standard Backprop + AdamW** (Rumelhart 1986; Loshchilov 2019) | Exact reverse-mode autodiff + 1st/2nd moment tracking | Universally scalable; reliably discovers hierarchical representations | $O(B \cdot S \cdot L \cdot d)$ activation memory; $O(P)$ optimizer state memory; requires continuous dense GEMMs; backwards locked |
| **Direct Feedback Alignment (DFA)** (Lillicrap 2016; Nøkland 2016) | Replace $W^T$ in error backward pass with fixed random matrix $B$ | Eliminates weight transport problem; simplifies backward pass | Fails on deep networks/transformers; forward-backward alignment decays rapidly as depth increases |
| **Forward-Forward Algorithm** (Hinton 2022) | Two forward passes (positive vs. negative data); local goodness maximization | Eliminates backward pass entirely; 0 activation storage across layers | Requires synthetic negative data generation; suffers from inter-layer interference; poor scaling to autoregressive language modeling |
| **Target Propagation (DTP)** (Bengio 2014; Lee 2015) | Layer-wise target activations via learned/approximate inverse mappings | Local layer updates; modular training | Inverting non-linear representations is ill-conditioned; training inverse mappings consumes more compute/memory than backprop |
| **Predictive Coding (PC)** (Rao & Ballard 1999; Millidge 2021) | Hierarchical error nodes; local Hebbian updates at relaxation equilibrium | Biologically plausible; converges to backprop gradients asymptotically | Requires iterative relaxation per sample ($10\times\text{--}50\times$ inference passes per step); high inference compute overhead |
| **Zeroth-Order (MeZO)** (Malladi et al. 2023) | Random directional perturbation (SPSA) along forward scalar loss | Exact memory of forward pass; 0 graph overhead | Variance of gradient estimator scales with parameter count $O(D)$; effective for fine-tuning, catastrophic for pre-training from scratch |
| **Greedy Layer-Wise Training** (Belilovsky 2019; Nøkland 2019) | Train layer $l$ with local auxiliary objective; freeze and train $l+1$ | Peak memory bounded to single layer; asynchronous execution | Greedy bottleneck: early layers cannot anticipate representations needed 10 layers ahead; lower asymptotic ceiling |
| **Low-Rank / Subspace Projection** (GaLore, Zhao 2024; Muon, Jordan 2024) | Project gradients into low-rank subspace or spectral orthogonalization | Retains backprop convergence with fraction of optimizer state memory | Still requires standard backward pass activations; GEMM heavy |

---

## 2. The Core Scientific Tension

The survey exposes a fundamental scientific tension:

> **The Representation Co-Adaptation Dilemma:**  
> Why has Backpropagation consistently defeated all local / forward-only / greedy alternatives at scale?  
>
> In Backpropagation, every layer's update is conditioned on the *exact composition* of all downstream layers: $\frac{\partial \mathcal{L}}{\partial W_l} = \left(\prod_{k=l+1}^L W_k^T \sigma'_k\right) \frac{\partial \mathcal{L}}{\partial z_L} a_{l-1}^T$.  
> This global coupling guarantees that features at layer $l$ adapt specifically to assist layer $l+1 \dots L$.  
> 
> Local methods decouple layers to save memory and compute, but in doing so, they sever this global cooperative signal, resulting in feature redundancy, information collapse, or greedy sub-optimality.

---

## 3. Unresolved Scientific Questions

From this tension, four precise, unresolved scientific questions emerge:

1. **Question 1: Can global credit assignment be achieved without backward activation storage?**  
   Is it possible to communicate downstream error requirements to upstream layers without caching every intermediate activation tensor across the forward graph?

2. **Question 2: Is continuous dense gradient descent necessary for representation learning?**  
   In nature and in discrete optimization, useful representations are often formed through competitive selection, sparse activation, and quantized structural adaptation. Can sparse, localized parameter updates match dense gradient descent efficiency per unit of compute?

3. **Question 3: Can the curvature of the loss surface be navigated with dramatically fewer steps?**  
   Why does SGD take $10^5$ to $10^6$ iterations to find good minima? Can directional coordinate updates, subspace second-order methods, or exact partial solves in latent representations bypass the micro-stepping regime of first-order gradient descent?

4. **Question 4: Can modular feature synthesis replace end-to-end monolithic backprop?**  
   Rather than treating the network as a giant monolithic function requiring reverse automatic differentiation, can intermediate representations be constructed as information-preserving transforms (e.g. contrastive whitening, maximum entropy projections) where downstream layers refine predictable residual errors?
