# Mathematical Specification: Dual-Stream Algebraic Target Synthesis & Subspace Inversion (DATS-SI)

## 1. Architectural Definition and Notation

Let a deep feedforward network comprise $L$ layers:
- Input feature space: $\mathcal{X} \subset \mathbb{R}^{d_0}$
- Intermediate hidden representation spaces: $\mathcal{H}_l \subset \mathbb{R}^{d_l}$ for $l \in \{1, 2, \dots, L-1\}$
- Output target space: $\mathcal{Y} \subset \mathbb{R}^{d_L}$
- For a batch of $N$ samples, let $X \in \mathbb{R}^{N \times d_0}$ and $Y \in \mathbb{R}^{N \times d_L}$.

Each layer $l \in \{1, \dots, L\}$ is parameterized by:
- Weight matrix: $W_l \in \mathbb{R}^{d_{l-1} \times d_l}$
- Bias vector: $b_l \in \mathbb{R}^{d_l}$
- Compact affine matrix: $\tilde{W}_l = \begin{bmatrix} W_l \\ b_l^T \end{bmatrix} \in \mathbb{R}^{(d_{l-1} + 1) \times d_l}$

---

## 2. Forward Representation Stream

For an input batch $X$:
1. Augmented base representation:
   $$\tilde{H}_0 = [X, \mathbf{1}_N] \in \mathbb{R}^{N \times (d_0 + 1)}$$
2. For $l = 1, 2, \dots, L$:
   $$Z_l = \tilde{H}_{l-1} \tilde{W}_l = H_{l-1} W_l + \mathbf{1}_N b_l^T \in \mathbb{R}^{N \times d_l}$$
   $$H_l = \phi_l(Z_l) \in \mathbb{R}^{N \times d_l}$$
   where $\phi_l: \mathbb{R} \to \mathbb{R}$ is the activation function. At the final layer $L$, $\phi_L$ can be linear (for regression) or an identity/soft mapping (for logits).
3. Augmented hidden representation for layer $l < L$:
   $$\tilde{H}_l = [H_l, \mathbf{1}_N] \in \mathbb{R}^{N \times (d_l + 1)}$$

---

## 3. Backward Algebraic Target Synthesis Stream

Unlike backpropagation, which propagates a scalar loss gradient vector $\delta_l = \nabla_{z_l} \mathcal{L}$ using the chain rule, DATS-SI constructs an **ideal target representation manifold** $T_l \in \mathbb{R}^{N \times d_l}$ directly at each hidden layer.

### 3.1 Boundary Condition at Layer $L$
At the output layer, the target is dictated by the true label:
- For regression: $T_L = Y$.
- For classification with one-hot labels $Y \in \{0, 1\}^{N \times C}$:
  Target logits are scaled to ensure a positive classification margin $\alpha > 0$:
  $$T_L = \alpha (2Y - 1) \quad \text{or} \quad T_L = \text{margin\_target}(Y, \alpha)$$
Let $\tilde{Z}_L = \psi_L(T_L)$. For linear output activation, $\psi_L(T_L) = T_L$.

### 3.2 Recursive Algebraic Inversion for Hidden Layers $l = L-1, L-2, \dots, 1$
At layer $l$, we require the post-activation $H_l$ to satisfy the linear projection into the subsequent layer's target pre-activation $\tilde{Z}_{l+1}$:
$$H_l W_{l+1} + \mathbf{1}_N b_{l+1}^T \approx \tilde{Z}_{l+1}$$
$$(H_l W_{l+1}) \approx \tilde{Z}_{l+1} - \mathbf{1}_N b_{l+1}^T$$

To find the optimal target representations $T_l^{\text{ideal}} \in \mathbb{R}^{N \times d_l}$ without knowing gradients, we solve the algebraic minimum-norm inverse:
$$T_l^{\text{raw}} = (\tilde{Z}_{l+1} - \mathbf{1}_N b_{l+1}^T) W_{l+1}^{\dagger}$$
where $W_{l+1}^{\dagger}$ is the regularized Moore-Penrose pseudoinverse:
$$W_{l+1}^{\dagger} = W_{l+1}^T \left( W_{l+1} W_{l+1}^T + \gamma I_{d_{l+1}} \right)^{-1}$$
Here $\gamma > 0$ is a Tikhonov regularizer guaranteeing well-conditioned matrix inversion.

### 3.3 Target Relaxation & Stability Preservation
To prevent target divergence across iterative training sweeps, the synthesized target is contracted with the current forward state using a relaxation factor $\beta \in (0, 1]$:
$$T_l = (1 - \beta) H_l + \beta T_l^{\text{raw}}$$
This establishes a contraction mapping in representation space, preventing chaotic oscillation.

### 3.4 Pre-activation Algebraic Preimage Lifting
To find the required pre-activation target $\tilde{Z}_l \in \mathbb{R}^{N \times d_l}$ such that $\phi_l(\tilde{Z}_l) \approx T_l$:
$$\tilde{Z}_l = \psi_l(T_l)$$
where $\psi_l = \phi_l^{-1}$.

**Theorem (Exact Bijective Preimage):**
If $\phi_l(z) = \text{LeakyReLU}(z; \alpha)$ with slope $0 < \alpha < 1$, then $\phi_l: \mathbb{R} \to \mathbb{R}$ is strictly monotonically increasing and bijective. Its exact algebraic inverse is:
$$\psi_l(u) = \begin{cases} u & \text{if } u \ge 0 \\ \frac{u}{\alpha} & \text{if } u < 0 \end{cases}$$
Because $\psi_l$ is exact and closed-form, **no autoencoder or auxiliary reconstruction network is required**, and **no information is lost in the inversion**.

---

## 4. Closed-Form Parameter Update (Zero Gradients)

At each layer $l \in \{1, \dots, L\}$, we have:
- Input data matrix: $\tilde{H}_{l-1} = [H_{l-1}, \mathbf{1}_N] \in \mathbb{R}^{N \times (d_{l-1} + 1)}$
- Target pre-activation matrix: $\tilde{Z}_l \in \mathbb{R}^{N \times d_l}$

The parameter matrix $\tilde{W}_l = \begin{bmatrix} W_l \\ b_l^T \end{bmatrix}$ is solved by minimizing the exact Frobenius projection error with $L_2$ Tikhonov regularization:
$$\tilde{W}_l^* = \arg\min_{\tilde{W}} \left\| \tilde{H}_{l-1} \tilde{W} - \tilde{Z}_l \right\|_F^2 + \lambda \|\tilde{W}\|_F^2$$

### Exact Solution:
Setting the matrix derivative to zero yields the normal equations:
$$\tilde{H}_{l-1}^T \left( \tilde{H}_{l-1} \tilde{W}_l - \tilde{Z}_l \right) + \lambda \tilde{W}_l = 0$$
$$\left( \tilde{H}_{l-1}^T \tilde{H}_{l-1} + \lambda I_{(d_{l-1} + 1)} \right) \tilde{W}_l^* = \tilde{H}_{l-1}^T \tilde{Z}_l$$
$$\tilde{W}_l^* = \left( \tilde{H}_{l-1}^T \tilde{H}_{l-1} + \lambda I \right)^{-1} \tilde{H}_{l-1}^T \tilde{Z}_l$$

### Properties of the Solution:
1. **Convex Global Optimum:** Conditioned on the target subspace $\tilde{Z}_l$, this is a strictly convex quadratic problem. The solution is the **exact, unique global minimum**—not an infinitesimal step along a gradient.
2. **Computational Complexity:**
   - Covariance computation $\tilde{H}_{l-1}^T \tilde{H}_{l-1}$: $\mathcal{O}(N \cdot d_{l-1}^2)$.
   - Target projection $\tilde{H}_{l-1}^T \tilde{Z}_l$: $\mathcal{O}(N \cdot d_{l-1} \cdot d_l)$.
   - Linear solve / inversion: $\mathcal{O}(d_{l-1}^3)$.
   Since $d_{l-1} \ll N$ in modern architectures (e.g. $d = 128, 256$), the solve takes milliseconds on commodity CPU hardware.
3. **No Hyperparameter Tuning for Optimizers:** Eliminates learning rates, Adam moments, warmup schedules, and gradient clipping entirely.

---

## 5. Streaming / Online Formulation: Recursive Least Squares (RLS) Subspace Tracking

For online learning where samples arrive one by one or in small mini-batches $x^{(t)}, y^{(t)}$:
Let $P_l^{(t)} = \left( \sum_{\tau=1}^t \tilde{h}_{l-1}^{(\tau)} (\tilde{h}_{l-1}^{(\tau)})^T + \lambda I \right)^{-1} \in \mathbb{R}^{(d_{l-1}+1) \times (d_{l-1}+1)}$.

Using the Sherman-Morrison-Woodbury theorem, $P_l^{(t)}$ and $\tilde{W}_l^{(t)}$ are updated with $\mathcal{O}(d_{l-1}^2)$ complexity without inverting any matrix:
1. Compute gain vector:
   $$k_l^{(t)} = \frac{P_l^{(t-1)} \tilde{h}_{l-1}^{(t)}}{1 + (\tilde{h}_{l-1}^{(t)})^T P_l^{(t-1)} \tilde{h}_{l-1}^{(t)}}$$
2. Compute innovation (target residual):
   $$e_l^{(t)} = \tilde{z}_l^{(t)} - (\tilde{h}_{l-1}^{(t)})^T \tilde{W}_l^{(t-1)}$$
3. Update parameter matrix:
   $$\tilde{W}_l^{(t)} = \tilde{W}_l^{(t-1)} + k_l^{(t)} e_l^{(t)}$$
4. Update inverse covariance:
   $$P_l^{(t)} = P_l^{(t-1)} - k_l^{(t)} (\tilde{h}_{l-1}^{(t)})^T P_l^{(t-1)}$$

---

## 6. Full Algorithmic Protocol: Alternating Bidirectional Subspace Sweep

```
Algorithm 1: DATS-SI Training Sweep
Input: Training dataset (X, Y), depth L, widths {d_0, ..., d_L},
       regularization lambda, pseudoinverse reg gamma, relaxation beta, sweeps S
Output: Trained parameter matrices {W_l, b_l}_{l=1}^L

1. Initialize {W_l, b_l}_{l=1}^L with orthogonal or He normal initialization.
2. For sweep s = 1 to S:
3.    # Forward Pass
4.    H_0 = X
5.    For l = 1 to L:
6.        Z_l = H_{l-1} W_l + 1 b_l^T
7.        H_l = phi_l(Z_l)
8.
9.    # Target Synthesis & Backward Inversion
10.   T_L = encode_target(Y)
11.   Z_tilde_L = psi_L(T_L)
12.   For l = L-1 down to 1:
13.       W_pinv = W_{l+1}^T (W_{l+1} W_{l+1}^T + gamma I)^{-1}
14.       T_l_raw = (Z_tilde_{l+1} - 1 b_{l+1}^T) W_pinv
15.       T_l = (1 - beta) H_l + beta T_l_raw
16.       Z_tilde_l = psi_l(T_l)
17.
18.   # Exact Algebraic Update (simultaneous across all layers)
19.   For l = 1 to L:
20.       H_aug = [H_{l-1}, 1]
21.       W_aug = solve(H_aug^T H_aug + lambda I, H_aug^T Z_tilde_l)
22.       W_l = W_aug[:-1, :]
23.       b_l = W_aug[-1, :]
24. Return {W_l, b_l}_{l=1}^L
```

---

## 7. Comparative Theoretical Properties

| Property | Conventional Backpropagation (AdamW/SGD) | Dual-Stream Algebraic Target Synthesis (DATS-SI) |
| :--- | :--- | :--- |
| **Optimization Primitive** | Chain-rule Jacobians + first-order gradient descent | Closed-form Tikhonov subspace projection + algebraic inversion |
| **Learning Rate** | Highly sensitive ($\eta$ requires tuning/decay) | **None** (No learning rate exists) |
| **Backward Locking** | Strictly locked (Layer $l$ waits for all $L \dots l+1$ backward grads) | **Decoupled** (Target synthesis only requires adjacent layer projection) |
| **Weight Symmetry** | Required ($W^T$ must be exact forward weight in backprop) | **Flexible** (Uses pseudoinverse $W^\dagger$ or fixed feedback projections) |
| **Gradient Vanishing/Exploding** | Exponential with depth $L$ due to $\prod W_k^T \sigma'$ | **Eliminated** (Each layer solved via well-conditioned regularized system) |
| **Activation Differentiability** | Required ($C^1$ almost everywhere) | **Not required** (Only requires piecewise invertibility) |
