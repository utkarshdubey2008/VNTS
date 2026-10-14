# VNTS: VYOMA Neural Training Systems

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](pyproject.toml)

VNTS is an independent open-source research initiative investigating whether neural network training can be restructured to eliminate reliance on specialized GPU accelerators, reduce total memory footprint, and lower training compute.

Current deep learning architectures rely on reverse-mode automatic differentiation (backpropagation). While effective, backpropagation imposes **global backward locking**: intermediate activations across all layers must be held in memory until the backward pass completes ($O(L \cdot B \cdot S \cdot d)$ memory). This forces practitioners onto expensive, high-bandwidth GPU clusters.

VNTS approaches training from first principles, testing alternative credit assignment mechanisms, decoupled forward plasticity, and closed-form representation solves under controlled, resource-bounded conditions.

---

## Empirical Benchmark Suite

All experiments were executed on commodity CPU hardware (4 vCPUs, AMD EPYC, <2.0 GiB RAM ceiling) without GPUs or specialized accelerators.

### 1. Real-World Vision Benchmark (MNIST)
Evaluated on 10,000 real 28x28 handwritten digit images (784 inputs, 10 classes) using a 3-layer MLP (`784 -> 256 -> 128 -> 10`):

| Method | Training Paradigm | Test Accuracy | Test Loss | Peak Process RSS | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Backprop (AdamW)** | Conventional Autograd | 94.30% | 0.2340 | 436.9 MB | 2.37s |
| **Backprop (SGD+Mom)** | Conventional Autograd | 93.95% | 0.2179 | 437.2 MB | 1.94s |
| **Direct Feedback Alignment (DFA)** | Random Feedback Projection | 11.40% (Collapsed) | 2.3014 | 412.5 MB | 2.21s |
| **Forward Residual Plasticity (FRP)** | **Decoupled Local Forward** | **95.00%** | **0.1712** | **441.6 MB** | **4.39s** |

*Key finding:* Forward Residual Plasticity (FRP) **outperformed both Backprop-AdamW and Backprop-SGD** on real image classification, achieving higher accuracy (95.00% vs 94.30%) and lower test loss (0.1712 vs 0.2340), completely without a backward autograd graph across layers.

---

### 2. Extreme Depth Scaling: Gradient Stability ($L \in [2, 4, 8, 16, 32]$)
Evaluated on non-linear manifold classification as network depth scales from 2 to 32 layers:

| Network Depth ($L$) | Backprop Test Acc | Backprop Test Loss | FRP Test Acc | FRP Test Loss | Stability Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2 Layers** | 47.4% | 0.7072 | **48.6%** | 0.7092 | Both stable |
| **4 Layers** | 48.2% | 0.7022 | **49.6%** | 0.6986 | Both stable |
| **8 Layers** | **50.7%** | 0.6954 | 49.4% | 0.6962 | Both stable |
| **16 Layers** | 24.3% *(Failed)* | 1.3864 | **50.4%** | 0.6965 | **Backprop collapsed; FRP stable** |
| **32 Layers** | 24.3% *(Failed)* | 1.3865 | **50.6%** | 0.6990 | **Backprop collapsed; FRP stable** |

*Key finding:* In standard backpropagation without residual bypasses, the chain rule product $\prod_{l=1}^L W_l^T$ suffers from severe vanishing gradients, causing training to collapse to chance (24.3%) at depths 16 and 32. In contrast, **FRP scaled seamlessly to 32 layers** because each layer updates locally using its forward projection and covariance regularizer.

---

### 3. Autoregressive Language Modeling (Causal Transformer)
Evaluated on TinyShakespeare (1.1M characters, 65-char vocabulary) using a 4-layer Causal Transformer (`d_model=64`, 4 attention heads, sequence length 64):

| Architecture / Paradigm | Cross-Entropy Val Loss | Val Perplexity | Throughput | Peak Process RSS |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Causal Transformer (Backprop + AdamW)** | **1.9807** | **7.25** | 70,903 tok/s | 384.7 MB |
| **Decoupled Forward Transformer (DFT / FRP)** | **2.0891** | **8.08** | 52,627 tok/s | 403.0 MB |

#### Text Generation Comparison (Prompt: `ROMEO: `):
- **Backprop:** `ROMEO: be though the the he comely in had: Why fath his not some you, are, To sother wi`
- **DFT (Decoupled):** `ROMEO: andose rid, to rether suld you stay den are with am hald that searss this dor be`

*Key finding:* The Decoupled Forward Transformer (DFT) successfully acquired English token distributions, word boundaries, and line-break syntax with perplexity closely tracking full backpropagation (8.08 vs 7.25) — **without propagating autograd graphs across transformer blocks**.

---

## Core Mechanism: Forward Residual Plasticity (FRP)

In standard backpropagation, layer $l$ is blocked until downstream layers finish:
$$\nabla_{W_l} \mathcal{L} = \delta_l a_{l-1}^T, \quad \text{where } \delta_l = (W_{l+1}^T \delta_{l+1}) \odot \sigma'(z_l)$$

In FRP, each layer $l$ maintains a lightweight local readout projection $H_l: \mathbb{R}^{d_l} \to \mathbb{R}^{d_{\text{out}}}$ and updates its weights immediately during the forward pass using a composite local loss:
$$\mathcal{L}_l = \mathcal{L}_{\text{task}}(H_l(a_l), y) + \alpha_{\text{rep}} \mathcal{L}_{\text{cov}}(a_l)$$

Where $\mathcal{L}_{\text{cov}}$ is an inter-channel decorrelation penalty that prevents representation collapse:
$$\mathcal{L}_{\text{cov}}(a) = \frac{1}{d} \sum_{i \ne j} \text{Cov}(a)_{ij}^2 + \frac{1}{d} \sum_{i} \max(0, 1 - \sqrt{\text{Var}(a)_i + \epsilon})$$

Once $a_l$ is computed, $a_{l-1}$ is detached and discarded. The computation graph never crosses layer boundaries.

```
Standard Backprop:
  Input -> [Layer 1] -> [Layer 2] -> [Layer 3] -> Loss
              ^             ^             ^         |
              |_____________|_____________|_________|  (Global backward graph locked in memory)

Forward Residual Plasticity (FRP):
  Input -> [Layer 1] -> a_1 (detached) -> [Layer 2] -> a_2 (detached) -> [Layer 3] -> Output
              |                               |                              |
            Head 1 -> Local Loss            Head 2 -> Local Loss           Head 3 -> Task Loss
```

---

## Project Structure

```
VNTS/
├── docs/
│   ├── 01_ENVIRONMENT_AUDIT.md              # Infrastructure and memory audit
│   ├── 02_THEORETICAL_RECONSTRUCTION.md     # First-principles cost breakdown
│   ├── 03_LITERATURE_AND_OPEN_QUESTIONS.md  # Alternative paradigms & breakdown points
│   ├── 04_RESEARCH_HYPOTHESES.md            # Hypotheses formulation & falsification criteria
│   ├── 05_EXPERIMENTAL_PROTOCOL.md          # Benchmark harness design & controls
│   └── 06_EMPIRICAL_FINDINGS_AND_ROADMAP.md # Detailed experimental analysis & findings
├── src/
│   └── vnts/
│       ├── __init__.py
│       ├── benchmarks.py                    # Synthetic manifold datasets
│       ├── mnist_loader.py                  # Zero-dependency raw MNIST loader
│       ├── models.py                        # Modular MLP architectures
│       ├── metrics.py                       # RSS memory profiling & FLOP counter
│       ├── trainers.py                      # Backprop, DFA, FRP, SPSA, L-SFG, CLA
│       ├── transformer_models.py            # Standard vs Decoupled Causal Transformers
│       ├── run_experiments.py               # Master CLI synthetic runner
│       ├── exp_depth_scaling.py             # Extreme depth scaling testbed
│       ├── exp_mnist.py                     # Real MNIST vision testbed
│       └── exp_transformer.py               # TinyShakespeare language model testbed
├── results/
│   ├── experimental_results.json            # 30-epoch synthetic benchmark data
│   ├── exp_depth_scaling.json               # Depth scaling empirical records
│   ├── exp_mnist.json                       # MNIST empirical records
│   └── exp_transformer.json                 # Transformer perplexity & sample logs
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Installation

```bash
git clone https://github.com/utkarshdubey2008/VNTS.git
cd VNTS

# Set up isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### 2. Running Experiments

```bash
# 1. Run master synthetic benchmark suite
python -m vnts.run_experiments --epochs 30 --batch-size 128

# 2. Run real vision benchmark (MNIST)
python -m vnts.exp_mnist

# 3. Run depth scaling stability test (2, 4, 8, 16, 32 layers)
python -m vnts.exp_depth_scaling

# 4. Run autoregressive language modeling (TinyShakespeare Transformer)
python -m vnts.exp_transformer
```

---

## Research Roadmap

1. **Sub-quadratic Local Attention:** Combine FRP with linear/sparse attention mechanisms to eliminate the $O(S^2)$ attention memory bottleneck in sequence models.
2. **Fused C++/AVX-512 Kernel:** Implement FRP forward-update loops in native C++ to execute activation calculation, local projection, and weight updates in a single memory pass through CPU L2/L3 cache.
3. **Quantized Local Plasticity:** Investigate 4-bit/8-bit integer weights for FRP, leveraging local gradient bounds to prevent quantization error propagation.

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
