"""
Phase 9b: Improved Real-World Benchmarks
=========================================
Fixes from Phase 9a findings:
  1. CIFAR-10: Use PCA-128 pre-projection → ORMA + Ensemble (target 40%+)
  2. Adult Income: Proper categorical encoding with one-hot
  3. Character LM: Upgraded to context-aware positional embedding features

Key insight from Phase 9a:
  - VNTS ORMA on raw 3072-dim CIFAR-10 pixels = 13.4% (curse of dimensionality)
  - After PCA-128: closed-form methods approach 40%+ on 2000 samples
  - Language: frequency bag-of-chars loses all sequential structure
  - Fix: use sliding-window positional n-gram features instead
"""
import os, sys, json, time, math, warnings
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import vnts
from vnts.nn.enhanced import PCAProjectedORMA, EnsembleORMA, IncrementalORMA
from vnts.nn.orma import ORMANetwork
from vnts import AlgebraicEngine

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "phase9b_improved.json"

results = {}

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 1b: CIFAR-10 with PCA-128 Pre-projection
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BENCHMARK 1b: CIFAR-10 — PCA-128 + ORMA vs MLP")
print("="*60)

try:
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    DATA_ROOT = Path("/tmp/vnts_cifar10")
    print("Loading CIFAR-10 (cached)...")
    train_ds = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)

    N_TRAIN, N_TEST = 2000, 500
    torch.manual_seed(42)
    train_idx = torch.randperm(len(train_ds))[:N_TRAIN]
    test_idx  = torch.randperm(len(test_ds))[:N_TEST]

    def load_subset(ds, idx):
        imgs, labels = [], []
        for i in idx.tolist():
            img, lbl = ds[i]
            imgs.append(img)
            labels.append(lbl)
        return torch.stack(imgs), torch.tensor(labels)

    print(f"Loading samples...")
    X_train_img, y_train = load_subset(train_ds, train_idx)
    X_test_img,  y_test  = load_subset(test_ds,  test_idx)

    X_train_flat = X_train_img.view(N_TRAIN, -1).float()
    X_test_flat  = X_test_img.view(N_TEST,  -1).float()

    # ── PCA-128 + VNTS ORMA ────────────────────────────────────────────────
    print("\n[PCA-128 + VNTS ORMA] Training...")
    t0 = time.perf_counter()
    pca_orma = PCAProjectedORMA(
        in_features=3072, pca_dim=128, hidden_dim=256, out_features=10, depth=5, reg_lambda=1e-2
    )
    pca_orma.fit(X_train_flat, F.one_hot(y_train, 10).float())
    pca_orma_time = time.perf_counter() - t0

    with torch.no_grad():
        preds = torch.argmax(pca_orma(X_test_flat), dim=1)
        pca_orma_acc = (preds == y_test).float().mean().item()
    print(f"[PCA-128 + ORMA] Accuracy: {pca_orma_acc:.4f} | {pca_orma_time:.2f}s")

    # ── PCA-128 + Ensemble ORMA (5 members) ───────────────────────────────
    print("\n[PCA-128 + Ensemble ORMA x5] Training...")
    t0 = time.perf_counter()
    ens_orma = EnsembleORMA(
        in_features=3072, pca_dim=128, hidden_dim=256, out_features=10,
        n_estimators=5, depth=5, reg_lambda=1e-2
    )
    ens_orma.fit(X_train_flat, F.one_hot(y_train, 10).float())
    ens_orma_time = time.perf_counter() - t0

    with torch.no_grad():
        preds_ens = torch.argmax(ens_orma(X_test_flat), dim=1)
        ens_orma_acc = (preds_ens == y_test).float().mean().item()
    print(f"[Ensemble ORMA x5] Accuracy: {ens_orma_acc:.4f} | {ens_orma_time:.2f}s")

    # ── PCA-128 + Logistic Regression (fair baseline) ──────────────────────
    print("\n[PCA-128 + LR] Training...")
    from sklearn.linear_model import LogisticRegression

    # Fit PCA on train using sklearn for fair comparison
    from sklearn.decomposition import PCA
    pca = PCA(n_components=128, random_state=42)
    Z_train = pca.fit_transform(X_train_flat.numpy())
    Z_test  = pca.transform(X_test_flat.numpy())

    t0 = time.perf_counter()
    lr = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", multi_class="multinomial",
                            n_jobs=1, random_state=42)
    lr.fit(Z_train, y_train.numpy())
    lr_pca_time = time.perf_counter() - t0
    lr_pca_acc = lr.score(Z_test, y_test.numpy())
    print(f"[PCA-128 + LR] Accuracy: {lr_pca_acc:.4f} | {lr_pca_time:.2f}s")

    # ── MLP backprop on PCA-128 (fair baseline) ───────────────────────────
    print("\n[PCA-128 + MLP backprop] Training...")
    Z_tr_t = torch.tensor(Z_train, dtype=torch.float32)
    Z_te_t = torch.tensor(Z_test,  dtype=torch.float32)

    class PCAMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 10)
            )
        def forward(self, x): return self.net(x)

    mlp_pca = PCAMLP()
    opt = optim.Adam(mlp_pca.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    for epoch in range(50):
        mlp_pca.train()
        perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, 128):
            xb = Z_tr_t[perm[i:i+128]]
            yb = y_train[perm[i:i+128]]
            loss = F.cross_entropy(mlp_pca(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    mlp_pca_time = time.perf_counter() - t0

    mlp_pca.eval()
    with torch.no_grad():
        preds_mlp = torch.argmax(mlp_pca(Z_te_t), dim=1)
        mlp_pca_acc = (preds_mlp == y_test).float().mean().item()
    print(f"[PCA-128 + MLP] Accuracy: {mlp_pca_acc:.4f} | {mlp_pca_time:.2f}s")

    results["cifar10_pca128"] = {
        "n_train": N_TRAIN, "n_test": N_TEST, "pca_dim": 128,
        "vnts_pca_orma":      {"accuracy": pca_orma_acc,  "train_time_s": pca_orma_time},
        "vnts_ensemble_orma": {"accuracy": ens_orma_acc,  "train_time_s": ens_orma_time},
        "baseline_pca_lr":    {"accuracy": lr_pca_acc,    "train_time_s": lr_pca_time},
        "baseline_pca_mlp":   {"accuracy": mlp_pca_acc,   "train_time_s": mlp_pca_time},
    }
    print(f"\n[CIFAR-10 PCA-128 Summary]")
    print(f"  VNTS PCA-ORMA:    {pca_orma_acc:.1%}  ({pca_orma_time:.1f}s)")
    print(f"  VNTS Ensemble x5: {ens_orma_acc:.1%}  ({ens_orma_time:.1f}s)")
    print(f"  PCA + LR:         {lr_pca_acc:.1%}  ({lr_pca_time:.1f}s)")
    print(f"  PCA + MLP:        {mlp_pca_acc:.1%}  ({mlp_pca_time:.1f}s)")

except Exception as e:
    print(f"[CIFAR-10b FAILED] {e}")
    import traceback; traceback.print_exc()
    results["cifar10_pca128"] = {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 2b: UCI Adult — Proper Categorical Encoding
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BENCHMARK 2b: UCI Adult Income — With Categorical Encoding")
print("="*60)

try:
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    import pandas as pd

    print("Loading UCI Adult dataset (pandas mode)...")
    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    X_df = adult.data
    y_raw = adult.target

    # Encode binary target
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    # Identify cat / num columns
    cat_cols = X_df.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = X_df.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    X_proc = preprocessor.fit_transform(X_df)
    X_proc = X_proc.astype(np.float32)
    print(f"Processed shape: {X_proc.shape}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_proc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    # Cap training to 10K
    X_tr, y_tr = X_tr[:10000], y_tr[:10000]
    n_feat = X_tr.shape[1]
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_te_t = torch.tensor(y_te, dtype=torch.long)

    # ── VNTS PCA-ORMA ──────────────────────────────────────────────────────
    print("\n[VNTS PCA-32 + ORMA] Training...")
    t0 = time.perf_counter()
    pca_orma2 = PCAProjectedORMA(
        in_features=n_feat, pca_dim=min(32, n_feat-1),
        hidden_dim=64, out_features=2, depth=5, reg_lambda=1e-3
    )
    pca_orma2.fit(X_tr_t, F.one_hot(y_tr_t, 2).float())
    vnts2_time = time.perf_counter() - t0
    with torch.no_grad():
        preds2 = torch.argmax(pca_orma2(X_te_t), dim=1)
        vnts2_acc = (preds2 == y_te_t).float().mean().item()
    print(f"[VNTS PCA-ORMA] Accuracy: {vnts2_acc:.4f} | {vnts2_time:.2f}s")

    # ── Baseline LR ────────────────────────────────────────────────────────
    print("\n[Baseline LR] Training...")
    t0 = time.perf_counter()
    lr2 = LogisticRegression(max_iter=300, C=1.0, solver="saga", n_jobs=1, random_state=42)
    lr2.fit(X_tr, y_tr)
    lr2_time = time.perf_counter() - t0
    lr2_acc = lr2.score(X_te, y_te)
    print(f"[Baseline LR] Accuracy: {lr2_acc:.4f} | {lr2_time:.2f}s")

    # ── Baseline MLP sklearn ───────────────────────────────────────────────
    print("\n[Baseline MLP] Training...")
    t0 = time.perf_counter()
    mlp2 = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, random_state=42)
    mlp2.fit(X_tr, y_tr)
    mlp2_time = time.perf_counter() - t0
    mlp2_acc = mlp2.score(X_te, y_te)
    print(f"[Baseline MLP] Accuracy: {mlp2_acc:.4f} | {mlp2_time:.2f}s")

    results["adult_income_v2"] = {
        "n_train": len(X_tr), "n_test": len(X_te), "n_features": n_feat,
        "vnts_pca_orma":  {"accuracy": vnts2_acc,  "train_time_s": vnts2_time},
        "baseline_lr":    {"accuracy": lr2_acc,     "train_time_s": lr2_time},
        "baseline_mlp":   {"accuracy": mlp2_acc,    "train_time_s": mlp2_time},
    }
    print(f"\n[Adult Income Summary]")
    print(f"  VNTS PCA-ORMA: {vnts2_acc:.1%}  ({vnts2_time:.1f}s)")
    print(f"  LR:            {lr2_acc:.1%}  ({lr2_time:.1f}s)")
    print(f"  MLP:           {mlp2_acc:.1%}  ({mlp2_time:.1f}s)")

except Exception as e:
    print(f"[Adult Income v2 FAILED] {e}")
    import traceback; traceback.print_exc()
    results["adult_income_v2"] = {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 3b: Character LM with Positional N-gram Features
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BENCHMARK 3b: Char-LM — Positional N-gram Features")
print("="*60)

try:
    from pathlib import Path as _P
    import urllib.request

    SHAKESPEARE_PATH = _P("/tmp/vnts_shakespeare.txt")
    if not SHAKESPEARE_PATH.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            SHAKESPEARE_PATH
        )
    text = SHAKESPEARE_PATH.read_text()

    split = int(len(text) * 0.8)
    train_text, val_text = text[:split], text[split:]

    chars = sorted(set(text))
    vocab_size = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    encode = lambda s: [c2i[c] for c in s]

    SEQ_LEN = 16
    STRIDE   = 2

    def make_positional_features(text_str, max_samples=4000, seq_len=SEQ_LEN, stride=STRIDE):
        """
        Positional one-hot: each position in the context window gets its own
        vocab-sized slot → feature vector of size seq_len * vocab_size.
        This preserves sequential structure unlike bag-of-chars.
        Caps at max_samples to stay within RAM.
        """
        enc = encode(text_str)
        xs, ys = [], []
        for i in range(0, min(len(enc) - seq_len - 1, max_samples * stride), stride):
            context = enc[i:i + seq_len]
            # Positional one-hot: (seq_len, vocab_size) → flatten to (seq_len * vocab_size,)
            feat = torch.zeros(seq_len * vocab_size)
            for pos, tok in enumerate(context):
                feat[pos * vocab_size + tok] = 1.0
            xs.append(feat)
            ys.append(enc[i + seq_len])
        return torch.stack(xs), torch.tensor(ys)

    feat_dim = SEQ_LEN * vocab_size  # 16 * 65 = 1040
    print(f"Feature dim: {feat_dim} (positional one-hot)")
    print("Building datasets...")
    X_tr, y_tr = make_positional_features(train_text, 4000)
    X_va, y_va = make_positional_features(val_text,   1000)
    print(f"Train: {X_tr.shape}, Val: {X_va.shape}")

    # ── VNTS ORMA-LM with positional features ────────────────────────────
    print("\n[VNTS ORMA-LM positional] Training...")
    t0 = time.perf_counter()
    orma_lm = ORMANetwork(in_features=feat_dim, hidden_dim=256, out_features=vocab_size, depth=5, reg_lambda=1e-2)
    orma_lm.fit(X_tr, F.one_hot(y_tr, vocab_size).float())
    vnts_lm_time = time.perf_counter() - t0

    with torch.no_grad():
        logits_val = orma_lm(X_va)
        nll = F.cross_entropy(logits_val, y_va).item()
        vnts_ppl = math.exp(nll)
        vnts_val_acc = (torch.argmax(logits_val, 1) == y_va).float().mean().item()
    print(f"[VNTS ORMA-LM] Val PPL: {vnts_ppl:.2f} | Val acc: {vnts_val_acc:.3f} | {vnts_lm_time:.2f}s")

    # ── Baseline MLP on same features ─────────────────────────────────────
    print("\n[Baseline MLP-LM positional] Training...")
    class PosMLPLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feat_dim, 256), nn.ReLU(),
                nn.Linear(256, vocab_size)
            )
        def forward(self, x): return self.net(x)

    mlp_lm = PosMLPLM()
    opt_lm = optim.Adam(mlp_lm.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for epoch in range(10):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 256):
            xb, yb = X_tr[perm[i:i+256]], y_tr[perm[i:i+256]]
            loss = F.cross_entropy(mlp_lm(xb), yb)
            opt_lm.zero_grad(); loss.backward(); opt_lm.step()
    mlp_lm_time = time.perf_counter() - t0

    mlp_lm.eval()
    with torch.no_grad():
        lv = mlp_lm(X_va)
        mlp_ppl = math.exp(F.cross_entropy(lv, y_va).item())
        mlp_val_acc = (torch.argmax(lv, 1) == y_va).float().mean().item()
    print(f"[Baseline MLP-LM] Val PPL: {mlp_ppl:.2f} | Val acc: {mlp_val_acc:.3f} | {mlp_lm_time:.2f}s")

    results["char_lm_positional"] = {
        "feature_type": "positional_one_hot",
        "feature_dim": feat_dim,
        "seq_len": SEQ_LEN,
        "vnts_orma_lm": {"val_perplexity": vnts_ppl, "val_accuracy": vnts_val_acc, "train_time_s": vnts_lm_time},
        "baseline_mlp_lm": {"val_perplexity": mlp_ppl, "val_accuracy": mlp_val_acc, "train_time_s": mlp_lm_time},
    }
    print(f"\n[Char-LM Positional Summary]")
    print(f"  VNTS ORMA: PPL={vnts_ppl:.1f}  Val acc={vnts_val_acc:.3f}  ({vnts_lm_time:.1f}s)")
    print(f"  MLP:       PPL={mlp_ppl:.1f}  Val acc={mlp_val_acc:.3f}  ({mlp_lm_time:.1f}s)")

except Exception as e:
    print(f"[Char-LM b FAILED] {e}")
    import traceback; traceback.print_exc()
    results["char_lm_positional"] = {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("PHASE 9b COMPLETE")
print(f"Results → {RESULTS_FILE}")
print("="*60)
print(json.dumps(results, indent=2))
