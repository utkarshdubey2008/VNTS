"""
Comprehensive unit test suite for VNTS Core unified library.
"""

import os
import tempfile
import torch
import pytest

import vnts
from vnts import Model, Sequential, VNTSModule, AlgebraicEngine
from vnts.nn import Dense, OrthogonalMemory, COMTSequence, SpatialPatch, InvertibleLeakyReLU, Softmax


def test_vnts_module_zero_autograd():
    """Verify that every tensor in VNTS strictly has requires_grad=False."""
    model = Model.create_classifier(in_features=16, hidden_dims=[32, 32], num_classes=4)
    for name, p in model.pipeline.layers[0]._parameters.items():
        if p is not None:
            assert not p.requires_grad, f"Parameter {name} has requires_grad=True!"

    dense = Dense(10, 5)
    assert not dense.weight.requires_grad
    if dense.bias is not None:
        assert not dense.bias.requires_grad

    mem = OrthogonalMemory(10, 5)
    assert not mem.weight.requires_grad


def test_algebraic_engine_solvers():
    """Verify primal and dual Ridge solve accuracy."""
    torch.manual_seed(42)
    # Primal N >= D
    X = torch.randn(100, 10)
    W_true = torch.randn(10, 2)
    Y = X @ W_true + 0.01 * torch.randn(100, 2)
    W_est = AlgebraicEngine.solve_ridge(X, Y, reg_lambda=1e-4)
    err = torch.norm(W_est - W_true) / torch.norm(W_true)
    assert err < 0.05, f"Primal solve error too high: {err.item():.4f}"

    # Dual N < D
    X_wide = torch.randn(20, 100)
    W_wide_true = torch.randn(100, 2)
    Y_wide = X_wide @ W_wide_true
    W_wide_est = AlgebraicEngine.solve_ridge(X_wide, Y_wide, reg_lambda=1e-4)
    pred = X_wide @ W_wide_est
    fit_err = torch.norm(pred - Y_wide) / torch.norm(Y_wide)
    assert fit_err < 0.05, f"Dual solve error too high: {fit_err.item():.4f}"


def test_vnts_classifier_e2e():
    """Test high-level Model classifier assimilation and prediction."""
    from sklearn.datasets import make_classification
    torch.manual_seed(42)
    X_np, y_np = make_classification(
        n_samples=400, n_features=16, n_informative=12, n_classes=2, random_state=42
    )
    X = torch.tensor(X_np, dtype=torch.float32)
    Y = torch.tensor(y_np, dtype=torch.long)

    X_train, X_test = X[:300], X[300:]
    Y_train, Y_test = Y[:300], Y[300:]

    model = Model.create_classifier(in_features=16, hidden_dims=[128, 64], num_classes=2)
    info = model.assimilate(X_train, Y_train)
    assert info["status"] == "assimilated"
    assert info["time_s"] < 0.5

    eval_res = model.evaluate(X_test, Y_test)
    assert "accuracy" in eval_res
    assert eval_res["accuracy"] >= 0.70, f"Accuracy too low: {eval_res['accuracy']}"

    # Check predict_proba
    probs = model.predict_proba(X_test)
    assert probs.shape == (100, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(100), atol=1e-4)


def test_vnts_vision_classifier():
    """Test spatial vision model on synthetic 2D image data."""
    torch.manual_seed(42)
    B = 50
    # 1-channel 16x16 images
    images = torch.randn(B, 1, 16, 16)
    labels = torch.randint(0, 4, (B,))

    model = Model.create_vision_classifier(
        in_channels=1,
        num_classes=4,
        patch_size=4,
        stride=4,
        patch_features=32,
        hidden_dims=[32]
    )
    info = model.assimilate(images, labels)
    assert info["status"] == "assimilated"

    preds = model.predict(images)
    assert preds.shape == (B,)
    assert preds.min() >= 0 and preds.max() < 4


def test_vnts_comt_sequence():
    """Test COMTSequence autoregressive causal token prediction."""
    torch.manual_seed(42)
    vocab_size = 8
    seq_len = 10
    batch_size = 32

    # Deterministic sequence: next token = (prev + 1) % vocab_size
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for b in range(batch_size):
        start = b % vocab_size
        tokens[b] = torch.tensor([(start + t) % vocab_size for t in range(seq_len)])

    seq_model = COMTSequence(vocab_size=vocab_size, hidden_dim=64, alpha=0.85)
    seq_model.assimilate_sequence(tokens, streaming=False)

    logits = seq_model(tokens)
    assert logits.shape == (batch_size, seq_len, vocab_size)

    # Next token predictions
    pred_tokens = torch.argmax(logits[:, :-1, :], dim=-1)
    target_tokens = tokens[:, 1:]
    acc = (pred_tokens == target_tokens).float().mean().item()
    assert acc == 1.0, f"COMT sequence next-token accuracy was {acc}, expected 1.0"


def test_vnts_continual_learning_lifelong():
    """Verify lifelong learning with OrthogonalMemory retains earlier tasks."""
    torch.manual_seed(42)
    d = 128
    mem = OrthogonalMemory(in_features=d, out_features=2, enable_compaction=True)

    # Task 1
    X1 = torch.randn(60, d)
    Y1 = torch.randn(60, 2)
    mem.assimilate(X1, Y1)
    p1_initial = mem(X1)

    # Task 2
    X2 = torch.randn(60, d)
    Y2 = torch.randn(60, 2)
    mem.assimilate(X2, Y2)
    p1_after_t2 = mem(X1)

    drift = torch.norm(p1_after_t2 - p1_initial) / torch.norm(p1_initial)
    assert drift < 1e-4, f"Lifelong retention drift too high: {drift.item():.4e}"


def test_vnts_save_and_load():
    """Verify model weights and architecture state can be saved and restored accurately."""
    torch.manual_seed(42)
    model = Model.create_classifier(in_features=8, hidden_dims=[16], num_classes=2)
    X = torch.randn(20, 8)
    Y = torch.randint(0, 2, (20,))
    model.assimilate(X, Y)

    preds_before = model.forward(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "vnts_model.pt")
        model.save(save_path)

        loaded_model = Model.create_classifier(in_features=8, hidden_dims=[16], num_classes=2)
        loaded_model.load(save_path)

        preds_after = loaded_model.forward(X)
        assert torch.allclose(preds_before, preds_after, atol=1e-6)
