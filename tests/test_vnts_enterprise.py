"""
Enterprise test suite for VNTS: MORA Attention, Time-Series Forecaster, VNTS-LM, and Res-VNTS Vision.
"""

import torch
import pytest

from vnts.nn import MultiScaleOrthogonalAttention, ContinuousStateSpaceTracker, ResVNTSVision
from vnts.models import VNTSLanguageModel


def test_mora_attention_forward():
    """Verify MultiScaleOrthogonalAttention forward pass and zero autograd."""
    B, T, D = 8, 16, 64
    x = torch.randn(B, T, D)
    attn = MultiScaleOrthogonalAttention(embed_dim=D, num_heads=4)

    out, state = attn(x)
    assert out.shape == (B, T, D)
    assert state.shape == (B, 4, 16, 16)
    assert not out.requires_grad


def test_continuous_time_series_tracker():
    """Verify ContinuousStateSpaceTracker fits multivariate sequence in closed form."""
    torch.manual_seed(42)
    B, T, D = 10, 50, 4
    # Multi-frequency sine waves
    t = torch.linspace(0, 10, T).unsqueeze(0).repeat(B, 1)
    x_seq = torch.stack([torch.sin(t), torch.cos(2 * t), torch.sin(0.5 * t), torch.cos(t)], dim=-1)
    # Target is next step prediction
    target_seq = torch.roll(x_seq, shifts=-1, dims=1)

    tracker = ContinuousStateSpaceTracker(in_features=D, state_dim=32, forecast_horizon=1)
    res = tracker.fit_sequence(x_seq, target_seq, streaming=False)

    assert res["samples_processed"] == B * T
    assert res["fit_time_s"] < 0.2
    assert res["mse_loss"] < 0.2, f"Time-series MSE too high: {res['mse_loss']}"


def test_vnts_language_model():
    """Verify VNTSLanguageModel (powered by MORA) trains and generates text."""
    torch.manual_seed(42)
    corpus = "Artificial Intelligence without backpropagation is fast and computationally efficient! " * 6
    lm = VNTSLanguageModel.from_corpus(corpus, embed_dim=64, num_heads=4, ffn_dim=128)

    stats = lm.train_text(corpus, seq_len=16)
    assert stats["tokens_processed"] > 0
    assert stats["training_time_s"] < 0.5
    assert stats["next_token_accuracy"] > 0.40

    prompt = "Artificial Intelligence"
    gen = lm.generate(prompt=prompt, max_tokens=30, temperature=0.7)
    assert len(gen) > len(prompt)
    assert gen.startswith(prompt)


def test_res_vnts_vision_deep():
    """Verify ResVNTSVision propagates through multi-scale color patches and residual stack."""
    torch.manual_seed(42)
    B = 20
    images = torch.randn(B, 3, 32, 32)
    labels = torch.randint(0, 5, (B,))

    net = ResVNTSVision(in_channels=3, num_classes=5, depth=3, features_dim=64)
    elapsed = net.assimilate(images, labels)
    assert elapsed < 0.5

    logits = net(images)
    assert logits.shape == (B, 5)
    preds = torch.argmax(logits, dim=1)
    assert preds.shape == (B,)
