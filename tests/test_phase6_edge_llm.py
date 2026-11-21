"""
Unit tests for Phase 6: Edge Deployment & Micro-LLM Causal Sequence Modeling.
"""

import resource
import torch
import pytest

from vnts import MicroLLM


def test_micro_llm_zero_autograd():
    """Verify MicroLLM has zero autograd graphs and all parameters require no gradient."""
    corpus = "the quick brown fox jumps over the lazy dog"
    model = MicroLLM.from_text(corpus, embed_dim=64, num_heads=4)

    assert not model.embeddings.requires_grad
    assert not model.R_heads.requires_grad
    assert not model.W_out.requires_grad


def test_micro_llm_encoding_decoding():
    """Verify lossless character encoding and decoding."""
    corpus = "abcdefghijklmnopqrstuvwxyz 0123456789"
    model = MicroLLM.from_text(corpus, embed_dim=32, num_heads=2)

    sample = "hello world 42"
    encoded = model.encode_text(sample)
    decoded = model.decode_tokens(encoded)
    assert decoded == sample


def test_micro_llm_training_and_generation():
    """Verify closed-form training on text corpus and autoregressive text generation."""
    torch.manual_seed(42)
    # Repeating synthetic rhythmic sequence to ensure predictable causal pattern
    pattern = "vnts learning without backpropagation is fast and cheap. " * 8
    model = MicroLLM.from_text(pattern, embed_dim=64, num_heads=4)

    res = model.train_corpus(pattern, seq_len=16)
    assert res["tokens_processed"] > 0
    assert res["training_time_s"] < 0.5
    assert res["next_token_accuracy"] > 0.40

    # Test generation
    prompt = "vnts learning"
    generated = model.generate(prompt=prompt, max_new_tokens=30, temperature=0.7)
    assert len(generated) > len(prompt)
    assert generated.startswith(prompt)


def test_micro_llm_edge_memory_budget():
    """Verify training operates comfortably within edge budget (< 5 MB allocated)."""
    import tracemalloc
    tracemalloc.start()

    corpus = "Tiny Edge LLM learning on standard commodity hardware. " * 15
    model = MicroLLM.from_text(corpus, embed_dim=64, num_heads=4)
    model.train_corpus(corpus, seq_len=16)

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_bytes / (1024 * 1024)
    assert peak_mb < 5.0, f"Memory footprint exceeded edge budget: {peak_mb:.2f} MB"
