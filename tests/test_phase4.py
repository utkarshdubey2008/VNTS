"""
Comprehensive Unit Tests for Phase 4 Modules:
1. Spectral Memory Compactor (SMC-DN)
2. Causal Orthogonal Sequence Model (COMT)
3. Unsupervised Contrastive Subspace Learner (CSD)
4. Spatial Subspace Convolution (Spatial-OSM)
"""

import pytest
import torch
from src.osm.compaction import SpectralMemoryCompactor, DynamicNeurogenesisEngine
from src.osm.causal_sequence import CausalOrthogonalSequenceModel
from src.osm.unsupervised import ContrastiveSubspaceLearner
from src.osm.spatial_conv import SpatialOSMNetwork, SpatialPatchProjector


def test_spectral_memory_compaction():
    """Verify that spectral compaction frees dimensions while retaining energy."""
    D = 128
    R = 80
    U, _ = torch.linalg.qr(torch.randn(D, R))
    W = torch.randn(D, 10)
    
    compactor = SpectralMemoryCompactor(capacity_threshold=0.6, target_compression_ratio=0.5)
    assert compactor.needs_compaction(R, D)
    
    U_compact, freed = compactor.compact(U, W)
    assert U_compact.shape[0] == D
    assert U_compact.shape[1] < R
    assert freed > 0
    # Orthonormality check
    I_rec = U_compact.T @ U_compact
    assert torch.allclose(I_rec, torch.eye(U_compact.shape[1]), atol=1e-5)


def test_dynamic_neurogenesis():
    """Verify that dynamic neurogenesis expands projection channels safely."""
    engine = DynamicNeurogenesisEngine(growth_step=32, max_dim=256)
    W_proj = torch.randn(16, 64)
    b_proj = torch.randn(64)
    
    W_new, b_new = engine.grow_expansion_layer(W_proj, b_proj, in_dim=16, growth=32)
    assert W_new.shape == (16, 96)
    assert b_new.shape == (96,)
    # Original channels must remain unaltered
    assert torch.equal(W_new[:, :64], W_proj)
    assert torch.equal(b_new[:64], b_proj)


def test_comt_sequence_learning():
    """Verify that COMT learns causal sequence next-token prediction in O(1) memory."""
    vocab_size = 8
    seq = [i % vocab_size for i in range(500)]
    
    comt = CausalOrthogonalSequenceModel(vocab_size=vocab_size, embed_dim=32, seed=42)
    assert not comt.W_readout.requires_grad
    assert not comt.embedding.requires_grad
    assert not comt.R.requires_grad
    
    res = comt.train_sequence(seq)
    assert res["acc"] >= 0.95
    eval_res = comt.evaluate_sequence(seq[:100])
    assert eval_res["acc"] >= 0.95


def test_unsupervised_csd_alignment():
    """Verify that CSD extracts discriminative representations without labels."""
    torch.manual_seed(42)
    N = 200
    d = 16
    X = torch.randn(N, d)
    # Augmented views with small perturbation
    view_a = X + 0.05 * torch.randn_like(X)
    view_b = X + 0.05 * torch.randn_like(X)
    
    csd = ContrastiveSubspaceLearner(in_dim=d, latent_dim=8, reg_lambda=1e-3, seed=42)
    assert not csd.W.requires_grad
    
    csd.fit_views(view_a, view_b)
    Z_a = csd.transform(view_a)
    Z_b = csd.transform(view_b)
    
    assert Z_a.shape == (N, 8)
    assert Z_b.shape == (N, 8)
    # Check that representations of augmented views are strongly correlated
    corr = torch.cosine_similarity(Z_a, Z_b, dim=1).mean().item()
    assert corr > 0.85, f"Unsupervised views failed to align: correlation = {corr}"


def test_spatial_osm_forward_and_learn():
    """Verify that SpatialOSMNetwork executes patch convolution and closed-form fitting."""
    torch.manual_seed(42)
    B = 20
    images = torch.randn(B, 3, 16, 16)
    labels = torch.randint(0, 2, (B,))
    
    model = SpatialOSMNetwork(
        in_channels=3,
        image_size=16,
        patch_size=4,
        patch_embed_dim=32,
        out_dim=2,
        seed=42
    )
    
    out = model.forward(images)
    assert out.shape == (B, 2)
    assert not out.requires_grad
    
    model.learn_task(images, labels, num_classes=2)
    eval_res = model.evaluate(images, labels)
    assert eval_res["accuracy"] >= 0.90
