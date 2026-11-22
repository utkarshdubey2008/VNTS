"""
Unit tests for ORMA (Orthogonal Residual Manifold Alignment) and Document Intelligence Engine.
"""

import torch
import pytest

from vnts.nn import ORMANetwork
from vnts.apps import DocumentIntelligenceEngine


def test_orma_monotonic_residual_reduction():
    """Verify that ORMANetwork strictly monotonically reduces residual error across depth."""
    torch.manual_seed(42)
    N, D, K = 100, 16, 4
    X = torch.randn(N, D)
    Y = torch.randn(N, K)

    depth = 6
    net = ORMANetwork(in_features=D, hidden_dim=32, out_features=K, depth=depth, reg_lambda=1e-3)
    res = net.fit(X, Y)

    residual_norms = res["residual_norms"]
    assert len(residual_norms) == depth + 1

    # Check strict monotonic non-increasing property: ||R_{l+1}|| <= ||R_l|| + eps
    for l in range(len(residual_norms) - 1):
        assert residual_norms[l + 1] <= residual_norms[l] + 1e-4, (
            f"Monotonicity violation at layer {l}: {residual_norms[l]} -> {residual_norms[l + 1]}"
        )

    # Final residual must be lower than initial base model residual
    assert residual_norms[-1] < residual_norms[0]
    print(f"ORMA Residual reduction: {residual_norms[0]:.4f} -> {residual_norms[-1]:.4f}")


def test_document_intelligence_engine():
    """Verify DocumentIntelligenceEngine indexes text into subspaces and retrieves query accurately."""
    engine = DocumentIntelligenceEngine(embed_dim=64, subspace_rank=4)

    doc1 = "The Indian Space Research Organisation ISRO launched exploration spacecraft to lunar and martian orbits."
    doc2 = "Deep learning architectures replace manual feature engineering with gradient based optimization algorithms."
    doc3 = "Quantum computing relies on superposition and quantum entanglement to evaluate unitary state transformations."

    engine.index_document("doc_space", doc1)
    engine.index_document("doc_deeplearning", doc2)
    engine.index_document("doc_quantum", doc3)

    assert len(engine.documents) == 3

    # Query 1: Space query should match doc1
    results = engine.search("spacecraft satellite lunar orbit launch", top_k=1)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc_space"

    # Query 2: Machine learning query should match doc2
    results_ml = engine.search("neural network gradient descent training optimization", top_k=1)
    assert len(results_ml) == 1
    assert results_ml[0]["doc_id"] == "doc_deeplearning"
