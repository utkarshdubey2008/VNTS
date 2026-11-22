"""
VNTS Document Intelligence & Subspace Semantic Search Engine.
Indexes and retrieves knowledge across document corpora using closed-form orthogonal subspace projections.
Runs entirely on CPU in milliseconds with zero GPU and zero external vector DB dependencies.
"""

from typing import List, Dict, Tuple, Optional
import time
import torch
import torch.nn.functional as F

from ..engine.solver import AlgebraicEngine
from ..models.bpe_llm import ByteLLM


class DocumentIntelligenceEngine:
    """
    Subspace-Based Semantic Document Indexer and Retrieval Engine.
    Embeds documents into orthogonal manifold subspaces and retrieves passages
    by maximizing projection energy: Score(q, d) = ||embed(q) @ Q_d||_2^2.
    """
    def __init__(self, embed_dim: int = 128, subspace_rank: int = 8):
        self.embed_dim = embed_dim
        self.subspace_rank = subspace_rank

        # ByteLLM used for token context embedding
        self.llm = ByteLLM(embed_dim=embed_dim, num_heads=4)

        # Document storage: list of dicts with text and subspace basis Q
        self.documents: List[Dict[str, any]] = []

    def embed_text(self, text: str) -> torch.Tensor:
        """Embeds text into a normalized D-dimensional feature representation."""
        tokens = self.llm.encode_text(text)
        if len(tokens) == 0:
            return torch.zeros(self.embed_dim)
        contexts = self.llm.track_context(tokens.unsqueeze(0))  # (1, T, embed_dim)
        # Average pooling across tokens
        mean_emb = torch.mean(contexts[0], dim=0)
        return F.normalize(mean_emb, p=2, dim=0)

    def index_document(self, doc_id: str, text: str, chunk_size: int = 128):
        """
        Chunks document, embeds segments, and extracts an orthonormal basis Q_doc (D x r).
        """
        # Segment into sliding text windows
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size // 2):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk)

        if not chunks:
            chunks = [text]

        # Embed all chunks: (num_chunks x embed_dim)
        chunk_embeddings = []
        for c in chunks:
            emb = self.embed_text(c)
            chunk_embeddings.append(emb)

        H = torch.stack(chunk_embeddings)  # (N x D)

        # Extract orthonormal basis for document representation subspace
        Q = AlgebraicEngine.orthonormal_basis(H)
        # Keep top subspace_rank columns
        r = min(self.subspace_rank, Q.shape[1])
        Q_subspace = Q[:, :r]

        self.documents.append({
            "id": doc_id,
            "text": text,
            "chunks": chunks,
            "basis": Q_subspace,
            "chunk_embeddings": H
        })

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, any]]:
        """
        Retrieves top_k relevant documents using orthogonal subspace projection energy.
        """
        t0 = time.perf_counter()
        q_emb = self.embed_text(query)  # (D,)

        scores = []
        for doc in self.documents:
            Q = doc["basis"]  # (D x r)
            # Projection of query into document subspace: q_proj = q @ Q : (r,)
            q_proj = q_emb @ Q
            # Energy = ||q @ Q||_2^2
            energy = torch.sum(q_proj ** 2).item()
            scores.append((energy, doc))

        # Sort descending by energy
        scores.sort(key=lambda x: x[0], reverse=True)

        elapsed = time.perf_counter() - t0
        results = []
        for score, doc in scores[:top_k]:
            results.append({
                "doc_id": doc["id"],
                "subspace_energy": score,
                "preview": doc["text"][:150] + ("..." if len(doc["text"]) > 150 else ""),
                "latency_ms": elapsed * 1000.0
            })

        return results
