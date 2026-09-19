from __future__ import annotations

from .bm25_index import BM25Index
from .embeddings import Embedder
from .models import RetrievalFilters, RetrievedChunk
from .vectorstore import VectorStore


class HybridRetriever:
    """Dense (Qdrant) + sparse (BM25) retrieval fused with Reciprocal Rank Fusion.

    Dense search catches paraphrases and cross-language questions; BM25 catches
    exact identifiers (invoice numbers, ICE, article numbers) that embeddings blur.
    Access control and metadata filters are applied inside both searches, so a
    chunk the user may not see never reaches the fusion step.
    """

    def __init__(self, store: VectorStore, bm25: BM25Index, embedder: Embedder, rrf_k: int = 60):
        self.store, self.bm25, self.embedder, self.rrf_k = store, bm25, embedder, rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidates: int = 20,
        filters: RetrievalFilters | None = None,
        allowed_departments: list[str] | None = None,
        query_vector=None,
    ) -> list[RetrievedChunk]:
        if query_vector is None:
            query_vector = self.embedder.embed_query(query)
        dense = self.store.search(query_vector, candidates, filters, allowed_departments)
        sparse = self.bm25.search(query, candidates, filters, allowed_departments)

        fused: dict[str, RetrievedChunk] = {}
        for rank, (chunk, score) in enumerate(dense):
            rc = fused.setdefault(chunk.chunk_id, RetrievedChunk(chunk=chunk))
            rc.dense_score = score
            rc.fused_score += 1.0 / (self.rrf_k + rank + 1)
        for rank, (chunk, score) in enumerate(sparse):
            rc = fused.setdefault(chunk.chunk_id, RetrievedChunk(chunk=chunk))
            rc.bm25_score = score
            rc.fused_score += 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(fused.values(), key=lambda r: r.fused_score, reverse=True)
        return ranked[:top_k]
