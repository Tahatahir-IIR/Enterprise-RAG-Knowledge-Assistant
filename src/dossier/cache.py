"""Semantic cache: near-duplicate questions skip retrieval and the LLM.

Entries are scoped by (user access scope, filters) so a cached finance answer is
never served to an HR user, and by embedding similarity so "montant total de la
facture F-2025-014 ?" also hits for "quel est le total de la facture F-2025-014".
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


class SemanticCache:
    def __init__(self, client: QdrantClient, dim: int, collection: str = "semantic_cache", threshold: float = 0.95):
        self.client, self.collection, self.threshold = client, collection, threshold
        if not client.collection_exists(collection):
            client.create_collection(
                collection, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE)
            )
        self.hits = 0
        self.misses = 0

    @staticmethod
    def scope_hash(user_scope: str, filter_scope: str) -> str:
        return hashlib.sha1(f"{user_scope}|{filter_scope}".encode()).hexdigest()

    def lookup(self, vector: np.ndarray, scope: str) -> dict[str, Any] | None:
        res = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=1,
            query_filter=qm.Filter(must=[qm.FieldCondition(key="scope", match=qm.MatchValue(value=scope))]),
            with_payload=True,
        )
        if res.points and res.points[0].score >= self.threshold:
            self.hits += 1
            return dict(res.points[0].payload)
        self.misses += 1
        return None

    def store(
        self, vector: np.ndarray, scope: str, question: str, response: dict[str, Any], doc_ids: list[str] | None = None
    ) -> None:
        payload = {
            "scope": scope,
            "question": question,
            "created": int(time.time()),
            "doc_ids": sorted(set(doc_ids or [])),
            "response": response,
        }
        self.client.upsert(
            self.collection,
            points=[qm.PointStruct(id=str(uuid.uuid4()), vector=vector.tolist(), payload=payload)],
        )

    def invalidate_document(self, doc_id: str) -> None:
        """Drop every cached answer that cited a deleted/updated document."""
        self.client.delete(
            self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(key="doc_ids", match=qm.MatchValue(value=doc_id))])
            ),
        )

    def clear(self) -> None:
        self.client.delete(self.collection, points_selector=qm.Filter(must=[]))
        self.hits = self.misses = 0

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "entries": int(self.client.count(self.collection, exact=True).count),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "threshold": self.threshold,
        }
