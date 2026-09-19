from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .models import Chunk, RetrievalFilters


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    """Thin wrapper over Qdrant. Uses a server when QDRANT_URL is set,
    otherwise an embedded on-disk instance (single process only)."""

    def __init__(self, dim: int, url: str = "", path: str = "", collection: str = "chunks"):
        if url:
            self.client = QdrantClient(url=url)
        elif path == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=path)
        self.dim = dim
        self.collection = collection
        self._server_mode = bool(url)
        self._ensure_collection(collection)

    def _ensure_collection(self, name: str) -> None:
        if not self.client.collection_exists(name):
            self.client.create_collection(
                name, vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)
            )
            if not self._server_mode:
                return  # embedded Qdrant scans payloads; indexes only matter on a server
            for field_name, schema in [
                ("department", qm.PayloadSchemaType.KEYWORD),
                ("doc_type", qm.PayloadSchemaType.KEYWORD),
                ("language", qm.PayloadSchemaType.KEYWORD),
                ("doc_id", qm.PayloadSchemaType.KEYWORD),
                ("tags", qm.PayloadSchemaType.KEYWORD),
                ("date_ts", qm.PayloadSchemaType.INTEGER),
            ]:
                try:
                    self.client.create_payload_index(name, field_name, schema)
                except Exception:
                    pass  # embedded mode does not need indexes

    # ---- write ----
    def upsert_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        points = [
            qm.PointStruct(id=_point_id(c.chunk_id), vector=vectors[i].tolist(), payload=c.payload())
            for i, c in enumerate(chunks)
        ]
        for i in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[i : i + 128])

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])
            ),
        )

    # ---- read ----
    @staticmethod
    def build_filter(
        filters: RetrievalFilters | None, allowed_departments: list[str] | None
    ) -> qm.Filter | None:
        must: list[Any] = []
        if allowed_departments is not None:
            must.append(qm.FieldCondition(key="department", match=qm.MatchAny(any=allowed_departments)))
        if filters:
            if filters.department:
                must.append(qm.FieldCondition(key="department", match=qm.MatchValue(value=filters.department.lower())))
            if filters.doc_type:
                must.append(qm.FieldCondition(key="doc_type", match=qm.MatchValue(value=filters.doc_type.lower())))
            if filters.language:
                must.append(qm.FieldCondition(key="language", match=qm.MatchValue(value=filters.language.lower())))
            if filters.tags:
                must.append(qm.FieldCondition(key="tags", match=qm.MatchAny(any=[t.lower() for t in filters.tags])))
            if filters.date_from or filters.date_to:
                rng: dict[str, int] = {}
                if filters.date_from:
                    rng["gte"] = int(filters.date_from.strftime("%Y%m%d"))
                if filters.date_to:
                    rng["lte"] = int(filters.date_to.strftime("%Y%m%d"))
                must.append(qm.FieldCondition(key="date_ts", range=qm.Range(**rng)))
        return qm.Filter(must=must) if must else None

    def search(
        self,
        vector: np.ndarray,
        limit: int,
        filters: RetrievalFilters | None = None,
        allowed_departments: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        res = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=limit,
            query_filter=self.build_filter(filters, allowed_departments),
            with_payload=True,
        )
        return [(Chunk(**p.payload), float(p.score)) for p in res.points]

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
