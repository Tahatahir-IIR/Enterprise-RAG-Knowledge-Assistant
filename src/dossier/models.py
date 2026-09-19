from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    department: str = Field(..., description="Owning department, used for access control")
    doc_type: str = Field(..., description="invoice | policy | contract | procedure | report | other")
    language: str | None = Field(None, description="fr | ar | auto-detected when empty")
    doc_date: date | None = Field(None, description="Business date of the document")
    tags: list[str] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    department: str
    doc_type: str
    language: str
    doc_date: str | None = None
    tags: list[str] = []
    pages: int
    chunks: int
    ocr_pages: int = 0


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    page: int
    chunk_index: int
    text: str
    department: str
    doc_type: str
    language: str
    doc_date: str | None = None
    date_ts: int | None = None
    tags: list[str] = []

    def payload(self) -> dict[str, Any]:
        return self.model_dump()


class RetrievalFilters(BaseModel):
    department: str | None = None
    doc_type: str | None = None
    language: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    tags: list[str] = Field(default_factory=list)

    def cache_scope(self) -> str:
        return self.model_dump_json()


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0


class Citation(BaseModel):
    n: int
    source: str
    page: int
    doc_id: str
    excerpt: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    top_k: int | None = None
    use_cache: bool = True


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool
    citations: list[Citation]
    grounding_score: float
    low_confidence: bool
    cache_hit: bool
    latency_ms: int
    language: str
    retrieved: list[dict[str, Any]] = []
