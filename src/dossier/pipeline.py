from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from .auth import User
from .bm25_index import BM25Index
from .cache import SemanticCache
from .config import Settings, get_settings
from .embeddings import Embedder, build_embedder
from .generation import generate
from .ingestion import build_chunker, chunk_pages, detect_language, load_document
from .llm import LLM, build_llm
from .models import (
    AskResponse,
    Chunk,
    Citation,
    DocumentInfo,
    DocumentMetadata,
    RetrievalFilters,
)
from .retrieval import HybridRetriever
from .vectorstore import VectorStore

log = logging.getLogger(__name__)

REFUSAL = {
    "fr": "Je n'ai pas trouvé cette information dans les documents auxquels vous avez accès.",
    "ar": "لم أجد هذه المعلومة في المستندات التي يمكنك الاطلاع عليها.",
}


class RagPipeline:
    """Wires ingestion, hybrid retrieval, semantic cache and grounded generation."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        qdrant_path: str | None = None,
    ):
        self.settings = settings or get_settings()
        s = self.settings
        self.embedder = embedder or build_embedder(s)
        self.llm = llm or build_llm(s)
        self.store = VectorStore(
            dim=self.embedder.dim,
            url=s.qdrant_url,
            path=qdrant_path or s.qdrant_path,
            collection=s.chunks_collection,
        )
        s.index_path.mkdir(parents=True, exist_ok=True)
        self.bm25 = BM25Index(s.index_path / "bm25.json")
        self.retriever = HybridRetriever(self.store, self.bm25, self.embedder, rrf_k=s.rrf_k)
        self.cache = SemanticCache(
            self.store.client, self.embedder.dim, s.cache_collection, s.cache_similarity_threshold
        )
        self.chunker = build_chunker(s.chunk_strategy, s.chunk_size, s.chunk_overlap, self.embedder)
        self.registry_path = s.index_path / "documents.json"
        self.documents: dict[str, DocumentInfo] = self._load_registry()

    # ---------------- registry ----------------
    def _load_registry(self) -> dict[str, DocumentInfo]:
        if self.registry_path.exists():
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            return {k: DocumentInfo(**v) for k, v in data.items()}
        return {}

    def _save_registry(self) -> None:
        self.registry_path.write_text(
            json.dumps({k: v.model_dump() for k, v in self.documents.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---------------- ingestion ----------------
    def ingest_file(self, path: str | Path, meta: DocumentMetadata, source_name: str | None = None) -> DocumentInfo:
        path = Path(path)
        source = source_name or path.name
        doc_id = hashlib.sha1(f"{source}|{meta.department}".encode()).hexdigest()[:12]
        if doc_id in self.documents:
            self.delete_document(doc_id)

        pages = load_document(path)
        full_text = "\n".join(p.text for p in pages)
        language = (meta.language or detect_language(full_text)).lower()
        pieces = chunk_pages(pages, self.chunker)

        date_str = meta.doc_date.isoformat() if meta.doc_date else None
        date_ts = int(meta.doc_date.strftime("%Y%m%d")) if meta.doc_date else None
        chunks = [
            Chunk(
                chunk_id=f"{doc_id}:{i}",
                doc_id=doc_id,
                source=source,
                page=page,
                chunk_index=i,
                text=text,
                department=meta.department.lower(),
                doc_type=meta.doc_type.lower(),
                language=language,
                doc_date=date_str,
                date_ts=date_ts,
                tags=[t.lower() for t in meta.tags],
            )
            for i, (page, text) in enumerate(pieces)
        ]
        if chunks:
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            self.store.upsert_chunks(chunks, vectors)
            self.bm25.add(chunks)

        info = DocumentInfo(
            doc_id=doc_id,
            source=source,
            department=meta.department.lower(),
            doc_type=meta.doc_type.lower(),
            language=language,
            doc_date=date_str,
            tags=[t.lower() for t in meta.tags],
            pages=len(pages),
            chunks=len(chunks),
            ocr_pages=sum(1 for p in pages if p.ocr),
        )
        self.documents[doc_id] = info
        self._save_registry()
        log.info("ingested %s: %d pages, %d chunks (%s)", source, info.pages, info.chunks, language)
        return info

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        self.store.delete_document(doc_id)
        self.bm25.delete_document(doc_id)
        self.cache.invalidate_document(doc_id)
        del self.documents[doc_id]
        self._save_registry()
        return True

    def list_documents(self, user: User | None = None) -> list[DocumentInfo]:
        docs = list(self.documents.values())
        if user is not None:
            docs = [d for d in docs if user.can_access(d.department)]
        return sorted(docs, key=lambda d: d.source)

    # ---------------- question answering ----------------
    def ask(
        self,
        question: str,
        user: User,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
        use_cache: bool = True,
        include_retrieved: bool = True,
    ) -> AskResponse:
        t0 = time.perf_counter()
        s = self.settings
        filters = filters or RetrievalFilters()
        allowed = user.allowed_departments()
        lang = detect_language(question)
        qvec = self.embedder.embed_query(question)
        scope = SemanticCache.scope_hash(user.scope_key(), filters.cache_scope())

        if use_cache and s.cache_enabled:
            hit = self.cache.lookup(qvec, scope)
            if hit:
                resp = AskResponse(**hit["response"])
                resp.cache_hit = True
                resp.question = question
                resp.latency_ms = int((time.perf_counter() - t0) * 1000)
                return resp

        retrieved = self.retriever.retrieve(
            question, top_k or s.top_k, s.candidates, filters, allowed, query_vector=qvec
        )
        best_dense = max((r.dense_score for r in retrieved), default=0.0)
        has_lexical = any(r.bm25_score > 0 for r in retrieved)
        gated = not retrieved or (best_dense < s.min_dense_score and not has_lexical)

        if gated:
            gen_answer, found, citations, grounding, low = REFUSAL[lang], False, [], 1.0, False
        else:
            g = generate(self.llm, question, retrieved)
            if g.found:
                gen_answer, found, citations, grounding, low = g.answer, True, g.citations, g.grounding_score, g.low_confidence
            else:
                gen_answer, found, citations, grounding, low = REFUSAL[lang], False, [], 1.0, False

        resp = AskResponse(
            question=question,
            answer=gen_answer,
            found=found,
            citations=citations,
            grounding_score=grounding,
            low_confidence=low,
            cache_hit=False,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            language=lang,
            retrieved=[
                {
                    "source": r.chunk.source,
                    "page": r.chunk.page,
                    "department": r.chunk.department,
                    "dense_score": round(r.dense_score, 4),
                    "bm25_score": round(r.bm25_score, 4),
                    "fused_score": round(r.fused_score, 5),
                    "preview": r.chunk.text[:160],
                }
                for r in retrieved
            ]
            if include_retrieved
            else [],
        )
        if use_cache and s.cache_enabled and found and not low:
            payload = resp.model_dump()
            self.cache.store(qvec, scope, question, payload, doc_ids=[c.doc_id for c in citations])
        return resp

    def close(self) -> None:
        self.store.close()


def citations_to_dicts(cits: list[Citation]) -> list[dict]:
    return [c.model_dump() for c in cits]
