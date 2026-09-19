from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Okapi

from .models import Chunk, RetrievalFilters

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_AR_DIACRITICS = re.compile(r"[ً-ْـ]")

# Minimal stopword lists; enough to keep BM25 from ranking on "de"/"في".
_STOP = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "au", "aux", "à", "a",
    "que", "qui", "pour", "par", "sur", "dans", "est", "sont", "ce", "cette", "ces", "se",
    "quel", "quelle", "quels", "quelles", "ou", "the", "of", "and", "to", "is",
    "في", "من", "على", "إلى", "عن", "ما", "هو", "هي", "هل", "أن", "مع", "ال", "و",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))  # strip accents
    text = _AR_DIACRITICS.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    return text.lower()


def tokenize(text: str) -> list[str]:
    toks = _TOKEN_RE.findall(normalize(text))
    return [t.lstrip("ال") if t.startswith("ال") and len(t) > 4 else t for t in toks if t not in _STOP]


class BM25Index:
    """In-memory BM25 over chunk payloads, persisted as JSON next to the vector index."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.chunks = [Chunk(**c) for c in data]
            self._rebuild()

    def _rebuild(self) -> None:
        corpus = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def add(self, chunks: list[Chunk]) -> None:
        known = {c.chunk_id for c in self.chunks}
        self.chunks.extend(c for c in chunks if c.chunk_id not in known)
        self._rebuild()
        self.save()

    def delete_document(self, doc_id: str) -> None:
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
        self._rebuild()
        self.save()

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps([c.model_dump() for c in self.chunks], ensure_ascii=False), encoding="utf-8"
            )

    @staticmethod
    def _passes(c: Chunk, filters: RetrievalFilters | None, allowed: list[str] | None) -> bool:
        if allowed is not None and c.department not in allowed:
            return False
        if not filters:
            return True
        if filters.department and c.department != filters.department.lower():
            return False
        if filters.doc_type and c.doc_type != filters.doc_type.lower():
            return False
        if filters.language and c.language != filters.language.lower():
            return False
        if filters.tags and not set(t.lower() for t in filters.tags) & set(c.tags):
            return False
        if (filters.date_from or filters.date_to) and c.date_ts is None:
            return False
        if filters.date_from and c.date_ts < int(filters.date_from.strftime("%Y%m%d")):
            return False
        if filters.date_to and c.date_ts > int(filters.date_to.strftime("%Y%m%d")):
            return False
        return True

    def search(
        self,
        query: str,
        limit: int,
        filters: RetrievalFilters | None = None,
        allowed_departments: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[tuple[Chunk, float]] = []
        for i in ranked:
            if scores[i] <= 0:
                break
            c = self.chunks[i]
            if self._passes(c, filters, allowed_departments):
                out.append((c, float(scores[i])))
                if len(out) >= limit:
                    break
        return out

    def __len__(self) -> int:
        return len(self.chunks)
