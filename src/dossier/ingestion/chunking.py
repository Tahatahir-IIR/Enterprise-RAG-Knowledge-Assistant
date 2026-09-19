"""Three chunking strategies, compared in scripts/compare_chunking.py.

fixed     : sliding character window. Baseline, ignores structure.
recursive : splits on paragraphs, then sentences, then words (LangChain splitter).
            Keeps headings with their body most of the time. Default.
semantic  : splits into sentences, embeds them, and cuts where the cosine
            similarity between neighbouring sentences drops. Topic-coherent
            chunks at the price of an embedding pass during ingestion.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .loaders import Page

if TYPE_CHECKING:
    from ..embeddings import Embedder

# Sentence boundaries for French and Arabic (Arabic uses ؟ and ۔ in addition to . ! ?)
_SENT_RE = re.compile(r"(?<=[.!?؟۔])\s+|\n{2,}")


class Chunker(ABC):
    name: str

    @abstractmethod
    def split(self, text: str) -> list[str]: ...


class FixedChunker(Chunker):
    name = "fixed"

    def __init__(self, size: int = 800, overlap: int = 120):
        self.size, self.overlap = size, max(0, min(overlap, size - 1))

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        step = self.size - self.overlap
        out = [text[i : i + self.size].strip() for i in range(0, len(text), step)]
        return [c for c in out if c]


class RecursiveChunker(Chunker):
    name = "recursive"

    def __init__(self, size: int = 800, overlap: int = 120):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "؟ ", "! ", "? ", "، ", ", ", " ", ""],
        )

    def split(self, text: str) -> list[str]:
        return [c.strip() for c in self.splitter.split_text(text) if c.strip()]


class SemanticChunker(Chunker):
    name = "semantic"

    def __init__(self, embedder: Embedder, size: int = 800, percentile: float = 25.0):
        self.embedder = embedder
        self.max_size = size
        self.percentile = percentile

    def split(self, text: str) -> list[str]:
        sentences = [s.strip() for s in _SENT_RE.split(text) if s and s.strip()]
        if len(sentences) <= 1:
            return sentences
        vecs = self.embedder.embed_documents(sentences)
        sims = np.array(
            [float(np.dot(vecs[i], vecs[i + 1])) for i in range(len(sentences) - 1)]
        )
        threshold = float(np.percentile(sims, self.percentile)) if len(sims) > 2 else -1.0
        chunks, current = [], [sentences[0]]
        for i in range(1, len(sentences)):
            drop = sims[i - 1] < threshold
            too_long = len(" ".join(current)) + len(sentences[i]) > self.max_size
            if drop or too_long:
                chunks.append(" ".join(current))
                current = [sentences[i]]
            else:
                current.append(sentences[i])
        if current:
            chunks.append(" ".join(current))
        return chunks


def build_chunker(
    strategy: str, size: int = 800, overlap: int = 120, embedder: Embedder | None = None
) -> Chunker:
    if strategy == "fixed":
        return FixedChunker(size, overlap)
    if strategy == "recursive":
        return RecursiveChunker(size, overlap)
    if strategy == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking needs an embedder")
        return SemanticChunker(embedder, size)
    raise ValueError(f"unknown chunk strategy {strategy!r}")


def chunk_pages(pages: list[Page], chunker: Chunker) -> list[tuple[int, str]]:
    """Return (page_number, chunk_text) pairs, chunking page by page so every
    chunk maps to exactly one citable page."""
    out: list[tuple[int, str]] = []
    for page in pages:
        for piece in chunker.split(page.text):
            out.append((page.number, piece))
    return out
