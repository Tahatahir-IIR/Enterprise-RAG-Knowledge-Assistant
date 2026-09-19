"""Prompting, citation parsing and a post-hoc groundedness check.

Three defences against hallucination live here:
1. The prompt forbids outside knowledge and requires a [n] citation per claim.
2. The model may answer NOT_FOUND; the pipeline turns that into an honest refusal.
3. After generation, each answer sentence is checked for lexical support in the
   cited chunks. Unsupported answers are flagged low_confidence in the API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .bm25_index import tokenize
from .llm import LLM, NOT_FOUND_TOKEN
from .models import Citation, RetrievedChunk

SYSTEM_PROMPT = """You are Dossier, an internal assistant for a company in Morocco.
You answer questions ONLY from the numbered SOURCES provided. Rules:
- Every factual sentence must end with the citation(s) of the source(s) it comes from, like [1] or [2][3].
- Never use outside knowledge. Never guess amounts, dates, names or article numbers.
- If the sources do not contain the answer, reply with exactly: NOT_FOUND
- Answer in the language of the question (French or Arabic). Keep the answer short and precise.
- Quote numbers exactly as written in the sources (e.g. 12 500,00 MAD)."""

_CITE_RE = re.compile(r"\[(\d+)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?؟])\s+|\n+")


@dataclass
class Generation:
    answer: str
    found: bool
    citations: list[Citation] = field(default_factory=list)
    grounding_score: float = 1.0
    low_confidence: bool = False


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    parts = ["SOURCES:"]
    for i, rc in enumerate(chunks, start=1):
        c = rc.chunk
        parts.append(f"[{i}] ({c.source}, page {c.page})\n{c.text.strip()}")
    parts.append(f"QUESTION: {question}")
    return "\n\n".join(parts)


def _support(sentence: str, chunk_texts: list[str]) -> float:
    toks = set(tokenize(sentence))
    if not toks:
        return 1.0
    best = 0.0
    for text in chunk_texts:
        overlap = len(toks & set(tokenize(text))) / len(toks)
        best = max(best, overlap)
    return best


def groundedness(answer: str, chunks: list[RetrievedChunk], cited: set[int]) -> float:
    """Share of answer sentences whose content words appear in a cited chunk.

    Cheap, model-free, and language-agnostic. Not a substitute for an LLM judge,
    but it catches the common failure where the model adds an unsupported claim.
    """
    texts = [chunks[n - 1].chunk.text for n in cited if 0 < n <= len(chunks)] or [
        rc.chunk.text for rc in chunks
    ]
    sentences = [s for s in _SENT_SPLIT.split(_CITE_RE.sub("", answer)) if len(s.strip()) > 3]
    if not sentences:
        return 1.0
    scores = [_support(s, texts) for s in sentences]
    return sum(1 for s in scores if s >= 0.5) / len(scores)


def generate(llm: LLM, question: str, chunks: list[RetrievedChunk], support_threshold: float = 0.5) -> Generation:
    if not chunks:
        return Generation(answer=NOT_FOUND_TOKEN, found=False, grounding_score=1.0)
    raw = llm.complete(SYSTEM_PROMPT, build_user_prompt(question, chunks)).strip()
    if NOT_FOUND_TOKEN in raw.upper()[:40] or not raw:
        return Generation(answer=NOT_FOUND_TOKEN, found=False)

    cited_ns = {int(n) for n in _CITE_RE.findall(raw) if 0 < int(n) <= len(chunks)}
    if not cited_ns:
        # An answer without a single valid citation violates the contract; treat it as
        # a refusal rather than presenting unverifiable text (models often phrase
        # "no answer in the sources" in prose instead of emitting NOT_FOUND).
        return Generation(answer=NOT_FOUND_TOKEN, found=False)
    citations = [
        Citation(
            n=n,
            source=chunks[n - 1].chunk.source,
            page=chunks[n - 1].chunk.page,
            doc_id=chunks[n - 1].chunk.doc_id,
            excerpt=chunks[n - 1].chunk.text[:300],
        )
        for n in sorted(cited_ns)
    ]
    score = groundedness(raw, chunks, cited_ns)
    low = score < support_threshold or not citations
    return Generation(answer=raw, found=True, citations=citations, grounding_score=round(score, 3), low_confidence=low)
