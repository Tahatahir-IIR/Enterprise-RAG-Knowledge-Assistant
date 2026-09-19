"""Compare chunking strategies on retrieval quality only (no LLM involved).

Builds a throw-away index per strategy from data/raw/manifest.json, then
measures hit@k and MRR of the expected source over eval/questions.json.

  python scripts/compare_chunking.py --sizes 400 800 --out eval/chunking.md
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import date
from pathlib import Path

from dossier.auth import User
from dossier.config import get_settings
from dossier.embeddings import build_embedder
from dossier.llm import StubLLM
from dossier.models import DocumentMetadata, RetrievalFilters
from dossier.pipeline import RagPipeline


def run(strategy: str, size: int, overlap: int, manifest: list[dict], base: Path, questions: list[dict], k: int, embedder) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"chunk_{strategy}_"))
    settings = get_settings().model_copy(
        update={
            "chunk_strategy": strategy,
            "chunk_size": size,
            "chunk_overlap": overlap,
            "qdrant_url": "",
            "qdrant_path": str(tmp / "qdrant"),
            "data_dir": str(tmp / "data"),
            "cache_enabled": False,
        }
    )
    p = RagPipeline(settings=settings, embedder=embedder, llm=StubLLM())
    try:
        for e in manifest:
            p.ingest_file(
                base / e["file"],
                DocumentMetadata(
                    department=e["department"],
                    doc_type=e["doc_type"],
                    language=e.get("language"),
                    doc_date=date.fromisoformat(e["doc_date"]) if e.get("doc_date") else None,
                    tags=e.get("tags", []),
                ),
            )
        admin = User("eval", "eval", "admin", [])
        hits, rr = [], []
        for q in questions:
            if not q["answerable"]:
                continue
            res = p.retriever.retrieve(q["question"], top_k=k, candidates=settings.candidates, filters=RetrievalFilters(), allowed_departments=admin.allowed_departments())
            sources = [r.chunk.source for r in res]
            rank = next((i + 1 for i, s in enumerate(sources) if s in q["sources"]), 0)
            hits.append(1 if rank else 0)
            rr.append(1 / rank if rank else 0.0)
        n_chunks = p.store.count()
        avg_len = sum(len(c.text) for c in p.bm25.chunks) / max(1, len(p.bm25.chunks))
        return {
            "strategy": strategy,
            "size": size,
            "overlap": overlap,
            "chunks": n_chunks,
            "avg_chars": int(avg_len),
            f"hit@{k}": round(sum(hits) / len(hits), 3),
            "mrr": round(sum(rr) / len(rr), 3),
        }
    finally:
        p.close()
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/raw/manifest.json")
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--sizes", type=int, nargs="+", default=[400, 800])
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="eval/chunking.md")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    embedder = build_embedder()

    results = []
    for strategy in ["fixed", "recursive", "semantic"]:
        for size in args.sizes:
            r = run(strategy, size, args.overlap, manifest, manifest_path.parent, questions, args.top_k, embedder)
            results.append(r)
            print(r)

    keys = list(results[0].keys())
    md = ["# Chunking strategy comparison", "", "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in results:
        md.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
    Path(args.out).write_text("\n".join(md), encoding="utf-8")
    print(f"\nWritten {args.out}")


if __name__ == "__main__":
    main()
