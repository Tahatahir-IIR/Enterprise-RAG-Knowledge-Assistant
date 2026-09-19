"""End-to-end evaluation on eval/questions.json against the current index.

Metrics
  retrieval hit@k   expected source appears in the top-k retrieved chunks
  MRR               mean reciprocal rank of the first expected source
  refusal accuracy  answerable questions answered, unanswerable ones refused
  hallucination     answered although unanswerable (lower is better)
  keyword accuracy  expected value present in the answer (answered questions)
  citation validity every citation points at an expected source
  latency           median cold latency and median semantic-cache latency

Run as admin so access control does not hide documents:
  python scripts/evaluate.py            (uses the embedded index; stop the API first)
  python scripts/evaluate.py --out eval/results.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from dossier.auth import User
from dossier.pipeline import RagPipeline


def evaluate(p: RagPipeline, questions: list[dict], k: int) -> dict:
    admin = User("eval", "eval", "admin", [])
    p.cache.clear()
    rows, hits, rr, cold, warm = [], [], [], [], []
    tp = fp = tn = fn = 0
    kw_ok = kw_total = 0
    cit_ok = cit_total = 0

    for q in questions:
        r = p.ask(q["question"], admin, top_k=k, use_cache=True)
        cold.append(r.latency_ms)
        r2 = p.ask(q["question"], admin, top_k=k, use_cache=True)
        if r2.cache_hit:
            warm.append(r2.latency_ms)

        retrieved_sources = [x["source"] for x in r.retrieved]
        if q["answerable"]:
            rank = next((i + 1 for i, s in enumerate(retrieved_sources) if s in q["sources"]), 0)
            hits.append(1 if rank else 0)
            rr.append(1 / rank if rank else 0.0)
            if r.found:
                tp += 1
                kw_total += 1
                kw_ok += int(any(kw.lower() in r.answer.lower() for kw in q["keywords"]))
                for c in r.citations:
                    cit_total += 1
                    cit_ok += int(c.source in q["sources"])
            else:
                fn += 1
        else:
            if r.found:
                fp += 1
            else:
                tn += 1
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "answerable": q["answerable"],
                "found": r.found,
                "hit": bool(retrieved_sources) and any(s in q["sources"] for s in retrieved_sources),
                "answer": r.answer[:200],
                "citations": [f"{c.source}#p{c.page}" for c in r.citations],
                "grounding": r.grounding_score,
                "latency_ms": r.latency_ms,
            }
        )

    n_ans = tp + fn
    n_unans = fp + tn
    return {
        "n": len(questions),
        "top_k": k,
        "llm": p.llm.name,
        "embedding_dim": p.embedder.dim,
        "chunk_strategy": p.chunker.name,
        "retrieval_hit_at_k": round(sum(hits) / len(hits), 3) if hits else None,
        "mrr": round(sum(rr) / len(rr), 3) if rr else None,
        "answered_when_answerable": round(tp / n_ans, 3) if n_ans else None,
        "refused_when_unanswerable": round(tn / n_unans, 3) if n_unans else None,
        "hallucination_rate": round(fp / n_unans, 3) if n_unans else None,
        "keyword_accuracy": round(kw_ok / kw_total, 3) if kw_total else None,
        "citation_validity": round(cit_ok / cit_total, 3) if cit_total else None,
        "median_latency_ms": int(statistics.median(cold)) if cold else None,
        "median_cache_latency_ms": int(statistics.median(warm)) if warm else None,
        "cache_hit_rate_on_repeat": round(len(warm) / len(questions), 3),
        "rows": rows,
    }


def to_markdown(res: dict) -> str:
    lines = [
        f"# Evaluation ({res['n']} questions, top_k={res['top_k']})",
        "",
        f"LLM: `{res['llm']}` · embedding dim: {res['embedding_dim']} · chunking: `{res['chunk_strategy']}`",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key in [
        "retrieval_hit_at_k", "mrr", "answered_when_answerable", "refused_when_unanswerable",
        "hallucination_rate", "keyword_accuracy", "citation_validity",
        "median_latency_ms", "median_cache_latency_ms", "cache_hit_rate_on_repeat",
    ]:
        lines.append(f"| {key} | {res[key]} |")
    lines += ["", "## Per question", "", "| # | Q | answerable | found | hit | grounding | citations |", "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        lines.append(
            f"| {r['id']} | {r['question'][:60]} | {r['answerable']} | {r['found']} | {r['hit']} | {r['grounding']} | {', '.join(r['citations'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="eval/results.md")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    p = RagPipeline()
    try:
        if not p.documents:
            raise SystemExit("Index is empty. Run scripts/generate_dataset.py then scripts/ingest.py first.")
        res = evaluate(p, questions, args.top_k)
    finally:
        p.close()

    md = to_markdown(res)
    Path(args.out).write_text(md, encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(md.splitlines()[:18]))
    print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
