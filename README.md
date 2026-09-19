# Enterprise RAG Knowledge Assistant

An internal document assistant for companies operating in Morocco, in French and Arabic.
Final-year (Data & AI) personal project.

You upload business documents (invoices, HR policies, contracts, procedures), ask a question
in either language, and get a short answer with a citation to the document and page. If the
information is not in the documents, the system says so instead of making something up.

I wanted a project that looks like what a Moroccan IT department would actually ask for,
not a chatbot demo: documents in two languages, access rights per department, and above all
a way to measure whether the model hallucinates.

## How it works

```
question ──► semantic cache ──(hit)──► answer in a few ms
                 │ (miss)
                 ▼
   dense search (Qdrant, multilingual embeddings)        ┐
   + access-control filter + business metadata filters   ├──► RRF fusion ──► top-k passages
   BM25 keyword search (invoice numbers, ICE, amounts)   ┘            │
                                                                      ▼
                                              score too low? ──► honest refusal
                                                                      │
                                                                      ▼
                          local LLM (Ollama): "answer only from the SOURCES,
                          cite [n] on every sentence, otherwise reply NOT_FOUND"
                                                                      │
                                                                      ▼
                          citation parsing + groundedness score ──► answer + sources
```

Design choices I can defend:

- **Hybrid search.** Embeddings understand that "when do we have to pay" should find
  "délai de paiement", but they easily confuse `F-2025-007` with `F-2025-001`. BM25 handles
  the second case. The two rankings are merged with Reciprocal Rank Fusion so there are no
  weights to tune by hand.
- **Filters run inside the search, not after it.** Filtering the top-k afterwards can return
  zero results for a restricted user and leaks ranking information. Qdrant applies metadata
  filters during the ANN search.
- **The semantic cache lives in Qdrant.** One dependency fewer than Redis, and looking up a
  similar question is a vector search anyway. Every entry is bound to the user's access
  scope, otherwise an HR user could receive an answer cached by finance.
- **Hallucination control is layered.** A relevance threshold before calling the model, a
  prompt restricted to the sources with mandatory citations, `NOT_FOUND` as an allowed
  answer, and a lexical check after generation. An answer without a valid citation is
  treated as a refusal.
- **Three chunking strategies** (fixed, recursive, semantic) with a comparison script,
  because it is the question everyone asks about a RAG system and I wanted a measured
  answer rather than an opinion.

## Results

40 questions (French and Arabic, 6 of them with no answer in the corpus) over a synthetic
corpus of 19 documents. Chat model qwen3.5 4B through Ollama, multilingual-e5-small
embeddings on CPU.

| Metric | Value |
|---|---|
| Expected document in the top 5 | 100 % |
| Answerable questions actually answered | 34 / 34 |
| Unanswerable questions actually refused | 6 / 6 |
| Answers containing the expected value | 94 % |
| Citations pointing at the right document | 98 % |
| Median latency, cold | 3.6 s |
| Median latency, semantic cache hit | 29 ms |

All 8 Arabic questions were answered with valid citations, including those whose answer only
existed in a French document. The full report is in [eval/results.md](eval/results.md) and
the chunking comparison in [eval/chunking.md](eval/chunking.md). On this small corpus all
three strategies find the right document; recursive chunking at 800 characters produces
40 % fewer passages than fixed chunking for the same mean rank, so fewer tokens per call,
which is why it is the default.

## What I learned building it

- An 8 GB GPU does not hold a 4B chat model and bge-m3 at the same time. Ollama swaps them
  on every request (10 to 17 s) and the embedding runner sometimes dies with a 500. I added
  retries with backoff, then moved embeddings to the CPU with sentence-transformers. That is
  the setup behind the numbers above.
- Small models do not always respect the `NOT_FOUND` token: they write "لا توجد إجابة في
  المصادر" in prose instead. Hence the rule: no citation, no answer.
- Sending Arabic through `curl` from Git Bash on Windows corrupts the encoding. Tests go
  through Python `requests`.
- Embedded Qdrant accepts a single process. Stop the API before running the ingestion
  scripts, or ingest through the API with `--api`.
- Generating Arabic PDFs with fpdf2 needs a font plus a bidirectional shaping engine, so the
  Arabic HR policy is Markdown, which the ingestion handles the same way.

## Running it

Install:

```bash
pip install -e ".[dev,data,ui,embeddings]"
cp .env.example .env
```

With Ollama (recommended, the setup used for the results):

```bash
ollama pull qwen3.5:4b
# in .env: LLM_BACKEND=ollama, OLLAMA_MODEL=qwen3.5:4b, EMBEDDING_BACKEND=sentence-transformers
python scripts/generate_dataset.py     # synthetic corpus -> data/raw
python scripts/ingest.py               # build the index
uvicorn dossier.api:app --port 8000    # API, docs at http://localhost:8000
streamlit run ui/app.py                # UI at http://localhost:8501
```

Demo keys are in [config/users.yaml](config/users.yaml): `admin-key` sees everything;
`finance-key`, `hr-key` and `procurement-key` only see their own department.

Without any model (tests, CI): `LLM_BACKEND=stub EMBEDDING_BACKEND=hash MIN_DENSE_SCORE=0.15`.
The stub extracts sentences instead of generating; it is a plumbing test, not a quality test.

Everything in containers: `docker compose up --build` starts Qdrant, Ollama, the API and the UI.

Evaluation and chunking comparison (with the API stopped):

```bash
python scripts/evaluate.py
python scripts/compare_chunking.py --sizes 400 800
```

## Example call

```bash
curl -X POST http://localhost:8000/ask -H "X-API-Key: finance-key" -H "Content-Type: application/json" \
  -d '{"question": "Quel est le montant TTC de la facture F-2025-003 ?", "filters": {"doc_type": "invoice"}}'
```

```json
{
  "answer": "Le montant TTC de la facture F-2025-003 est de 20 400,00 MAD [1].",
  "found": true,
  "citations": [{"n": 1, "source": "facture_F-2025-003.pdf", "page": 1, "excerpt": "FACTURE N° F-2025-003 ..."}],
  "grounding_score": 1.0,
  "low_confidence": false,
  "cache_hit": false,
  "latency_ms": 3591
}
```

An HR user asking the same question gets a refusal: the invoice is never retrieved for them,
and the answer cached for finance is not served to them either.

## Code layout

The Python package is called `dossier` (French for a file of documents).

```
src/dossier/
  api.py          FastAPI routes (upload, ask, documents, cache)
  pipeline.py     ingestion and answering, wires everything together
  ingestion/      PDF/OCR/text loading, language detection, chunking
  embeddings.py   sentence-transformers, Ollama or hash (tests)
  vectorstore.py  Qdrant with business filters and access control
  bm25_index.py   BM25 with FR/AR normalisation (accents, diacritics, ال prefix)
  retrieval.py    RRF fusion
  generation.py   prompt, citations, groundedness score
  cache.py        semantic cache scoped per access level
  auth.py         API keys, roles, departments
scripts/          corpus generation, ingestion, evaluation, chunking comparison
eval/             the 40 questions and the reports
tests/            26 tests (chunking, pipeline, access control, cache, API)
```

## Next steps

- LLM-as-judge faithfulness (RAGAS) alongside the lexical score.
- A reranker (bge-reranker-v2-m3) between fusion and generation.
- Structured extraction of invoice fields (supplier, ICE, HT/TVA/TTC).
- SSO instead of API keys, and an audit log of questions.
- Darija questions with a Darija-tuned model (Atlas-Chat).

The corpus is entirely synthetic (fictional company "Maghreb Industries SA").
MIT license.
