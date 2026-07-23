# Serving layer — working context

Scratch notes kept while building the agent-facing serving layer. Not a spec; a
live log so the work is resumable.

## Goal
Expose the existing `Pipeline` as a knowledge base for AI agents (and general
HTTP callers). User chose **neutral core + BOTH MCP and FastAPI** on one shared,
framework-neutral service module. No re-index per call.

## Key facts about the existing code (verified by reading, 2026-07-23)
- Entry point is `legalrag.core.pipeline.Pipeline(cfg, corpus)`:
  - `build_index()` — builds in-memory index once (call at startup).
  - `answer(query, qid="") -> AnswerBundle(retrieval, answer, report)`.
  - `retrieve(query, qid="") -> RetrievalResult(scored=[ScoredChunk])`.
  - `.corpus` (Corpus), `.graph` (CitationGraph), `.index_id`.
- Build a pipeline: `load_experiment(name)` -> cfg; `load_corpus(cfg.corpus)`.
  (mirrors `cli._build_pipeline`).
- Models (`core/models/results.py`):
  - `GeneratedAnswer(text, citations:[AnswerCitation], abstained, used_chunk_ids)`
  - `AnswerCitation(doc_id, para_ids, quote)` + `.pinpoint_key()`
  - `ScoredChunk(chunk_id, doc_id, score, para_ids, text, rank, components)`
  - `VerificationReport(passed, checks:[CheckResult])` + `.failures()`
- `LegalDocument.pinpoint(pid) -> "Title ¶ n"`; `.title`, `.source.url`.
- Experiments available: e0_naive_baseline, e1_hybrid, e2_rerank, e7_graph,
  e5_bge_small_local, e5_bge_m3_local, e9_local_grounded(_small).
  Default e0 = fully deterministic, no services/keys (good default for smoke).
- pyproject extras convention: heavy deps live in named extras, lazy-imported.

## Design
```
src/legalrag/service/
  __init__.py       exports LegalRAGService + response models
  core.py           NEUTRAL: LegalRAGService.load(experiment); .answer(); .search()
                    shared pydantic response models (JSON-serializable)
                    threading.Lock around pipeline calls (components not proven
                    thread-safe; local single-node usage)
  api.py            FastAPI adapter: POST /answer, /search; GET /health, /graph/{id}
  mcp_server.py     FastMCP adapter: tools legal_answer, legal_search, legal_case
```
- Adapters lazy-import fastapi / mcp so importing core never needs them.
- Response shape (shared): AnswerResponse{query, answer, abstained, verified,
  verifier_flags, citations:[Citation{doc_id,title,pinpoint,para_ids,quote,source_url}],
  index_id, experiment}. SearchResponse{query, passages:[Passage{...}]}.
- Run: `python -m legalrag.service.api` (uvicorn), `python -m legalrag.service.mcp_server` (stdio).
- New extras: `serve = [fastapi, uvicorn]`, `mcp = [mcp]`.

## Why verified quote matters here
The verifier already drops any citation/quote not in the corpus, so every
Citation an agent receives is guaranteed grounded. That's the selling point of
serving THIS pipeline vs a raw vector search.

## Status log — COMPLETE
- [x] Read pipeline/models/corpus/config/pyproject — contracts captured above.
- [x] core.py — LegalRAGService.load/answer/search/case + shared response models, lock.
- [x] api.py — FastAPI: /health /answer /search /case/{doc_id:path}; create_app factory.
- [x] mcp_server.py — FastMCP: legal_answer/legal_search/legal_case; stdio + http transport.
- [x] __init__.py exports neutral core (adapters lazy-import their frameworks).
- [x] pyproject extras: serve=[fastapi,uvicorn], mcp=[mcp]; added to `all`.
- [x] mypy override for the 2 glue modules (untyped-decorator when frameworks absent).
- [x] Smoke: e0 via service — grounded answer, verified citations w/ quotes, search ranked.
- [x] tests/test_service_core.py (5 tests, hermetic on E0) — green.
- [x] Full suite 74 passed, ruff clean, mypy clean (50 files).
- [x] docs/service/README.md — how agents connect (MCP config JSON, HTTP, in-process).

## Verification notes / gotchas
- fastapi/uvicorn/mcp NOT installed in this venv — adapters lazy-import inside
  functions so module import + core smoke + typecheck all pass without them.
  Live adapter run is validated once extras are `pip install`ed (user step).
- uvicorn entry is `--factory legalrag.service.api:create_app` (no eager module
  `app`, so bare import never loads a pipeline).
- Next natural step (not done, not requested): persist index via the existing
  Qdrant adapter so startup skips re-embedding; wire `LEGALRAG_EXPERIMENT` to a
  real jurisdiction corpus once ingested.
