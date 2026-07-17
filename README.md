# Legal-RAG

Research platform for **advanced Retrieval-Augmented Generation over judicial corpora** — grounded, citation-verified question answering and research over large-scale case law and statutes.

**Research-first design:** every pipeline stage (chunker, embedder, retriever, reranker, generator, verifier) is a swappable registered component; a full RAG system is an experiment config, and every run is tracked and reproducible. The goal is to test, combine, and publish advanced RAG methodologies on legal corpora — and drive hallucinated citations to zero.

## Tracks & branches

| Branch | Track | Status |
|---|---|---|
| `main` | Shared engine core, canonical document model, eval harness, docs | Everything merges back here |
| `jurisdiction/us` | Prototype pipeline on open US data (Caselaw Access Project, CourtListener) + public benchmarks (CLERC, COLIEE) | **Active** |
| `jurisdiction/pakistan` | Research target: Pakistan's judiciary — scrapers, `pk-cite` citation extractor, **PakLegalQA** benchmark, US→PK transfer study | Groundwork |
| `jurisdiction/multi` | Multi-jurisdiction adapter architecture, cross-jurisdiction citation graph | Design notes |

Core rule: core code lands on `main`; jurisdiction branches hold only jurisdiction-specific work and sync from `main` at every phase boundary.

## Documentation

- **[Master Implementation Plan](docs/IMPLEMENTATION_PLAN.md)** — architecture, research questions, experiment ladder (E0–E10), phased roadmap (P0–P5), risks
- [US annex](docs/jurisdictions/us.md) · [Pakistan annex](docs/jurisdictions/pakistan.md) · [Multi-jurisdiction annex](docs/jurisdictions/multi.md)

## Status

Planning phase (P0 pending): repo scaffolding, docker dev stack (Qdrant · Postgres · MinIO · MLflow · Phoenix), canonical model + component registry, fixture corpus, E0 baseline. See [Immediate next actions](docs/IMPLEMENTATION_PLAN.md#13-immediate-next-actions-p0-kickoff).

> ⚖️ Research software. Outputs are legal *information* with verified sources, never legal advice.
