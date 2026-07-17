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

## Quickstart

```bash
python -m venv .venv && . .venv/Scripts/activate   # or bin/activate on *nix
pip install -e ".[dev]"

python evalsets/fixture/build_fixture.py           # (re)generate the fixture corpus
legalrag components                                # list every registered component
legalrag eval --experiment e0_naive_baseline       # tracked E0 run on the fixture
legalrag eval --experiment e1_hybrid               # same, hybrid retrieval (a config diff)
legalrag eval --experiment e7_graph                # + citation-graph expansion (E7)
legalrag query "damages for remoteness" -e e0_naive_baseline
legalrag graph "xx/high/2015-201/v1"               # authority + treatment of a case
```

No API keys or services are needed to run the above — the baseline uses
deterministic in-process components ([ADR 0003](docs/adr/0003-deterministic-smoke-path.md)).
Real embedders/LLMs/stores register the same way and are selected in config.

## Status — P0 complete; citation-graph rung (E7) landed

**P0 (Foundations):** package scaffold + CI, canonical document model, component
registry + stage interfaces, config-driven experiment runner, evaluation harness
(retrieval + generation metrics, per-query-type breakdown, run tracking),
deterministic verification stack, 14-doc fixture corpus (incl. adversarial
cases), and **E0 end-to-end** (recall@10 ≈ 0.96, **hallucinated-citation rate 0**).
Infra dev stack (Qdrant · Postgres · MinIO · MLflow · Phoenix) in [`infra/`](infra/).

**Citation graph + graph-expanded retrieval (E7):** domain-native citation graph
(PageRank + court-level + recency + negative-treatment authority scoring,
overruled/"still good law?" detection), and additive graph-neighbour injection
that lifts multi-hop recall@10 from **0.67 → 1.0** with no aggregate regression
and zero fabricated citations ([ADR 0004](docs/adr/0004-citation-graph-native-not-synthesized.md)).
Real embedder adapters (Voyage · sentence-transformers/BGE-M3) registered as
opt-in components. **48 tests, ruff + mypy(strict) green.**

Next: P1 on `jurisdiction/us` — CAP/CourtListener connectors, CLERC-dev slice,
eyecite adapter (feeding this graph with real citations), then the rest of the
E1–E10 ladder. See the [roadmap](docs/IMPLEMENTATION_PLAN.md#10-phased-roadmap).

> ⚖️ Research software. Outputs are legal *information* with verified sources, never legal advice.
