# ADR 0004 — The citation graph is domain-native, and authority ≠ relevance

- **Status:** Accepted
- **Date:** 2026-07-18

## Context

Retrieval over case law can exploit a structure general RAG lacks: judgments
cite judgments, and those citations carry *treatment* (followed / distinguished /
overruled). Two design questions arose while building the E7 rung: (1) where the
graph comes from, and (2) how to use it in retrieval without harm.

## Decision

**(1) Native, not synthesized.** The graph is built directly from resolved
`citations_out` produced by ingestion — real citations with real treatments —
not an LLM-summarized graph (contrast Microsoft GraphRAG). It is exact,
explainable, and free to compute. It powers authority scoring
(PageRank × court level × recency × negative-treatment penalty), "is this still
good law?" signals, and precedent-chain retrieval.

**(2) Graph *injection* is the safe default; authority *reranking* is opt-in.**
Measured on the fixture, a global authority rerank *degraded* aggregate quality:
the correct answer to a query is frequently not the most authoritative document
(statutes, a specific high-court holding, a scanned trial judgment). Relevance is
not authority. So `CitationNeighborExpander` defaults `authority_boost=0` and
instead *injects* citation neighbours strictly below the text hits, filling the
retrieval budget beyond `rerank_top_k` up to `final_k`. This is additive-only: it
recovers precedent-chain documents a text query misses (multi-hop recall@10 on
the fixture 0.67 → 1.0) without ever displacing a direct match, and never
introduces a fabricated citation.

## Consequences

- E7 is a real, non-regressing improvement on the fixture, concentrated exactly
  where expected (multi-hop / precedent queries) — the per-query-type breakdown
  and the adopt-rule (§6.3) working as designed.
- Authority reranking is retained as a knob but explicitly gated to future intent
  routing (E6): apply it only when the information need is about binding
  precedent. This is a finding, recorded here so it is not silently re-enabled.
