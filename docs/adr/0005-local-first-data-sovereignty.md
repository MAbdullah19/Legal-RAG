# ADR 0005 — Local-first (data-sovereignty) default posture

- **Status:** Accepted
- **Date:** 2026-07-20
- **Supersedes (in part):** the "Anthropic API default" phrasing of the hybrid LLM
  posture in the master plan (§3.2 principle 4, §5.6). The abstraction is unchanged;
  the *default* flips.

## Context

The master plan framed the hybrid LLM posture as "hosted API by default, local as
the sovereignty path." That default is wrong for this project's real target. Legal
corpora — casefiles, unpublished judgments, client material, and the Pakistan track's
scan-heavy court data — are **confidential**. Sending document text or queries to a
third-party embedding/generation API means the confidential corpus leaves the
machine and may be logged, cached, or retained by the provider. For the intended
users (courts, chambers, a university research group handling real filings), that is
a non-starter, independent of quality.

The engine was already built to make this a config choice, not a rewrite: every
model-touching stage (embedder, reranker, generator, verifier) is a registered
Protocol implementation, and both local and hosted implementations exist.

## Decision

Make **fully-local the default posture**. Hosted APIs become the opt-in path, not
the reverse.

- **Embeddings:** BGE-M3 via `sentence_transformer` (local, multilingual → also the
  Urdu path for PK). Voyage/OpenAI remain registered for the E5 bake-off, opt-in.
- **Reranking:** `bge_reranker` (BAAI/bge-reranker-v2-m3) cross-encoder, local.
- **Generation:** `grounded` generator over an `LLMClient`; the default
  `OpenAICompatClient` targets a **local** OpenAI-compatible server (Ollama / vLLM /
  LM Studio). No document text crosses the network.
- **Verification:** already deterministic + local (existence/quote checks); the NLI
  step, when added, uses a local MiniCheck/DeBERTa model.

The E0 smoke path (ADR 0003) is unchanged — it stays deterministic and dependency-free
for CI. Local-first governs the *research/production* configs (E2, E5, E9, …), which
run real models but still on-box.

## Consequences

- A researcher can run the entire pipeline — embed, retrieve, rerank, generate,
  verify — on confidential data with nothing leaving the machine (`ollama serve` +
  one model pull). This is the headline property for the legal setting.
- Heavy local deps (torch via sentence-transformers, a local LLM server) are the
  cost; they stay behind the `embeddings` / `llm` extras and lazy imports, so CI and
  a bare `pip install -e .[dev]` are unaffected.
- Hosted APIs are still first-class for measurement (quantifying the local-vs-API
  quality gap is RQ2), just not the default. Any experiment that uses one is
  explicit about it in its config.
- The grounded local generator cannot bypass trust: it emits citations that the
  deterministic verifier checks against the corpus and drops if unconfirmed, so the
  hallucinated-citation target (RQ3) holds even for a small local model.
