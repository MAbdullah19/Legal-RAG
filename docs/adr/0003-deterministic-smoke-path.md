# ADR 0003 — A deterministic, service-free smoke path

- **Status:** Accepted
- **Date:** 2026-07-18

## Context

The engine will eventually depend on external services (Qdrant, Postgres, MinIO,
MLflow, Phoenix) and paid APIs (Anthropic/Voyage). If the baseline experiment and
CI depended on those, every contributor and every CI run would need secrets,
network, and a running docker stack — and results would be non-deterministic.

## Decision

Ship a full E0 pipeline built entirely from **deterministic, in-process
components** with no network and no API keys:

- `hashing` embedder (blake2b signed hashing, log-tf, L2-normalized) instead of a
  cloud/model embedder,
- `memory` vector index and pure-Python `bm25` instead of Qdrant/OpenSearch,
- `extractive` generator (quotes verbatim from retrieved paragraphs) instead of
  an LLM,
- `deterministic` verifier (existence + quote checks) with no NLI model,
- `JSONRunTracker` instead of MLflow.

Every one of these is a registered component behind the same Protocol as its
production counterpart, so moving to the real integration is a config change.
`legalrag eval --smoke` runs this path and asserts the P0 exit criteria
(end-to-end run, zero hallucinated citations) in CI in seconds.

## Consequences

- CI is fast, hermetic, secret-free, and reproducible.
- The baseline is honest but weak (e.g. the hashing embedder is lexical-ish, so
  the E0→E1 hybrid gain is muted on the fixture); real embedders quantify that in
  the E5 bake-off. This is a known, documented property, not a bug.
- New contributors run the whole system with `pip install -e .[dev]` and nothing
  else.
