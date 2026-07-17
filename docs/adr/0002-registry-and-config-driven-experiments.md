# ADR 0002 — Registry + config-driven experiments; no framework backbone

- **Status:** Accepted
- **Date:** 2026-07-18

## Context

The research goal (master plan RQ1–RQ3) is to test, combine, and ablate RAG
techniques. That requires two properties: (1) a technique can be swapped without
touching the pipeline, and (2) a full system instance is a *value* (a config)
that can be hashed, tracked, and diffed. LangChain/LlamaIndex were considered as
the backbone and rejected: their abstractions churn and obscure exactly the
component boundaries we need to ablate across. They remain fine as *implementations*
behind our own interfaces (e.g. a Docling-backed parser).

## Decision

- Every pipeline stage is a `Protocol` in `legalrag.core.interfaces` plus a
  registry entry (`@register("stage", "name")`). The pipeline runner resolves
  components by name from config; adding a technique never edits the runner.
- An experiment is a YAML file naming registered components and their params.
  Config loading is a thin pydantic-validated layer (`legalrag.core.config`),
  not Hydra — Hydra may later *wrap* this for multirun sweeps, but the schema is
  ours so runs stay reproducible and framework-independent.
- Every run records: config hash, corpus snapshot id, index id, git commit,
  model versions, seed (`legalrag.core.eval.tracking`).

## Consequences

- Ablations are config diffs; `legalrag compare` can significance-test any two.
- The smoke path stays dependency-light (deterministic local components) so CI
  needs no services or API keys; heavy integrations register the same way.
- One-time cost: we maintain our own interfaces. Accepted — it is the point.
