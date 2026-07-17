# ADR 0001 — Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-07-18

## Context

This is a research platform whose value depends on being able to explain *why*
each default was chosen and reconstruct the reasoning when a default is later
overturned by an experiment. Papers cite reasons; a modular system with a dozen
swappable slots accumulates many small decisions.

## Decision

Every consequential default (component choice, interface shape, evaluation
protocol) gets a short numbered ADR in `docs/adr/`. ADRs are append-only:
superseding an ADR adds a new one that references the old, rather than editing
history. Format: Context / Decision / Consequences.

## Consequences

- Cheap to write, high-value for a solo researcher (defeats context loss).
- The experiment ladder's adopt/drop verdicts (master plan §6.3) each produce an
  ADR, so the paper's method section is largely assembled from them.
