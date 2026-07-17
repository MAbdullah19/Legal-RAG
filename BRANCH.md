# Branch: `jurisdiction/us` — Prototype Pipeline (ACTIVE)

**Mission:** Working end-to-end RAG pipeline on open US case law, where public benchmarks (CLERC, COLIEE) let every technique be measured against the literature. This branch runs the ablation ladder E0–E10 and produces the adopted core config that the Pakistan track transfers.

**Charter:** [docs/jurisdictions/us.md](docs/jurisdictions/us.md) · Master plan: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) (phases P1–P3)

## Scope

- **In:** CAP + CourtListener connectors, `us` jurisdiction pack (eyecite/courts-db adapters), CLERC eval wiring, experiments E0–E10, US corpus tiers (10k → 100k → 1M)
- **Out (lives on `main`):** canonical model, component interfaces/registry, eval harness, retrieval/generation/verify components — develop here if convenient, but **commit to `main` and merge back in**
- **Out (other branches):** anything PK- or multi-specific

## Immediate next steps

1. Wait for P0 scaffold on `main` (repo skeleton, docker stack, fixture corpus, E0 on fixtures) — or do P0 on `main` now
2. CAP bulk connector (`static.case.law`) + snapshot `us/clerc-dev-10k`
3. eyecite adapter → `citation_edges`; ingestion quality dashboard
4. E0 baseline on CLERC-dev → MLflow; write the error taxonomy
5. Climb the ladder: E1 (hybrid) → E2 (rerank) → …

## Merge policy

Sync from `main` at least at every phase boundary. Core improvements discovered here land on `main` first, never diverge here.
