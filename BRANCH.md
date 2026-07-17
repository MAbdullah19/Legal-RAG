# Branch: `jurisdiction/us` — Prototype Pipeline (ACTIVE)

**Mission:** Working end-to-end RAG pipeline on open US case law, where public benchmarks (CLERC, COLIEE) let every technique be measured against the literature. This branch runs the ablation ladder E0–E10 and produces the adopted core config that the Pakistan track transfers.

**Charter:** [docs/jurisdictions/us.md](docs/jurisdictions/us.md) · Master plan: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) (phases P1–P3)

## Scope

- **In:** CAP + CourtListener connectors, `us` jurisdiction pack (eyecite/courts-db adapters), CLERC eval wiring, experiments E0–E10, US corpus tiers (10k → 100k → 1M)
- **Out (lives on `main`):** canonical model, component interfaces/registry, eval harness, retrieval/generation/verify components — develop here if convenient, but **commit to `main` and merge back in**
- **Out (other branches):** anything PK- or multi-specific

## Progress

- [x] P0 engine merged from `main` (canonical model, registry, config runner, eval harness, verifier, E0/E1/E7)
- [x] **US pack citation extraction (hermetic):** `jurisdictions/us/` — `us_regex`
      reporter + case-name extractor with rule-based treatment heuristic;
      `eyecite` opt-in adapter (lazy); reporter→court registry; hand-crafted
      US sample corpus (`evalsets/us/mini/`, real SCOTUS/circuit cites).
      Extract → resolve (core `CitationResolver`) → citation graph verified
      end-to-end: Dobbs→Roe/Casey OVERRULED edges, authority ordering, 8 tests.

## Next steps

1. CAP bulk connector (`static.case.law`) + snapshot `us/clerc-dev-10k` — the deferred network step
2. Point `us_regex` (and eyecite once `pip install .[citations]`) at real opinions; measure resolution rate
3. E0 baseline on CLERC-dev → MLflow; write the error taxonomy
4. Climb the ladder: E1 (hybrid) → E2 (rerank) → …
5. Treatment classifier component (replace the keyword heuristic; few-shot → fine-tuned per plan §5.5)

Note (for authority tuning later): recency decay (τ=15y in `core`) crushes very old
landmark cases; revisit τ / add a citation-count floor when tuning on real CAP data.

## Merge policy

Sync from `main` at least at every phase boundary. Core improvements discovered here land on `main` first, never diverge here.
