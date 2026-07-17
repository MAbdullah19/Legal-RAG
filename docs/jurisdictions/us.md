# Jurisdiction Annex — United States (`jurisdiction/us`)

> **Role:** Prototype track. High-resource open data + public benchmarks = the measurement bed where the ablation ladder (master plan §7) runs and core techniques earn adoption. Everything validated here transfers to the Pakistan track via the transfer study (RQ4).

## 1. Data sources

| Source | What | Access | Notes |
|---|---|---|---|
| **Caselaw Access Project (CAP)** | ~6.9M digitized US cases (Harvard LIL), through ~2020 | Fully open since 2024; bulk at `static.case.law` | Primary historical corpus; CLERC is built from it — use CAP text so benchmark and corpus align |
| **CourtListener (Free Law Project)** | Ongoing opinions, oral args, judges DB | REST API + bulk exports | Coverage after CAP's cutoff; PACER/RECAP docket layer if pleadings ever needed |
| RECAP archive | Federal dockets/filings | Bulk/API | Out of scope for v1 (opinions first); casefile-style docs later |
| **US Code** | Federal statutes, XML | uscode.house.gov (OLRC) | Exercises the statute point-in-time model with clean data |
| State codes | Per-state statutes | Varies | Only if an experiment needs them — not v1 |

Corpus tiers (DVC-snapshotted, referenced by ID in experiment configs):

- `us/clerc-dev-10k` — ~10k docs aligned with CLERC dev queries. All bake-offs run here.
- `us/cap-100k` — 100k-case slice (stratified by court level + decade). Scale-behavior checks.
- `us/cap-1m` — 1M+ cases. Only the adopted config gets indexed here (P-continuous).

## 2. Jurisdiction pack components

| Component | Implementation |
|---|---|
| Citation extraction | **eyecite** + `reporters-db` (all US reporter formats, pin cites, short cites, id./supra resolution) |
| Court registry | `courts-db` → canonical `CourtRef` + hierarchy levels (SCOTUS → Cir. → Dist.; state analogues) for authority scoring |
| Structure recovery | CAP provides paragraph structure in bulk XML/JSON; CourtListener HTML → paragraph segmentation rules |
| Metadata | CAP/CL ship parties, dates, courts, docket numbers — mostly free; validates the canonical model cheaply |
| Precedent semantics | Federal/state binding-vs-persuasive rules encoded in the authority scorer config |

Practical consequence: **US ingestion is mostly parsing structured exports, not scraping** — which is exactly why the prototype runs here first: engineering effort concentrates on retrieval/generation research, not data wrangling.

## 3. Benchmarks & eval wiring

| Suite | Task | Use |
|---|---|---|
| **CLERC** | Case-law passage retrieval + retrieval-augmented drafting w/ citations | Primary: retrieval metrics + citation P/R for generation |
| COLIEE T1/T2 | Case retrieval + entailment | Secondary retrieval; entailment slice doubles as verifier eval data |
| LegalBench-RAG | Contract/privacy retrieval (CUAD, MAUD, ContractNLI, PrivacyQA) | Generalization check beyond case law |
| LegalBench (subset) | Legal reasoning (rule-application, IRAC) | Generation probes |
| `us-own-heldout` | ~150 QA+gold-citation items mined from CAP citation graph, expert-spot-checked | Anti-overfitting control; never used for tuning |

Query-type taxonomy tagged on every eval item (citation-lookup / party-lookup / conceptual / statute / multi-hop) → all metrics reported per-type, since techniques trade off across types (master plan E6).

## 4. Milestones (maps to master plan P1–P3)

- [ ] CAP bulk connector + CourtListener API connector (checksummed, resumable)
- [ ] `us/clerc-dev-10k` snapshot ingested; quality dashboard (parse rate, citation-resolution rate ≥95%)
- [ ] eyecite adapter populating `citation_edges`; graph sanity checks (no dangling resolution above threshold)
- [ ] E0 baseline on CLERC-dev logged to MLflow + error taxonomy doc
- [ ] E1–E5 rungs with adopt-rule verdicts
- [ ] E6–E7 + E9 (grounding + verification) — headline US results table
- [ ] `us/cap-100k` scale check of adopted config (latency/cost curves)
- [ ] Results writeup v1 (feeds paper + informs PK transfer)

## 5. US-specific risks

| Risk | Mitigation |
|---|---|
| CAP coverage ends ~2020 → recency gap | CourtListener fills forward; date-scoped eval queries |
| CLERC leakage into general LLM pretraining | Grounding-focused metrics (citation P/R vs provided context) are leakage-resistant; report retrieval and generation separately |
| Corpus scale tempts premature optimization | Tier discipline: research on 10k, scale only adopted configs |
