# Branch: `jurisdiction/pakistan` — Research Target

**Mission:** Adapt the US-validated pipeline to Pakistan's judiciary and quantify what low-resource transfer actually costs (RQ4). Home of the two headline contributions: **`pk-cite`** (Pakistani citation extractor) and **PakLegalQA** (first expert-validated PK legal QA/retrieval benchmark).

**Charter:** [docs/jurisdictions/pakistan.md](docs/jurisdictions/pakistan.md) · Master plan: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) (phase P4)

## Scope

- **In:** court-site scrapers (SC-Pak, SHC, LHC first), OCR-ladder evaluation on real scans, `pk` jurisdiction pack, `pk-cite`, PakLegalQA construction + annotation tooling, US→PK transfer study
- **Out (lives on `main`):** all engine components — this branch configures and stresses them, core fixes go to `main` first
- **Stretch:** Urdu page handling, statute point-in-time chains from pakistancode

## Can start now (parallel to US track — costs only observation time)

1. **Source audit** per annex §2 checklist → `docs/jurisdictions/pk-source-audit.md`: URL patterns, digital-vs-scanned sample ratios, metadata availability, robots/ToS posture for supremecourt.gov.pk, caselaw.shc.gov.pk, lhc.gov.pk, pakistancode.gov.pk
2. Collect a 50-judgment stratified PDF sample per court for the OCR audit
3. Start the `pk-cite` grammar test file: hand-collect ~200 real citation strings across reporters/eras
4. Begin annotator recruitment conversations (law students + one practitioner/faculty reviewer)

## Full pipeline work (after US track P3 delivers the adopted config)

Scrapers → corpus v0 (~5–10k judgments) → OCR routing tuned → `pk-cite` v1 (≥95% P / ≥90% R) → PakLegalQA pilot (100 items, κ ≥ 0.6 gate) → v1 freeze → frozen-transfer run → adaptation runs → transfer matrix.

## Merge policy

Sync from `main` at every phase boundary. Core improvements land on `main` first.
