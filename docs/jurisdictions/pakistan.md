# Jurisdiction Annex — Pakistan (`jurisdiction/pakistan`)

> **Role:** The research target. Everything the US track validates gets transferred here and stress-tested against low-resource reality: scraped scan-heavy PDFs, a citation grammar with no existing extractor, no public benchmark, partial Urdu. The two headline contributions live on this track: **`pk-cite`** and **PakLegalQA** (RQ4).

## 1. Landscape facts that shape the pipeline

- Superior judiciary: Supreme Court of Pakistan; five High Courts (Lahore, Sindh, Islamabad, Peshawar, Balochistan); Federal Shariat Court. (AJK & Gilgit-Baltistan courts → multi track.)
- Judgments of the superior courts are written in **English** (Urdu versions exist for some landmark SC judgments post the 2015 Art. 251 ruling). Urdu is a stretch goal, not a blocker.
- No widely-adopted neutral citation standard. Citations run through **commercial reporters**: PLD, SCMR, CLC, PLJ, MLD, YLR, NLR, etc. Case numbers (e.g., `Civil Petition No. 1234-L of 2019`) identify unreported judgments.
- Court websites publish judgment PDFs directly — a mix of digitally-born and scanned documents; ratio unknown → **first task is an audit, not a scraper**.
- Copyright reality: raw judgments are public records; **reporter headnotes/editorial content are copyrighted** (PLD Publishers etc.). We ingest only court-published raw judgments; reporter *citations* (the strings) are facts and freely usable.

## 2. Data sources & acquisition plan

| Source | What | Access mode | Priority |
|---|---|---|---|
| supremecourt.gov.pk | SC judgments PDFs, cause lists | Scrape (polite) | **P1** |
| Sindh High Court — caselaw.shc.gov.pk | Searchable case-law portal | Scrape/API-ish | **P1** (best-structured HC source) |
| Lahore High Court — lhc.gov.pk | Judgment search + PDFs | Scrape | **P2** |
| Islamabad High Court — ihc.gov.pk / mis.ihc.gov.pk | Judgments | Scrape | P2 |
| Peshawar HC, Balochistan HC, Federal Shariat Court | Judgments | Scrape | P3 |
| **pakistancode.gov.pk** | Consolidated federal statutes (English) | Download | **P1** (feeds statute model) |
| punjablaws.gov.pk, provincial code sites | Provincial statutes | Download/scrape | P3 |
| PakistanLawSite / commercial DBs | Reported judgments + headnotes | **Not used** (license) | — |

**Source audit checklist (do during US-track P1 — costs only observation time):** per source: URL patterns & pagination · listing completeness (years covered) · PDF type sample (digital vs scanned %, 50-doc sample) · metadata availability on listing pages (parties, date, bench, case no.) · robots.txt / ToS posture · rate tolerance · change frequency. Output: `docs/jurisdictions/pk-source-audit.md` with a go/no-go + effort score per source.

Scraper design: per-source connector implementing the core `Fetcher` protocol; checksummed re-crawls (only fetch new/changed); provenance row per doc; identified user-agent; off-peak scheduling; failure quarantine. Target corpus v1: **SC complete + SHC + LHC recent decade** before widening.

## 3. `pk-cite` — Pakistani citation extractor (contribution #1)

eyecite-equivalent for Pakistani reporters; regex-grammar core + resolution layer. Open-sourced as a standalone package with its own test suite.

Citation grammar to cover (with real-format examples):

| Pattern | Example | Notes |
|---|---|---|
| `PLD <year> <court> <page>` | `PLD 2018 SC 595` | Court tokens: SC, Lah., Kar., Pesh., Quetta, Isl., FSC… |
| `<year> <reporter> <page>` | `2019 SCMR 1420`, `2020 CLC 883`, `2017 YLR 210`, `2021 MLD 401`, `2016 PLJ 55` | Reporter registry with court-inference rules |
| Reporter + court qualifier | `2018 CLC 1290 [Sindh]` | Bracketed bench qualifiers |
| Case numbers | `C.P. No. 1234-L of 2019`, `Crl.A. 45/2020`, `W.P. No. 3421/2018` | Unreported judgments; per-court numbering styles |
| Statute refs | `Section 302(b) PPC`, `Article 199 of the Constitution`, `s. 9(a) NAB Ordinance 1999` | Links judgments → statute nodes |
| Foreign cites | `(1932) AC 562`, `AIR 1973 SC 1461` | Pakistani judgments cite UK/Indian precedent — tagged foreign, resolved on multi track |

Deliverables: extraction (spans + normalized form) → resolution against corpus registry (reported-cite ↔ case-number aliasing for the same judgment — the dedup clusterer helps build this alias table) → `citation_edges` population. Metric: extraction P/R on a 200-judgment hand-labeled sample (target ≥95% P, ≥90% R); resolution rate tracked on dashboard.

## 4. OCR & parsing strategy

Ladder (per master plan §5.1), with PK-specific notes:

1. PyMuPDF text-layer extraction; **text-layer quality score** (dictionary hit-rate + layout sanity) decides whether the "digital" text is trustworthy — many PK PDFs have garbage embedded text.
2. Tesseract for clean scans → Surya/PaddleOCR for degraded ones.
3. VLM OCR (vision LLM page transcription) for the hardest tail — budget-capped, logged.
4. Urdu (stretch): Nastaliq script defeats standard OCR (Tesseract `urd` is weak); route Urdu pages to VLM OCR or specialized models (UTRNet-class). Detection: script classifier at page level; Urdu pages tagged and excluded from v1 metrics rather than silently degrading them.

**OCR noise is a research variable, not just a nuisance (RQ4):** build an eval slice with known-noisy documents; measure retrieval degradation vs. OCR confidence; test noise-robustness of lexical vs dense retrievers (hypothesis: BM25 degrades faster than embeddings on OCR noise — measurable, publishable).

Structure recovery: superior-court judgments use numbered paragraphs fairly consistently → paragraph segmentation rules + fallback to layout blocks; headnote/appendix boundaries per-court heuristics.

## 5. PakLegalQA — evaluation set (contribution #2)

No public Pakistani legal IR/QA benchmark exists. Constructing one carefully is the core of RQ4 and likely the most-cited artifact of the project.

**Composition target (v1: 300–500 items):**

| Slice | Source of questions | ~% |
|---|---|---|
| Conceptual doctrine questions | Bar-exam prep materials, law-journal case comments (questions re-authored, not copied) | 40% |
| Case-lookup / citation queries | Sampled from corpus (known-item retrieval) | 20% |
| Statute application | pakistancode sections + leading interpretive cases | 20% |
| Multi-hop precedent chains | Built from citation graph (case A applied in B, distinguished in C) | 15% |
| Unanswerable-from-corpus | Deliberate, for abstention calibration | 5% |

**Item schema:** question · gold answer (prose) · gold citations (doc_id + ¶ pinpoints) · qrels (graded relevance for retrieval eval) · query-type tag · difficulty · annotator IDs.

**Protocol:** (1) 100-item pilot: two law-student annotators draft, practitioner/faculty reviewer validates; annotation guidelines doc versioned in `evalsets/`; (2) measure inter-annotator agreement on relevance grades — **gate: κ ≥ 0.6** before scaling; (3) scale to full set; reviewer adjudicates disagreements; (4) freeze dev/test split (60/40), test touched only at release candidates. Tooling: Label Studio (self-hosted) with a retrieval-assisted annotation view (annotators see candidate paragraphs, mark relevance + pinpoints). Recruit via NUST/law-department contacts; budget authorship/acknowledgment for annotators — standard for benchmark papers.

**Release:** questions + annotations under CC-BY; judgments referenced by citation + court URL (not redistributed) → clean licensing.

## 6. Transfer study design (RQ4 — the paper's spine)

1. **Frozen transfer:** run the US-adopted config (through E9) unchanged on PK corpus + PakLegalQA. Per-component degradation table vs US results.
2. **Diagnose:** error analysis by component — expected failure concentrations: citation extraction (pk-cite closes this), OCR-noise slices, embedding domain shift (US-legal-tuned vs multilingual), authority scoring (different court hierarchy — pack config closes this).
3. **Adapt:** re-run selected ladder rungs (E3 chunking, E5 embedding bake-off incl. BGE-M3 multilingual, E7 graph w/ PK court weights) on PK data; quantify recovery per adaptation.
4. **Report:** transfer matrix (component × [frozen / adapted] × metric) — the empirical answer to "what does it actually take to port legal RAG to a low-resource jurisdiction."

## 7. Milestones (maps to master plan P4)

- [ ] Source audit doc complete (during US P1 — parallel, low effort)
- [ ] SC-Pak scraper + SHC scraper live; corpus v0 (≈5–10k judgments) w/ quality dashboard
- [ ] OCR ladder evaluated on 200-doc stratified sample; confidence routing tuned
- [ ] `pk-cite` v1: ≥95% P / ≥90% R on hand-labeled sample; edges populated; alias table (reported-cite ↔ case-no.)
- [ ] PakLegalQA pilot (100 items) → κ gate → v1 (300+) frozen
- [ ] Frozen-transfer run + degradation table
- [ ] Adaptation runs + transfer matrix → PK results writeup
- [ ] Stretch: Urdu page handling; statute point-in-time chains from pakistancode amendment data

## 8. PK-specific risks

| Risk | Mitigation |
|---|---|
| Court sites unstable / structure changes mid-project | Snapshot everything at fetch time (raw HTML+PDF into MinIO); connectors are thin & disposable, parsers run offline from snapshots |
| Scan quality worse than sampled | VLM-OCR budget reserve; corpus v1 can bias to recent (digital) years without hurting research validity — date-scope claims honestly |
| Annotator recruitment slips | Pilot needs only 2 annotators + 1 reviewer; start recruitment during P2/P3, not at P4 kickoff |
| pk-cite grammar long-tail (older/obscure reporters) | Coverage-driven: log unmatched citation-like spans, extend grammar by frequency; report coverage curve |
| Perception risk (legal AI in judiciary context) | Research framing, abstention-first demos, no advice claims, engage faculty reviewers early |
