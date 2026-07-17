# Jurisdiction Annex — Multi-Jurisdiction (`jurisdiction/multi`)

> **Role:** Future prospect / architectural north star. Not built first — **extracted** from the two working jurisdiction packs (US, PK) once they exist, because abstractions distilled from two real implementations survive; abstractions designed up-front don't. Until then, this track carries design notes and enforces one rule on the other tracks: *no jurisdiction-specific logic inside `core/`*.

## 1. Why this is worth building (not just generality for its own sake)

1. **Pakistani courts cite UK and Indian precedent routinely** (Privy Council lineage, shared statutory ancestry — Contract Act 1872, PPC/IPC, CPC). A PK research answer about, say, contractual damages may hinge on `Hadley v Baxendale` or `AIR` -cited Indian authority. Cross-jurisdiction citation resolution therefore improves the *Pakistan* system — the multi track pays rent before it's a product.
2. Comparative-law research ("how do PK and Indian courts treat s. 34 arbitration challenges?") is a query class no single-jurisdiction system can answer.
3. Commonwealth jurisdictions share document DNA (IRAC-ish judgments, numbered paragraphs, reporter citations) → one engine + thin packs is plausibly sufficient, which is exactly RQ5.

## 2. JurisdictionPack — the target abstraction

A jurisdiction is a declarative pack + a small set of protocol implementations. Everything else is core.

```python
class JurisdictionPack(Protocol):
    meta: PackMeta                      # id ("pk"), languages, precedent regime notes
    connectors: list[Fetcher]           # per-source acquisition
    parser_profile: ParserProfile       # structure heuristics, OCR routing defaults
    citation_scheme: CitationScheme     # extractor + normalizer + resolver (eyecite / pk-cite / ...)
    court_registry: CourtRegistry       # hierarchy DAG + binding/persuasive rules → authority scorer
    doc_typology: dict[str, DocType]    # source labels → canonical types
    eval_pack: EvalPack | None          # benchmark suites + frozen sets for this jurisdiction
    language_profile: LanguageProfile   # scripts, OCR routes, embedding requirements
```

Boundary rules (enforced by review + a lint check on imports):

- `core/` never imports from `jurisdictions/` — dependency arrow points one way.
- Anything needed by two packs gets promoted into core behind a protocol.
- A pack must be able to stand up with **config + ≤ ~500 lines of code** for a Commonwealth-style jurisdiction. That number is the measurable abstraction-quality target (master plan P5 exit criterion).

## 3. Cross-jurisdiction capabilities (the new research surface)

| Capability | Mechanism |
|---|---|
| Foreign-citation resolution | Citation schemes registered globally; `pk` documents containing `[1932] AC 562` / `AIR 1973 SC 1461` resolve into UK/IN nodes when those packs exist; dangling-foreign edges tracked meanwhile |
| Cross-jurisdiction citation graph | Single graph, jurisdiction-tagged nodes; authority scoring becomes regime-aware (foreign precedent = persuasive weight, configurable per pack pair) |
| Comparative queries | Query decomposition per jurisdiction → per-corpus retrieval → aligned synthesis w/ per-jurisdiction citation blocks |
| Retrieval scoping | `jurisdiction` is a first-class filter; default scope from query context, overridable |
| Cross-lingual layer | Multilingual embedding space (BGE-M3 class) as common denominator when packs disagree on language |

## 4. Candidate third jurisdictions (the abstraction test, P5)

| Candidate | Data posture | Why |
|---|---|---|
| **UK — Find Case Law (National Archives)** | Open, structured XML, neutral citations | Cleanest validation target; directly useful to PK foreign-cite resolution |
| India — Indian Kanoon / eCourts | Scraping-restricted / uneven | Highest value for PK cross-citation, harder licensing — evaluate carefully |
| Canada — CanLII | API w/ terms | COLIEE alignment |

P5 exit criterion (from master plan): a UK skeleton pack reaches ingest→retrieve on a sample corpus in **< 1 week of effort using only the pack interface** — if it needs core changes, the abstraction failed and gets revised.

## 5. Roadmap position & discipline

- **Now → P4:** design notes only. The track's job is exerting architectural pressure on `main` (boundary rules above) and logging every "this would break for jurisdiction X" observation in `docs/jurisdictions/multi-notes.md`.
- **P5:** extract `JurisdictionPack` from US+PK packs · migrate both · UK skeleton test · cross-jurisdiction citation resolution POC (PK→UK) · comparative-query eval slice (small, hand-built).
- **Beyond (out of current scope):** jurisdiction routing UI, per-pack access policies, federated corpora across data-sovereignty boundaries (the hybrid LLM abstraction already anticipates this).

## 6. Risks

| Risk | Mitigation |
|---|---|
| Premature abstraction freezes wrong interfaces | Hard rule: no extraction before two packs work end-to-end |
| Licensing heterogeneity across jurisdictions | Per-doc license ledger already in canonical model; packs declare source license posture |
| Comparative-eval ground truth is expensive | Small hand-built slice (~30 items) is enough for a POC claim; scale only if RQ5 becomes a paper focus |
