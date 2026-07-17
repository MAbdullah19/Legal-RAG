# Branch: `jurisdiction/multi` — Future Prospect

**Mission:** One engine, many legal systems. Extract the `JurisdictionPack` abstraction from the two working packs (US, PK), build cross-jurisdiction citation resolution (Pakistani judgments cite UK/Indian precedent — this pays rent for the PK track), and enable comparative-law queries (RQ5).

**Charter:** [docs/jurisdictions/multi.md](docs/jurisdictions/multi.md) · Master plan: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) (phase P5)

## Discipline (why this branch is mostly quiet until P5)

**No extraction before two jurisdiction packs work end-to-end.** Abstractions distilled from working implementations survive; up-front ones don't. Until P5 this branch:

1. Accumulates design notes: log every "this would break for jurisdiction X" observation in `docs/jurisdictions/multi-notes.md`
2. Enforces the boundary rule on other tracks: `core/` never imports from `jurisdictions/`; anything two packs need gets promoted to core behind a protocol

## P5 work (after PK pipeline works)

1. Extract `JurisdictionPack` protocol from US + PK packs; migrate both
2. **Abstraction test:** stand up a UK skeleton pack (Find Case Law — open, structured XML, neutral citations) in < 1 week using only the pack interface. Needing core changes = the abstraction failed → revise
3. Cross-jurisdiction citation resolution POC: PK foreign cites (`[1932] AC 562`, `AIR 1973 SC 1461`) resolving into UK/IN graph nodes
4. Comparative-query eval slice (~30 hand-built items) + regime-aware authority scoring (foreign precedent = persuasive, per pack pair)

## Merge policy

This branch's endgame is folding into `main` as `src/legalrag/jurisdictions/` packages — it is a staging area for the refactor, not a fork.
