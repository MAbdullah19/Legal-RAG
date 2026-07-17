"""Controlled vocabularies for the canonical document model.

These enums are jurisdiction-neutral. Jurisdiction packs map their source
labels onto them (e.g. a US ``courts-db`` level or a PK reporter court token
both resolve to a :class:`CourtLevel`), which is what lets every downstream
component stay jurisdiction-agnostic.
"""

from __future__ import annotations

from enum import StrEnum


class DocType(StrEnum):
    JUDGMENT = "judgment"
    ORDER = "order"
    STATUTE = "statute"
    RULE = "rule"
    PLEADING = "pleading"
    NOTIFICATION = "notification"
    OTHER = "other"


class CourtLevel(StrEnum):
    """Hierarchy tier, coarse enough to be shared across jurisdictions.

    ``authority_weight`` seeds the authority scorer (master plan §5.5); packs may
    override via the court registry, but this gives a jurisdiction-free default.
    """

    APEX = "apex"           # Supreme Court of Pakistan, SCOTUS
    APPELLATE = "appellate"  # High Courts, US Circuit Courts
    TRIAL = "trial"          # District / sessions courts
    TRIBUNAL = "tribunal"    # specialised tribunals
    OTHER = "other"

    @property
    def authority_weight(self) -> float:
        return {
            CourtLevel.APEX: 1.0,
            CourtLevel.APPELLATE: 0.7,
            CourtLevel.TRIAL: 0.4,
            CourtLevel.TRIBUNAL: 0.5,
            CourtLevel.OTHER: 0.3,
        }[self]


class Treatment(StrEnum):
    """How a citing sentence treats the cited authority (master plan §5.5)."""

    FOLLOWED = "followed"
    RELIED_ON = "relied_on"
    EXPLAINED = "explained"
    DISTINGUISHED = "distinguished"
    OVERRULED = "overruled"
    CITED = "cited"          # neutral mention


NEGATIVE_TREATMENTS: frozenset[Treatment] = frozenset(
    {Treatment.DISTINGUISHED, Treatment.OVERRULED}
)


class CitationKind(StrEnum):
    REPORTER = "reporter"        # "PLD 2018 SC 595", "410 U.S. 113"
    NEUTRAL = "neutral"          # court-issued neutral citation
    CASE_NUMBER = "case_number"  # "C.P. No. 1234-L of 2019" (unreported)
    STATUTE = "statute"          # "Section 302 PPC", "Article 199"
    FOREIGN = "foreign"          # "[1932] AC 562", "AIR 1973 SC 1461"
    UNKNOWN = "unknown"


class QueryType(StrEnum):
    """Tagged on every eval item; all metrics report per-type (master plan §6.2)."""

    CITATION_LOOKUP = "citation_lookup"
    PARTY_LOOKUP = "party_lookup"
    CONCEPTUAL = "conceptual"
    STATUTE = "statute"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"


class LicenseBasis(StrEnum):
    PUBLIC_RECORD = "public_record"
    OPEN_LICENSE = "open_license"
    PERMISSION = "permission"
    UNKNOWN = "unknown"
