"""US court registry — maps reporters to the canonical CourtLevel (US annex §2).

The reporter a case is published in identifies its court tier: U.S./S. Ct./L. Ed.
are the Supreme Court; the Federal Reporter (F., F.2d/3d/4th) is the Courts of
Appeals; the Federal Supplement (F. Supp. …) is the District Courts. This feeds
the authority scorer (apex ≻ appellate ≻ trial) with no external dependency.
"""

from __future__ import annotations

import re

from legalrag.core.models import CourtLevel, CourtRef

# Canonical reporter -> court tier.
REPORTER_COURT_LEVEL: dict[str, CourtLevel] = {
    "U.S.": CourtLevel.APEX,
    "S. Ct.": CourtLevel.APEX,
    "L. Ed.": CourtLevel.APEX,
    "L. Ed. 2d": CourtLevel.APEX,
    "F.": CourtLevel.APPELLATE,
    "F.2d": CourtLevel.APPELLATE,
    "F.3d": CourtLevel.APPELLATE,
    "F.4th": CourtLevel.APPELLATE,
    "F. App'x": CourtLevel.APPELLATE,
    "F. Supp.": CourtLevel.TRIAL,
    "F. Supp. 2d": CourtLevel.TRIAL,
    "F. Supp. 3d": CourtLevel.TRIAL,
}

_COURT_BY_LEVEL: dict[CourtLevel, CourtRef] = {
    CourtLevel.APEX: CourtRef(
        court_id="us/scotus", name="Supreme Court of the United States",
        level=CourtLevel.APEX, jurisdiction="us",
    ),
    CourtLevel.APPELLATE: CourtRef(
        court_id="us/circuit", name="U.S. Court of Appeals",
        level=CourtLevel.APPELLATE, jurisdiction="us",
    ),
    CourtLevel.TRIAL: CourtRef(
        court_id="us/district", name="U.S. District Court",
        level=CourtLevel.TRIAL, jurisdiction="us",
    ),
}


def court_level_for_reporter(reporter: str) -> CourtLevel | None:
    return REPORTER_COURT_LEVEL.get(reporter)


def court_for_reporter(reporter: str) -> CourtRef | None:
    level = REPORTER_COURT_LEVEL.get(reporter)
    return _COURT_BY_LEVEL.get(level) if level else None


# CourtListener uses short court ids ("scotus", "ca9", "cadc", "cand", ...). Map
# them to the canonical tier so ingested opinions feed the authority scorer. v0
# covers the federal hierarchy (the CLERC/CAP dev slice); state apex courts are a
# known simplification (default TRIAL) revisited when a state slice is ingested.
_CIRCUIT_RE = re.compile(r"^ca(\d{1,2}|dc|fc)$")


def court_level_for_courtlistener_id(court_id: str) -> CourtLevel:
    if court_id == "scotus":
        return CourtLevel.APEX
    if _CIRCUIT_RE.match(court_id):
        return CourtLevel.APPELLATE
    return CourtLevel.TRIAL


def court_ref_for_courtlistener_id(court_id: str, name: str | None = None) -> CourtRef:
    """A CourtRef carrying the real court name from ingestion metadata, with the
    tier inferred from the court id (apex/appellate/trial)."""
    level = court_level_for_courtlistener_id(court_id)
    return CourtRef(
        court_id=f"us/{court_id}",
        name=name or _COURT_BY_LEVEL[level].name,
        level=level,
        jurisdiction="us",
    )
