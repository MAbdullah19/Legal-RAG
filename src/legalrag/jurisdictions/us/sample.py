"""A tiny hand-crafted US opinion corpus for hermetic testing (US annex §4).

Real US case names/citations, paraphrased/abbreviated opinion text — enough to
exercise the pack end-to-end (extract -> resolve -> citation graph) with no
network. Documents are stored WITHOUT ``citations_out``: the extractor populates
them, mirroring the real ingestion flow. Court is assigned from the reporter.

Run: ``python -m legalrag.jurisdictions.us.sample`` -> evalsets/us/mini/documents.jsonl
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from legalrag.core.corpus import write_documents
from legalrag.core.models import (
    Citation,
    CitationKind,
    DocType,
    LegalDocument,
    LicenseBasis,
    Paragraph,
    Provenance,
    Section,
)
from legalrag.jurisdictions.us.courts import court_for_reporter

_REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = _REPO_ROOT / "evalsets" / "us" / "mini"


def _slug(cite: str) -> str:
    return cite.lower().replace(".", "").replace(" ", "-")


def opinion(cite: str, reporter: str, title: str, year: int, paras: list[str]) -> LegalDocument:
    return LegalDocument(
        doc_id=f"us/scotus/{_slug(cite)}/v1",
        doc_type=DocType.JUDGMENT,
        jurisdiction="us",
        title=title,
        court=court_for_reporter(reporter),
        decision_date=date(year, 1, 1),
        citations_self=[Citation(raw=cite, kind=CitationKind.REPORTER, normalized=cite)],
        source=Provenance(
            source="us/sample",
            url=f"https://www.courtlistener.com/{_slug(cite)}/",
            license_basis=LicenseBasis.PUBLIC_RECORD,
            license_note="US judicial opinion (public record); text paraphrased for fixture",
        ),
        body=[Section(heading="Opinion", path=["Opinion"],
                      paragraphs=[Paragraph(text=t) for t in paras])],
    )


def build_us_sample() -> list[LegalDocument]:
    return [
        opinion(
            "381 U.S. 479", "U.S.", "Griswold v. Connecticut", 1965,
            ["The Court holds that a right to marital privacy is protected against state intrusion.",
             "Several fundamental guarantees create zones of privacy that the State may not invade."],
        ),
        opinion(
            "410 U.S. 113", "U.S.", "Roe v. Wade", 1973,
            ["This Court in Griswold v. Connecticut, 381 U.S. 479, recognized a constitutional right to privacy.",
             "Relying on Griswold, we hold that this right is broad enough to encompass the decision at issue."],
        ),
        opinion(
            "505 U.S. 833", "U.S.", "Planned Parenthood v. Casey", 1992,
            ["We reaffirm the essential holding of Roe v. Wade, 410 U.S. 113, while adopting an undue-burden standard.",
             "Adhering to Roe, the Court declines to overturn its central rule."],
        ),
        opinion(
            "597 U.S. 215", "U.S.", "Dobbs v. Jackson Women's Health Org.", 2022,
            ["Roe v. Wade, 410 U.S. 113, and Planned Parenthood v. Casey, 505 U.S. 833, are overruled.",
             "The Court holds the Constitution does not confer the right recognized in those decisions."],
        ),
        opinion(
            "716 F.3d 1213", "F.3d", "Isaacson v. Horne", 2013,
            ["Applying Roe v. Wade, 410 U.S. 113, and Planned Parenthood v. Casey, 505 U.S. 833, we assess the statute.",
             "The Court of Appeals is bound by the Supreme Court's viability framework."],
        ),
    ]


def main() -> None:
    docs = build_us_sample()
    n = write_documents(docs, OUT_DIR / "documents.jsonl")
    print(f"wrote {n} US sample opinions to {OUT_DIR}")


if __name__ == "__main__":
    main()
