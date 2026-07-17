"""Build the fixture corpus (master plan §9.2 / §13).

A tiny, fully synthetic mini-jurisdiction ("Testland", ``xx``) — invented parties
and a fictional statute, so nothing here states real law. It is handcrafted in
code (typed and deterministic) rather than hand-written JSONL, then serialized to
``documents.jsonl`` / ``queries.jsonl`` which the loader reads exactly like a real
scraped corpus.

It deliberately includes the adversarial cases the plan calls for so the whole
pipeline is exercised in CI:
  * a scanned judgment with OCR noise + low confidence (D9)
  * a duplicate judgment sharing a dedup cluster id (D10 == D3)
  * a circular citation pair (D7 <-> D8)
  * a statute version chain (S1 -> S2, point-in-time)

Run: ``python evalsets/fixture/build_fixture.py``
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from legalrag.core.corpus import write_documents, write_queries
from legalrag.core.models import (
    Citation,
    CitationKind,
    CitationMention,
    CourtLevel,
    CourtRef,
    DocType,
    DocVersion,
    EvalQuery,
    LegalDocument,
    Paragraph,
    Provenance,
    Qrel,
    QualityReport,
    QueryType,
    Section,
)

HERE = Path(__file__).resolve().parent

APEX = CourtRef(court_id="xx/apex", name="Supreme Court of Testland", level=CourtLevel.APEX, jurisdiction="xx")
HIGH = CourtRef(court_id="xx/high", name="Testland High Court", level=CourtLevel.APPELLATE, jurisdiction="xx")
TRIAL = CourtRef(court_id="xx/trial", name="Testland District Court", level=CourtLevel.TRIAL, jurisdiction="xx")


def prov(ref: str) -> Provenance:
    from legalrag.core.models import LicenseBasis

    return Provenance(
        source="fixture",
        url=f"https://courts.testland.example/{ref}",
        license_basis=LicenseBasis.PUBLIC_RECORD,
        license_note="synthetic public record (fixture)",
    )


def sec(heading: str, *texts: str) -> Section:
    return Section(heading=heading, path=[heading], paragraphs=[Paragraph(text=t) for t in texts])


def judgment(
    doc_id: str,
    title: str,
    court: CourtRef,
    d: date,
    self_cite: str,
    body: list[Section],
    *,
    judges: list[str] | None = None,
    out: list[CitationMention] | None = None,
    quality: QualityReport | None = None,
) -> LegalDocument:
    return LegalDocument(
        doc_id=doc_id,
        doc_type=DocType.JUDGMENT,
        jurisdiction="xx",
        title=title,
        court=court,
        decision_date=d,
        judges=judges or ["Justice A. Kareem"],
        citations_self=[Citation(raw=self_cite, kind=CitationKind.REPORTER, normalized=self_cite)],
        source=prov(doc_id.replace("/", "_")),
        body=body,
        citations_out=out or [],
        quality=quality or QualityReport(),
    )


def cite(raw: str, target: str, treatment: str | None = None) -> CitationMention:
    from legalrag.core.models import Treatment

    return CitationMention(
        raw=raw,
        kind=CitationKind.REPORTER,
        normalized=raw,
        target_doc_id=target,
        treatment=Treatment(treatment) if treatment else None,
    )


def build_documents() -> list[LegalDocument]:
    docs: list[LegalDocument] = []

    # D1 — bail in economic offences (apex)
    docs.append(judgment(
        "xx/apex/2018-050/v1", "Ahsan v. State", APEX, date(2018, 4, 12), "2018 TLR 50",
        [
            sec("Facts", "The appellant was accused of large-scale financial misappropriation and sought post-arrest bail after prolonged detention."),
            sec("Principles",
                "Bail in economic offences is not to be refused as punishment; the gravity of the alleged fraud is a relevant but not a decisive factor.",
                "Where the investigation is complete and the accused is unlikely to abscond or tamper with evidence, continued incarceration serves no purpose and bail should ordinarily follow."),
            sec("Held", "The petition is allowed and the appellant is admitted to bail subject to sureties."),
        ],
    ))

    # D2 — limitation / condonation of delay (high) -> statute S1
    docs.append(judgment(
        "xx/high/2019-114/v1", "Bano v. Collector", HIGH, date(2019, 9, 3), "2019 TLR 114",
        [
            sec("Facts", "The appeal was filed nineteen days beyond the limitation period prescribed by the Limitation Act of Testland, and the appellant sought condonation of the delay."),
            sec("Limitation",
                "Under the Limitation Act, an appeal to this Court must be filed within thirty days; the Court may condone delay only on proof of sufficient cause explaining every day of the delay.",
                "Sufficient cause is to be construed liberally where the party has acted bona fide, but a litigant who is negligent cannot invoke the Court's discretion."),
            sec("Held", "The explanation being unsatisfactory, condonation is refused and the appeal is dismissed as time-barred."),
        ],
        out=[cite("Limitation Act s. 3", "xx/statute/limitation-act/v1")],
    ))

    # D4 — foundational remoteness case (older apex), cited by D3/D5/D6
    docs.append(judgment(
        "xx/apex/1990-003/v1", "Foundational Carriage Co. v. Meridian Mills", APEX, date(1990, 1, 20), "1990 TLR 3",
        [
            sec("Rule of Remoteness",
                "Damages for breach of contract are recoverable only for loss that arises naturally from the breach, or that both parties could reasonably have contemplated at the time of contracting.",
                "Loss flowing from special circumstances not communicated to the defaulting party is too remote to be recovered."),
            sec("Held", "The carrier was not liable for the extraordinary loss of profits that had not been made known to it."),
        ],
        judges=["Justice R. Meridian"],
    ))

    # D5 — applies remoteness, later partly overruled (high) -> D4
    docs.append(judgment(
        "xx/high/2015-201/v1", "Zephyr Textiles v. Nasir", HIGH, date(2015, 6, 15), "2015 TLR 201",
        [
            sec("Facts", "Zephyr Textiles claimed lost profits after the defendant's late delivery of dyeing chemicals disrupted an export order."),
            sec("Analysis",
                "Applying the rule of remoteness, ordinary lost profits are recoverable, but the Court held that even uncommunicated export penalties were recoverable where the defendant ought to have known of the export trade.",
                "This extension of contemplation to matters merely capable of being known marks the outer boundary of recoverable damages."),
            sec("Held", "Damages including the export penalty were awarded to the plaintiff."),
        ],
        out=[cite("1990 TLR 3", "xx/apex/1990-003/v1", "relied_on")],
    ))

    # D3 — remoteness at apex, leading modern statement (apex) -> D4 (followed), D5 (cited)
    docs.append(judgment(
        "xx/apex/2020-077/v1", "Rahim Trading v. Delta Mills", APEX, date(2020, 11, 2), "2020 TLR 77",
        [
            sec("Introduction", "This appeal, reported as 2020 TLR 77, concerns the correct measure of damages for remoteness following a breach of a supply contract."),
            sec("Discussion",
                "We reaffirm the foundational rule that only losses arising naturally, or within the reasonable contemplation of the parties, are recoverable.",
                "The broader test suggested in Zephyr Textiles, extending liability to matters a defendant merely ought to have known, states the principle too widely and is not approved."),
            sec("Held", "The award of the uncommunicated export penalty is set aside; ordinary damages are affirmed."),
        ],
        judges=["Justice A. Kareem", "Justice S. Noor"],
        out=[
            cite("1990 TLR 3", "xx/apex/1990-003/v1", "followed"),
            cite("2015 TLR 201", "xx/high/2015-201/v1", "distinguished"),
        ],
    ))

    # D6 — multi-hop: overrules D5, follows D4 (apex)
    docs.append(judgment(
        "xx/apex/2021-090/v1", "Delta Mills v. Rahim Trading (Review)", APEX, date(2021, 3, 18), "2021 TLR 90",
        [
            sec("Scope", "On review the Court settles the status of the wider contemplation test after Rahim Trading."),
            sec("Holding",
                "To the extent that Zephyr Textiles held a defendant liable for losses it merely ought to have known of, that decision is overruled.",
                "The foundational two-limb rule of remoteness is followed as the sole test."),
            sec("Held", "The review is disposed of; Zephyr Textiles is overruled on the point of remoteness."),
        ],
        judges=["Justice A. Kareem"],
        out=[
            cite("2015 TLR 201", "xx/high/2015-201/v1", "overruled"),
            cite("1990 TLR 3", "xx/apex/1990-003/v1", "followed"),
        ],
    ))

    # D7 / D8 — adversarial: circular citations between companion appeals
    docs.append(judgment(
        "xx/high/2017-330/v1", "Orion Bank v. Pelagic Freight", HIGH, date(2017, 2, 8), "2017 TLR 330",
        [sec("Companion", "This appeal is heard together with Pelagic Freight v. Orion Bank and the reasoning there is adopted mutatis mutandis regarding the guarantee.")],
        out=[cite("2017 TLR 331", "xx/high/2017-331/v1", "cited")],
    ))
    docs.append(judgment(
        "xx/high/2017-331/v1", "Pelagic Freight v. Orion Bank", HIGH, date(2017, 2, 8), "2017 TLR 331",
        [sec("Companion", "For the construction of the guarantee, reference is made to Orion Bank v. Pelagic Freight decided concurrently.")],
        out=[cite("2017 TLR 330", "xx/high/2017-330/v1", "cited")],
    ))

    # D9 — adversarial: scanned judgment with OCR noise + low confidence
    docs.append(judgment(
        "xx/high/2016-410/v1", "Farah v. Testland Handwriting Board", HIGH, date(2016, 8, 22), "2016 TLR 410",
        [
            sec("Facts", "Th3 appe11ant challenged the admissibi1ity of expert handwriting evidence led to prove a disputed signature on a promissory note."),
            sec("Admissibility",
                "Opinion evidence of a handwriting expert is admissible but is weak evidence that must be corroborated before it can sustain a conviction or a civil finding.",
                "The court must itself compare the disputed and admitted signatures rather than surrender its judgment to the expert."),
            sec("Held", "The finding based solely on uncorroborated handwriting opinion is set aside."),
        ],
        quality=QualityReport(is_ocr=True, ocr_confidence=0.62, parse_score=0.55, warnings=["low OCR confidence", "character substitutions detected"]),
    ))

    # D3-duplicate — adversarial: reporter reprint of D3, shared dedup cluster
    dup = docs[4].model_copy(deep=True)
    dup = dup.model_copy(update={
        "doc_id": "xx/apex/2020-077/reprint",
        "source": prov("xx_apex_2020-077_reprint"),
        "quality": QualityReport(dedup_cluster_id="cluster-rahim-2020"),
    })
    # tag the original with the same cluster id so both are linked, neither deleted
    docs[4] = docs[4].model_copy(update={"quality": QualityReport(dedup_cluster_id="cluster-rahim-2020")})
    docs.append(dup)

    # D11 — off-topic trial-court noise (low authority)
    docs.append(judgment(
        "xx/trial/2019-905/v1", "Municipal Committee v. Trivial Vendor", TRIAL, date(2019, 5, 5), "2019 TLR 905",
        [sec("Order", "A minor licensing fee dispute over a roadside stall is resolved by directing payment of the outstanding municipal fee within thirty days.")],
        judges=["Judge M. Farooq"],
    ))

    # D12 — statute application referencing both statute versions
    docs.append(judgment(
        "xx/high/2018-220/v1", "Karim v. Revenue Board", HIGH, date(2018, 12, 1), "2018 TLR 220",
        [
            sec("Issue", "The question is which version of the Limitation Act governs an appeal filed in 2018 in respect of a cause accruing in 2014."),
            sec("Analysis",
                "The 2016 amendment to the Limitation Act shortened the appeal period but applies prospectively; rights of appeal accrued before the amendment are governed by the earlier text.",
                "Accordingly the appellant is entitled to the longer pre-amendment period."),
            sec("Held", "The appeal is held to be within time under the un-amended Limitation Act."),
        ],
        out=[
            cite("Limitation Act 1975 s. 3", "xx/statute/limitation-act/v1"),
            cite("Limitation (Amendment) Act 2016", "xx/statute/limitation-act/v2"),
        ],
    ))

    # S1 / S2 — statute version chain (point-in-time)
    docs.append(LegalDocument(
        doc_id="xx/statute/limitation-act/v1", doc_type=DocType.STATUTE, jurisdiction="xx",
        title="Limitation Act of Testland (1975)",
        citations_self=[Citation(raw="Limitation Act 1975", kind=CitationKind.STATUTE)],
        source=prov("xx_statute_limitation_v1"),
        version=DocVersion(version_label="1975", valid_from=date(1975, 1, 1), valid_to=date(2016, 6, 30)),
        body=[sec("Section 3 — Appeals",
                  "An appeal to the High Court shall be preferred within sixty days of the order appealed against.",
                  "The Court may admit an appeal after the prescribed period if satisfied that there was sufficient cause for not preferring it within time.")],
    ))
    docs.append(LegalDocument(
        doc_id="xx/statute/limitation-act/v2", doc_type=DocType.STATUTE, jurisdiction="xx",
        title="Limitation Act of Testland (as amended 2016)",
        citations_self=[Citation(raw="Limitation Act 2016", kind=CitationKind.STATUTE)],
        source=prov("xx_statute_limitation_v2"),
        version=DocVersion(version_label="2016", valid_from=date(2016, 7, 1), predecessor_doc_id="xx/statute/limitation-act/v1"),
        body=[sec("Section 3 — Appeals (amended)",
                  "An appeal to the High Court shall be preferred within thirty days of the order appealed against.",
                  "The Court may condone delay only upon a satisfactory explanation accounting for each day of delay.")],
    ))

    return docs


def build_queries() -> list[EvalQuery]:
    return [
        EvalQuery(qid="q1", text="principles for granting bail in economic offences",
                  query_type=QueryType.CONCEPTUAL, gold_doc_ids=["xx/apex/2018-050/v1"],
                  gold_citations=["xx/apex/2018-050/v1"]),
        EvalQuery(qid="q2", text="limitation period for filing an appeal and condonation of delay",
                  query_type=QueryType.STATUTE,
                  qrels=[Qrel(doc_id="xx/high/2019-114/v1", grade=3),
                         Qrel(doc_id="xx/high/2018-220/v1", grade=2),
                         Qrel(doc_id="xx/statute/limitation-act/v1", grade=2)],
                  gold_citations=["xx/high/2019-114/v1"]),
        EvalQuery(qid="q3", text="damages for remoteness in breach of contract reasonable contemplation",
                  query_type=QueryType.CONCEPTUAL,
                  qrels=[Qrel(doc_id="xx/apex/2020-077/v1", grade=3),
                         Qrel(doc_id="xx/apex/1990-003/v1", grade=3),
                         Qrel(doc_id="xx/high/2015-201/v1", grade=1),
                         Qrel(doc_id="xx/apex/2021-090/v1", grade=1)],
                  gold_citations=["xx/apex/2020-077/v1", "xx/apex/1990-003/v1"]),
        EvalQuery(qid="q4", text="2020 TLR 77",
                  query_type=QueryType.CITATION_LOOKUP, gold_doc_ids=["xx/apex/2020-077/v1"],
                  gold_citations=["xx/apex/2020-077/v1"]),
        EvalQuery(qid="q5", text="Rahim Trading Delta Mills",
                  query_type=QueryType.PARTY_LOOKUP,
                  gold_doc_ids=["xx/apex/2020-077/v1", "xx/apex/2020-077/reprint"],
                  gold_citations=["xx/apex/2020-077/v1"]),
        EvalQuery(qid="q6", text="is Zephyr Textiles still good law on remoteness overruled",
                  query_type=QueryType.MULTI_HOP,
                  qrels=[Qrel(doc_id="xx/apex/2021-090/v1", grade=3),
                         Qrel(doc_id="xx/high/2015-201/v1", grade=2),
                         Qrel(doc_id="xx/apex/1990-003/v1", grade=1)],
                  gold_citations=["xx/apex/2021-090/v1", "xx/high/2015-201/v1"]),
        EvalQuery(qid="q7", text="current limitation period after the 2016 amendment thirty days",
                  query_type=QueryType.STATUTE,
                  qrels=[Qrel(doc_id="xx/statute/limitation-act/v2", grade=3),
                         Qrel(doc_id="xx/high/2018-220/v1", grade=2)],
                  gold_citations=["xx/statute/limitation-act/v2"], as_of=date(2020, 1, 1)),
        EvalQuery(qid="q8", text="admissibility of expert handwriting evidence corroboration",
                  query_type=QueryType.CONCEPTUAL, gold_doc_ids=["xx/high/2016-410/v1"],
                  gold_citations=["xx/high/2016-410/v1"]),
        EvalQuery(qid="q9", text="penalty for cryptocurrency fraud under the Testland Digital Assets Act",
                  query_type=QueryType.UNANSWERABLE, answerable=False),
        EvalQuery(qid="q10", text="1990 TLR 3",
                  query_type=QueryType.CITATION_LOOKUP, gold_doc_ids=["xx/apex/1990-003/v1"],
                  gold_citations=["xx/apex/1990-003/v1"]),
    ]


def main() -> None:
    docs = build_documents()
    queries = build_queries()
    n_docs = write_documents(docs, HERE / "documents.jsonl")
    n_q = write_queries(queries, HERE / "queries.jsonl")
    print(f"wrote {n_docs} documents and {n_q} queries to {HERE}")


if __name__ == "__main__":
    main()
