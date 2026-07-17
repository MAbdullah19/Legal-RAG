"""Deterministic-first verification (master plan §5.7, RQ3).

Order matters: cheap deterministic checks that *cannot* be argued with run first.
A citation whose document or paragraph is not in the corpus is **dropped, never
emitted** — this alone eliminates the fabricated-citation failure mode
(*Mata v. Avianca*). Quote verification confirms quoted spans actually occur in
the cited paragraph. NLI/entailment (MiniCheck) is a later component that plugs
in after these; it escalates only flagged claims.
"""

from __future__ import annotations

from legalrag.core.corpus import Corpus
from legalrag.core.interfaces import Stage
from legalrag.core.models import (
    AnswerCitation,
    CheckResult,
    GeneratedAnswer,
    VerificationReport,
)
from legalrag.core.registry import register


def _normalize(s: str) -> str:
    return " ".join(s.split()).casefold()


@register(Stage.VERIFIER, "deterministic")
class DeterministicVerifier:
    """Citation-existence + quote verification. Drops unverifiable citations and
    reports what was dropped and why."""

    def __init__(self, corpus: Corpus, require_quote_match: bool = True) -> None:
        self.corpus = corpus
        self.require_quote_match = require_quote_match

    def _citation_ok(self, cit: AnswerCitation) -> tuple[bool, str]:
        doc = self.corpus.get(cit.doc_id)
        if doc is None:
            return False, f"doc {cit.doc_id!r} not in corpus"
        for pid in cit.para_ids:
            if doc.get_paragraph(pid) is None:
                return False, f"paragraph {cit.doc_id}#{pid} does not exist"
        if self.require_quote_match and cit.quote:
            haystack = _normalize(
                " ".join(self.corpus.paragraph_text(cit.doc_id, p) or "" for p in cit.para_ids)
            )
            if _normalize(cit.quote) not in haystack:
                return False, f"quote not found in {cit.doc_id}#{','.join(cit.para_ids)}"
        return True, ""

    def verify(
        self, answer: GeneratedAnswer
    ) -> tuple[GeneratedAnswer, VerificationReport]:
        kept: list[AnswerCitation] = []
        existence_bad: list[str] = []
        quote_bad: list[str] = []
        for cit in answer.citations:
            ok, reason = self._citation_ok(cit)
            if ok:
                kept.append(cit)
            elif "quote" in reason:
                quote_bad.append(f"{cit.pinpoint_key()}: {reason}")
            else:
                existence_bad.append(f"{cit.pinpoint_key()}: {reason}")

        checks = [
            CheckResult(
                name="citation_existence",
                passed=not existence_bad,
                detail="every cited doc/paragraph must exist in the corpus",
                offending=existence_bad,
            ),
            CheckResult(
                name="quote_match",
                passed=not quote_bad,
                detail="quoted spans must occur in the cited paragraph",
                offending=quote_bad,
            ),
        ]
        filtered = answer.model_copy(update={"citations": kept})
        # If the generator made claims but nothing survived, abstain rather than
        # present an unsupported answer.
        if answer.citations and not kept:
            filtered = filtered.model_copy(
                update={
                    "abstained": True,
                    "text": "Retrieved material could not be verified against the corpus; "
                    "no grounded answer is available.",
                }
            )
        report = VerificationReport(passed=all(c.passed for c in checks), checks=checks)
        return filtered, report
