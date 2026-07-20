"""Local-first stack (ADR 0005): registration, config wiring, and the grounded
generator's contract with the deterministic verifier.

These are hermetic — no model download, no LLM server. The LLM is a stub client,
so we test the *plumbing and trust boundary* (does a grounded answer's citations
survive verification, and are fabrications dropped?), not model quality.
"""

from __future__ import annotations

import json

from legalrag.core import builtins as _builtins  # noqa: F401  (registers components)
from legalrag.core.config import load_experiment
from legalrag.core.corpus import Corpus
from legalrag.core.generate.generators_external import GroundedGenerator, _extract_json
from legalrag.core.interfaces import AssembledContext, Stage
from legalrag.core.models import (
    DocType,
    LegalDocument,
    Paragraph,
    Provenance,
    ScoredChunk,
    Section,
)
from legalrag.core.registry import REGISTRY
from legalrag.core.verify.checks import DeterministicVerifier

_QUOTE = "The right of privacy is a penumbral right."


class _StubClient:
    """Canned LLM: returns whatever JSON string it was handed, ignoring the prompt."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    @property
    def model_id(self) -> str:
        return "stub:test"

    def complete(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        self.calls.append((system, user))
        return self.response


def _corpus_and_context() -> tuple[Corpus, AssembledContext]:
    doc = LegalDocument(
        doc_id="us/scotus/griswold/v1",
        doc_type=DocType.JUDGMENT,
        jurisdiction="us",
        title="Griswold v. Connecticut",
        source=Provenance(source="test"),
        body=[Section(paragraphs=[Paragraph(text=_QUOTE + " It shields the marital relationship.")])],
    )
    corpus = Corpus([doc])
    sc = ScoredChunk(
        chunk_id="us/scotus/griswold/v1#p0001",
        doc_id="us/scotus/griswold/v1",
        score=1.0,
        para_ids=["p0001"],
        text=_QUOTE + " It shields the marital relationship.",
    )
    return corpus, AssembledContext(text="[S1] ...", chunks=[sc])


# --- registration ---------------------------------------------------------

def test_local_components_registered() -> None:
    assert "bge_reranker" in REGISTRY.available(Stage.RERANKER)
    assert "grounded" in REGISTRY.available(Stage.GENERATOR)
    assert "sentence_transformer" in REGISTRY.available(Stage.EMBEDDER)


def test_local_experiment_configs_load() -> None:
    for name in ("e2_rerank", "e5_bge_m3_local", "e9_local_grounded"):
        cfg = load_experiment(name)
        assert cfg.name == name
        assert cfg.config_hash()  # validates + is reproducible
    e9 = load_experiment("e9_local_grounded")
    assert e9.generation.generator.name == "grounded"
    assert e9.retrieval.reranker.name == "bge_reranker"


# --- grounded generator + verifier trust boundary -------------------------

def test_grounded_answer_survives_verification() -> None:
    corpus, ctx = _corpus_and_context()
    stub = _StubClient(
        json.dumps(
            {
                "answer": "Griswold recognized a right of privacy.",
                "abstain": False,
                "citations": [{"source": 1, "quote": _QUOTE}],
            }
        )
    )
    gen = GroundedGenerator(client=stub)
    ans = gen.generate("Does the Constitution protect privacy?", ctx)
    assert not ans.abstained
    assert len(ans.citations) == 1
    assert ans.citations[0].doc_id == "us/scotus/griswold/v1"
    assert ans.citations[0].para_ids == ["p0001"]

    verified, report = DeterministicVerifier(corpus).verify(ans)
    assert report.passed
    assert len(verified.citations) == 1


def test_fabricated_quote_is_dropped_by_verifier() -> None:
    corpus, ctx = _corpus_and_context()
    stub = _StubClient(
        json.dumps(
            {
                "answer": "Griswold said something it did not.",
                "abstain": False,
                "citations": [{"source": 1, "quote": "a quote that is not in the source at all"}],
            }
        )
    )
    ans = GroundedGenerator(client=stub).generate("q", ctx)
    # generator faithfully maps to the real doc/paragraph; the *quote* is the lie,
    # so the deterministic verifier catches it.
    verified, report = DeterministicVerifier(corpus).verify(ans)
    assert not report.passed
    assert verified.citations == []
    assert verified.abstained


def test_unparseable_output_fails_closed() -> None:
    _, ctx = _corpus_and_context()
    ans = GroundedGenerator(client=_StubClient("I refuse to emit JSON.")).generate("q", ctx)
    assert ans.abstained
    assert ans.citations == []


def test_model_abstention_respected() -> None:
    _, ctx = _corpus_and_context()
    stub = _StubClient(json.dumps({"answer": "", "abstain": True, "citations": []}))
    ans = GroundedGenerator(client=stub).generate("q", ctx)
    assert ans.abstained


def test_out_of_range_source_ignored() -> None:
    _, ctx = _corpus_and_context()
    stub = _StubClient(
        json.dumps(
            {
                "answer": "ok",
                "abstain": False,
                "citations": [{"source": 7, "quote": _QUOTE}],  # only 1 source exists
            }
        )
    )
    ans = GroundedGenerator(client=stub).generate("q", ctx)
    assert ans.citations == []


def test_empty_context_abstains_without_calling_model() -> None:
    stub = _StubClient("should never be read")
    ans = GroundedGenerator(client=stub).generate("q", AssembledContext(text="", chunks=[]))
    assert ans.abstained
    assert stub.calls == []  # no LLM call when there is nothing to ground on


def test_extract_json_tolerates_fences_and_prose() -> None:
    obj = _extract_json('Sure!\n```json\n{"answer": "x", "citations": []}\n```\nHope that helps.')
    assert obj is not None and obj["answer"] == "x"
    assert _extract_json("no json here") is None
