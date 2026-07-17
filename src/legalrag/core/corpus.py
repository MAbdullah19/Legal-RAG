"""In-memory corpus + JSONL (de)serialization + corpus id resolution.

A :class:`Corpus` is the document registry the deterministic verifier checks
against (a cited doc/paragraph must exist here) and the source of pinpoint text.
JSONL is the on-disk snapshot format; real jurisdictions write the same format
from their ingestion connectors, so the fixture and a scraped PK corpus load
through identical code.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from legalrag.core.models import EvalQuery, LegalDocument

# Repo root: .../src/legalrag/core/corpus.py -> parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
EVALSETS_DIR = _REPO_ROOT / "evalsets"


class Corpus:
    def __init__(self, docs: Iterable[LegalDocument] = ()) -> None:
        self._docs: dict[str, LegalDocument] = {}
        for d in docs:
            self.add(d)

    def add(self, doc: LegalDocument) -> None:
        self._docs[doc.doc_id] = doc

    def get(self, doc_id: str) -> LegalDocument | None:
        return self._docs.get(doc_id)

    def paragraph_text(self, doc_id: str, pid: str) -> str | None:
        doc = self._docs.get(doc_id)
        if doc is None:
            return None
        para = doc.get_paragraph(pid)
        return para.text if para else None

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._docs

    def __len__(self) -> int:
        return len(self._docs)

    def __iter__(self) -> Iterator[LegalDocument]:
        return iter(self._docs.values())


# --------------------------------------------------------------------------
# JSONL (de)serialization
# --------------------------------------------------------------------------

def write_documents(docs: Iterable[LegalDocument], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(doc.model_dump_json())
            fh.write("\n")
            n += 1
    return n


def read_documents(path: Path) -> list[LegalDocument]:
    with path.open(encoding="utf-8") as fh:
        return [LegalDocument.model_validate_json(line) for line in fh if line.strip()]


def write_queries(queries: Iterable[EvalQuery], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(q.model_dump_json())
            fh.write("\n")
            n += 1
    return n


def read_queries(path: Path) -> list[EvalQuery]:
    with path.open(encoding="utf-8") as fh:
        return [EvalQuery.model_validate_json(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------
# Corpus id resolution
# --------------------------------------------------------------------------

def corpus_dir(corpus_id: str) -> Path:
    """``fixture`` -> ``evalsets/fixture``; ``us/clerc-dev-10k`` ->
    ``evalsets/us/clerc-dev-10k`` (convention shared by all jurisdictions)."""
    return EVALSETS_DIR.joinpath(*corpus_id.split("/"))


def load_corpus(corpus_id: str) -> Corpus:
    docs_path = corpus_dir(corpus_id) / "documents.jsonl"
    if not docs_path.exists():
        raise FileNotFoundError(f"corpus {corpus_id!r} not found at {docs_path}")
    return Corpus(read_documents(docs_path))


def load_eval_queries(corpus_id: str) -> list[EvalQuery]:
    q_path = corpus_dir(corpus_id) / "queries.jsonl"
    if not q_path.exists():
        return []
    return read_queries(q_path)


def as_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)
