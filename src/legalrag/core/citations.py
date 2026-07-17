"""Citation resolution — jurisdiction-neutral (master plan §4, §5.5).

An extractor's job is to find citation *mentions* in a document's text; the
resolver's job is to link each mention to the document it refers to. A mention
resolves when its normalized citation key matches a target document's
``citations_self`` (how that document is cited), or — for case-name references —
the target's title. Resolution is what turns free-text citations into the
``citation_edges`` that :class:`~legalrag.core.graph.citation_graph.CitationGraph`
consumes, so it is shared by every jurisdiction pack (US eyecite/regex, PK
``pk-cite``, …).
"""

from __future__ import annotations

from legalrag.core.corpus import Corpus
from legalrag.core.models import CitationMention


def _norm(s: str) -> str:
    return " ".join(s.split()).casefold()


class CitationResolver:
    def __init__(self, corpus: Corpus) -> None:
        self._by_key: dict[str, str] = {}
        self._by_title: dict[str, str] = {}
        for doc in corpus:
            for c in doc.citations_self:
                self._by_key.setdefault(c.key(), doc.doc_id)
                if c.normalized:
                    self._by_key.setdefault(_norm(c.normalized), doc.doc_id)
                self._by_key.setdefault(_norm(c.raw), doc.doc_id)
            self._by_title.setdefault(_norm(doc.title), doc.doc_id)

    def resolve_mention(self, mention: CitationMention) -> str | None:
        """The target doc_id for a mention, or None if it refers outside the
        corpus (a dangling citation — tracked, never fabricated)."""
        if mention.key() in self._by_key:
            return self._by_key[mention.key()]
        return self._by_title.get(_norm(mention.raw))

    def resolve_corpus(self, corpus: Corpus, overwrite: bool = False) -> int:
        """Populate ``target_doc_id`` on every resolvable mention in place.
        Returns the number of edges newly resolved."""
        resolved = 0
        for doc in corpus:
            for m in doc.citations_out:
                if m.target_doc_id is not None and not overwrite:
                    continue
                target = self.resolve_mention(m)
                if target is not None and target != doc.doc_id:
                    m.target_doc_id = target
                    resolved += 1
        return resolved

    def resolution_rate(self, corpus: Corpus) -> float:
        """Fraction of mentions that resolve to an in-corpus document — a core
        ingestion quality metric (master plan §5.1)."""
        total = 0
        hit = 0
        for doc in corpus:
            for m in doc.citations_out:
                total += 1
                if self.resolve_mention(m) is not None:
                    hit += 1
        return hit / total if total else 0.0
