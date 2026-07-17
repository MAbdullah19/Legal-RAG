"""Small, dependency-free text helpers shared by indexing/retrieval.

Deliberately simple: the smoke pipeline must run with no NLP downloads. Real
tokenizers/segmenters are registered components added later.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Used by the hashing embedder and BM25 so
    they share a vocabulary in the smoke path."""
    return _TOKEN_RE.findall(text.lower())


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~word count) for chunk sizing / context budgets."""
    return len(text.split())


def first_sentence(text: str, max_chars: int = 320) -> str:
    """First sentence, verbatim, trimmed — used for extractive quoting so the
    quote is always a substring of the source paragraph."""
    text = text.strip()
    if not text:
        return ""
    parts = _SENT_RE.split(text, maxsplit=1)
    sent = parts[0].strip()
    if len(sent) > max_chars:
        sent = sent[:max_chars].rsplit(" ", 1)[0]
    return sent
