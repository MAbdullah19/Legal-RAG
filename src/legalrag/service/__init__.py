"""Serving layer: expose the RAG pipeline as an agent-facing knowledge base.

- :mod:`legalrag.service.core` — framework-neutral service + response models.
- :mod:`legalrag.service.mcp_server` — MCP tools for LLM agents (lazy: needs ``[mcp]``).
- :mod:`legalrag.service.api` — FastAPI HTTP service (lazy: needs ``[serve]``).

Importing this package pulls in only the neutral core; the two adapters import
their frameworks lazily so ``core`` stays dependency-light.
"""

from __future__ import annotations

from legalrag.service.core import (
    AnswerResponse,
    CaseNode,
    Citation,
    LegalRAGService,
    Passage,
    SearchResponse,
)

__all__ = [
    "AnswerResponse",
    "CaseNode",
    "Citation",
    "LegalRAGService",
    "Passage",
    "SearchResponse",
]
