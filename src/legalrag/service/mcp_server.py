"""MCP adapter — exposes the knowledge base as tools for LLM agents.

An MCP-capable agent (Claude Code, Claude Desktop, the Agent SDK, ...) points at
this server and gets three tools with no glue code:

- ``legal_answer(query)`` — a grounded answer whose every citation the
  deterministic verifier has confirmed exists in the source corpus.
- ``legal_search(query, k)`` — ranked source passages, for the agent to reason
  over itself.
- ``legal_case(doc_id)`` — a case's citation-graph standing: authority score and
  whether it is still good law.

Run (stdio, the default an MCP client spawns)::

    LEGALRAG_EXPERIMENT=e0_naive_baseline python -m legalrag.service.mcp_server

Over a LAN (e.g. the 3080 box serving other machines), set
``LEGALRAG_MCP_TRANSPORT=streamable-http`` (or ``sse``).

Requires the ``mcp`` extra: ``pip install -e ".[mcp]"``.
"""

from __future__ import annotations

import os
from typing import Any

from legalrag.service.core import DEFAULT_EXPERIMENT, LegalRAGService

EXPERIMENT_ENV = "LEGALRAG_EXPERIMENT"
TRANSPORT_ENV = "LEGALRAG_MCP_TRANSPORT"


def build_server(experiment: str | None = None) -> Any:
    """Construct a FastMCP server whose tools serve one loaded experiment.

    The pipeline is loaded and indexed once here; each tool call reuses it.
    """
    # mcp 2.x renamed ``FastMCP`` to ``MCPServer``; the ``.tool()`` decorator and
    # ``.run(transport=...)`` surface are identical. pyproject allows mcp>=1.2, so
    # support both rather than force an upgrade.
    server_cls: Any
    try:
        from mcp.server import MCPServer

        server_cls = MCPServer
    except ImportError:  # pragma: no cover - mcp 1.x
        from mcp.server.fastmcp import FastMCP

        server_cls = FastMCP

    exp = experiment or os.environ.get(EXPERIMENT_ENV, DEFAULT_EXPERIMENT)
    service = LegalRAGService.load(exp)

    mcp = server_cls(
        "legal-rag",
        instructions=(
            "Grounded question answering and retrieval over a judicial corpus. "
            "Every citation returned by legal_answer is verified to exist in the "
            "source documents (verbatim quote checked), so cite them directly. "
            "Use legal_search when you want raw source passages to reason over, "
            "and legal_case to check whether a decision is still good law."
        ),
    )

    @mcp.tool()
    def legal_answer(query: str) -> dict[str, Any]:
        """Answer a legal question with citation-verified sources.

        Returns the answer text plus a list of citations (doc_id, pinpoint,
        verbatim quote, source_url); every citation is grounded in the corpus.
        ``abstained`` is true when the corpus does not support an answer."""
        return service.answer(query).model_dump()

    @mcp.tool()
    def legal_search(query: str, k: int = 10) -> dict[str, Any]:
        """Retrieve the top-``k`` source passages for a query (no generation).

        Returns ranked passages with doc_id, title, pinpoint, text and score."""
        return service.search(query, k=k).model_dump()

    @mcp.tool()
    def legal_case(doc_id: str) -> dict[str, Any]:
        """Look up a case's citation-graph standing by ``doc_id``.

        Returns its authority score, whether it is good law / questioned /
        overruled, and its citing / cited-by neighbours."""
        node = service.case(doc_id)
        if node is None:
            return {"error": f"{doc_id!r} not in corpus", "doc_id": doc_id}
        return node.model_dump()

    return mcp


def main() -> None:
    server = build_server()
    transport = os.environ.get(TRANSPORT_ENV, "stdio")
    server.run(transport=transport)


if __name__ == "__main__":
    main()
