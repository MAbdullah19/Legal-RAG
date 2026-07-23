# Serving layer — Legal-RAG as an agent knowledge base

Turns the research pipeline into a running, agent-facing knowledge base. One
framework-neutral core (`legalrag.service.core`) loads an experiment's pipeline
**once** and serves it; two thin adapters sit on top:

| Front door | Module | For |
|---|---|---|
| **MCP server** | `legalrag.service.mcp_server` | LLM agents (Claude Code / Desktop / Agent SDK) — tools auto-discovered, zero glue |
| **HTTP API** | `legalrag.service.api` (FastAPI) | web UIs, notebooks, scripts, non-MCP callers |

Both return the **same** response models (`legalrag.service.core`), so the two
never drift.

### Why serve *this* pipeline (not a raw vector search)

Every citation an agent receives has already passed the deterministic verifier —
the verbatim quote and paragraph are confirmed to exist in the source document.
Citations that fail are dropped before they ever reach the caller. An agent can
therefore quote what it gets back without re-checking. `abstained: true` means
the corpus doesn't support an answer (the pipeline refuses to guess).

## Install

```bash
pip install -e ".[mcp]"     # MCP server for agents
pip install -e ".[serve]"   # HTTP API
pip install -e ".[mcp,serve]"
```

The neutral core needs no extra — only the adapter you run does.

## Choosing the corpus / pipeline

Both adapters serve one experiment, selected by `LEGALRAG_EXPERIMENT`
(default `e0_naive_baseline`, fully local, no keys/services). Point it at any
config in `experiments/` — e.g. `e9_local_grounded_small` for real embeddings +
a local Ollama LLM, or a jurisdiction corpus once ingested.

## MCP (for agents)

```bash
legalrag serve mcp -e e0_naive_baseline               # convenience wrapper
# equivalently:
LEGALRAG_EXPERIMENT=e0_naive_baseline python -m legalrag.service.mcp_server
```

Runs over **stdio** by default (how an MCP client spawns it). For a shared server
on the LAN (e.g. the RTX 3080 box), set
`LEGALRAG_MCP_TRANSPORT=streamable-http` (or `sse`).

**Tools exposed:**

- `legal_answer(query)` → grounded answer + verified citations (doc_id, pinpoint, verbatim quote, source_url)
- `legal_search(query, k=10)` → ranked source passages, no generation
- `legal_case(doc_id)` → citation-graph standing: authority score + good-law / questioned / overruled

**Register with Claude Code** — add to your MCP config (`.mcp.json` /
`claude mcp add`):

```json
{
  "mcpServers": {
    "legal-rag": {
      "command": "python",
      "args": ["-m", "legalrag.service.mcp_server"],
      "env": { "LEGALRAG_EXPERIMENT": "e0_naive_baseline" }
    }
  }
}
```

The agent then sees `legal_answer` / `legal_search` / `legal_case` in its toolbox
and calls them on its own.

## HTTP API

```bash
legalrag serve http -e e0_naive_baseline --host 0.0.0.0 --port 8000   # convenience wrapper
# or:  uvicorn --factory legalrag.service.api:create_app
# or:  LEGALRAG_EXPERIMENT=e0_naive_baseline python -m legalrag.service.api
#      (host/port via LEGALRAG_HOST / LEGALRAG_PORT)
```

| Method | Route | Body / param | Returns |
|---|---|---|---|
| GET | `/health` | — | status, experiment, index_id, docs |
| POST | `/answer` | `{"query": "..."}` | `AnswerResponse` |
| POST | `/search` | `{"query": "...", "k": 10}` | `SearchResponse` |
| GET | `/case/{doc_id}` | path (doc_id may contain `/`) | `CaseNode` (404 if unknown) |

Interactive docs at `/docs` (Swagger). Example:

```bash
curl -s localhost:8000/answer -H "content-type: application/json" \
  -d '{"query": "damages for remoteness"}'
```

## In-process (same Python process)

No server needed — import the core directly:

```python
from legalrag.service import LegalRAGService

svc = LegalRAGService.load("e0_naive_baseline")   # builds the index once
resp = svc.answer("does a right to privacy cover contraception?")
resp.answer          # grounded text
resp.citations       # verified sources
resp.verified        # passed every verifier check
svc.search("privacy", k=5).passages
```

## Deployment note (multi-PC)

The pipeline is loaded and indexed **once at startup**, then reused per request
(calls are serialized with a lock — fine for a single-node server; scale out with
multiple workers). For the two-PC pattern in
[`docs/deployment.md`](../deployment.md), run this server on the RTX 3080 box next
to the local LLM + Qdrant, and let dev machines / agents reach it over the LAN
(`streamable-http` for MCP, or the HTTP API). Persisting the index (Qdrant
adapter) so startup skips re-embedding is the natural next step; today the
in-memory index rebuilds on boot.
