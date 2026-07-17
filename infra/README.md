# Development infrastructure

Backing services for **adopted** integrations. The E0 smoke pipeline and CI need
none of them (ADR 0003) — bring them up as the corresponding components are
wired on the US track.

```bash
cp .env.example .env          # fill in secrets if any; defaults match this stack
docker compose -f infra/docker-compose.yml up -d
```

| Service | Port | Wired in by | Purpose |
|---|---|---|---|
| Qdrant | 6333 | `qdrant` VectorIndex component | hybrid dense search, payload filters at scale |
| PostgreSQL | 5432 | metadata + `citation_edges` store | canonical rows, citation graph (NetworkX reads it) |
| MinIO | 9000/9001 | ingestion + DVC remote | raw PDFs, parsed JSON, corpus/index snapshots |
| MLflow | 5000 | `mlflow` RunTracker | experiment tracking (swaps the JSON tracker) |
| Phoenix | 6006 | OTel tracing | trace-level retrieval/generation error analysis |

Each maps to a registered component or a config target, so swapping the
in-process default for the service-backed implementation is a config change, not
a code change.
