# Deployment & multi-PC setup

This project runs in three postures, cheapest first. You only move up when a
posture stops being enough — nothing here is required for development or CI.

| Posture | Storage | Tracking | Needs |
|---|---|---|---|
| **Smoke / CI** | in-memory + JSONL | JSON files (`runs/`) | nothing (ADR 0003) |
| **Local research** | in-memory + local models | JSON files | `.[embeddings,llm]`, Ollama |
| **Backed / multi-PC** | Qdrant, Postgres, MinIO | MLflow | Docker (this doc) |

Everything stays **local-first** (ADR 0005): the Docker services run on your own
hardware / LAN, so confidential corpora never leave your network.

---

## What the Docker stack is

`infra/docker-compose.yml` defines five stateless *services* the engine talks to
over the network. Docker packages each as a container so you don't hand-install
databases per machine; `docker compose up` starts them together.

| Service | Port | Role | Swapped in via |
|---|---|---|---|
| **Qdrant** | 6333 | dense vector store at scale + payload filters | `vector_index: {name: qdrant}` |
| **Postgres** | 5432 | documents, metadata, citation edges, eval sets | (ingestion, roadmap) |
| **MinIO** | 9000 | object store for raw PDFs + parsed JSON (DVC remote) | (ingestion, roadmap) |
| **MLflow** | 5000 | experiment tracking across runs | `MLFLOW_TRACKING_URI` |
| **Phoenix** | 6006 | trace inspection for error analysis | `PHOENIX_COLLECTOR_ENDPOINT` |

> **GPU note:** the RTX-3080 work (BGE-M3 embeddings, the local LLM) runs as
> *native* processes (sentence-transformers, Ollama), **not** in Docker. Only the
> stateless services go in containers, and those are CPU/IO — so you do **not** need
> the NVIDIA Container Toolkit. Keep model inference on the host; keep state in Docker.

---

## Install Docker (your one interactive step)

Docker Desktop on Windows needs WSL2 + admin + a reboot, so run it yourself:

```powershell
winget install Docker.DockerDesktop
# reboot, launch Docker Desktop once so the engine starts, then:
docker compose -f infra/docker-compose.yml up -d
docker compose -f infra/docker-compose.yml ps      # all healthy?
```

Credentials come from repo-root `.env` (copy from `.env.example`). Bring it down
with `... down` (add `-v` to also wipe the data volumes).

---

## Point the engine at the stack

Swapping storage is a **config change, not code** — every backend is a registered
component behind a Protocol.

```yaml
# in an experiment yaml — use Qdrant instead of the in-memory index
index:
  vector_index: {name: qdrant, url: http://localhost:6333, collection: legalrag}
```

```bash
# .env
QDRANT_URL=http://localhost:6333
MLFLOW_TRACKING_URI=http://localhost:5000
```

The `qdrant` index reads `QDRANT_URL` when no `url:` is given, so the same config
works on any machine by changing one env var.

---

## The two-PC pattern (dev box + RTX 3080)

Run the stack on **one** host and point the other at it — Qdrant/MLflow bind to
`0.0.0.0`, so they're reachable over the LAN.

```
   ┌─────────────────────────┐          ┌──────────────────────────┐
   │  mid-tier dev PC         │          │  RTX 3080 box (server)   │
   │  - authoring, small runs │  LAN     │  - docker compose up     │
   │  QDRANT_URL ─────────────┼────────► │    qdrant:6333           │
   │  MLFLOW_TRACKING_URI ────┼────────► │    mlflow:5000           │
   │                          │          │  - BGE-M3 embed (GPU)    │
   │                          │          │  - Ollama 7B (GPU)       │
   └─────────────────────────┘          └──────────────────────────┘
```

Setup:

1. On the 3080 box, find its LAN IP (`ipconfig`), e.g. `192.168.1.20`.
2. Allow the ports through its firewall (6333, 5000) on the private network.
3. On the dev box, set `QDRANT_URL=http://192.168.1.20:6333` and
   `MLFLOW_TRACKING_URI=http://192.168.1.20:5000`.

Now the 3080 builds the heavy index once, and **both** machines query it and log
every experiment into **one** MLflow — so the ablation ladder is a single
comparable history, not two disconnected ones.

**What Docker does not do:** it won't split one computation across both PCs
(that's clustering — out of scope). It gives reproducible services and a shared
backend, which is exactly what a two-machine research setup needs.

---

## Wiring status

- **Qdrant** vector index — implemented (`core/retrieve/vector_external.py`),
  drop-in for `memory`. Pure mapping unit-tested; integration validated when the
  stack is up.
- **MLflow / Postgres / MinIO** — services defined and env wired; engine-side
  adapters (MLflow tracker, Postgres corpus/edge store) are the next wiring step.
  Until then the JSON tracker + JSONL corpora work unchanged.
