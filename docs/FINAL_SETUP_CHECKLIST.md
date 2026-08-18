# Final setup checklist — from prototype to a running, tested version

This is the list of steps **you** need to run to take Legal-RAG from its current
state (a validated, hermetically-tested engine that runs locally on the mid-tier
PC) to a full-scale, backed, reproducible system on the RTX 3080 box.

Everything the assistant could do without your hardware, credentials, or an
interactive install is **already done and committed**. Each step below is blocked
on one of exactly three things: **(H)** hardware you haven't switched to yet,
**(I)** an interactive/admin install, or **(D)** a decision or credential only you
can provide. The tag is shown on each step.

> **What already works today, with nothing below done:** the E0 smoke pipeline
> (`legalrag eval -e e0_naive_baseline --smoke`), the full local stack on the CPU
> box (bge-small + MiniLM + qwen2.5:1.5b via Ollama), the citation graph on real
> CourtListener metadata, and the JSON run tracker. 74 tests green, ruff +
> mypy(strict) clean. Nothing below is required for development or CI.

Legend: ☐ = you do it · **(H/I/D)** = why it's yours to do.

---

## Stage 0 — the current box (GTX 1650 4 GB / 8 GB RAM) — ✅ done and verified

The full local stack **and** the serving layer run here today. Recipe, as installed:

```powershell
# CPU torch on purpose: Ollama drives the GPU for generation through its own
# runtime, so torch+CUDA would only speed up the 33M/22M embed+rerank models
# while costing ~5 GB of disk and contending with Ollama for the same 4 GB VRAM.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[embeddings,llm,lexical,graph,verify,serve,mcp]"
```

Deliberately **not** installed on this box:

| Extra | Why skipped |
|---|---|
| `citations` | `eyecite` → `fast-diff-match-patch` has no cp314 wheel and needs MSVC Build Tools (admin install). Only needed on `jurisdiction/us`. |
| `stores` | Qdrant needs Docker Desktop, not installed here. The `memory` index is fine at fixture/small-slice scale. |
| `tracking` | MLflow wants a server; the JSON run tracker in `runs/` already works. |
| `parsing` | `docling` is multi-GB and nothing uses it yet. |

Model sizing for 4 GB VRAM / 8 GB RAM — use the `_small` configs, **not** the
3080 ones. `e5_bge_m3_local` / `e9_local_grounded` (BGE-M3 + 7B) will thrash here:

| Works here | Model set |
|---|---|
| `e0_naive_baseline`, `e1_hybrid`, `e7_graph` | deterministic, no models |
| `e5_bge_small_local` | bge-small-en-v1.5 (33M) + ms-marco-MiniLM-L-6-v2 (22M), CPU |
| `e9_local_grounded_small` | the above + `qwen2.5:1.5b-instruct` on the GPU via Ollama |

Serving layer, both front doors verified on this box:

```powershell
legalrag serve http -e e0_naive_baseline --port 8000   # /health /answer /search /case/{doc_id}
legalrag serve mcp  -e e0_naive_baseline               # legal_answer / legal_search / legal_case
```

> **Known result on the 1.5B:** `e9_local_grounded_small` abstains on ~44 % of
> fixture queries (`answer_rate` 0.56) where the extractive generator answers all
> of them. Hallucinated-citation rate stays **0**, so the safety bar holds — the
> small model is simply under-answering. Expect this to close with a 7B.

---

## Stage 1 — Set up the RTX 3080 box

- ☐ **(I)** Install Python 3.11–3.13 (the code targets `>=3.11`; 3.14 works here
  but 3.12/3.13 have the widest wheel coverage for torch/mlflow). Clone the repo.
- ☐ **(H,I)** Install **CUDA-enabled** PyTorch (the mid-tier box has `torch+cpu`):
  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  python -c "import torch; print(torch.cuda.is_available())"   # must print True
  ```
- ☐ **(I)** Install the project with the extras you'll use:
  ```powershell
  pip install -e ".[embeddings,llm,stores,lexical,citations,graph,tracking,verify,dev]"
  ```
- ☐ **(I)** Confirm the gate still passes on the new box:
  ```powershell
  python -m pytest -q; python -m ruff check src tests; python -m mypy
  ```
  Expected: all tests green, ruff clean, mypy clean.

## Stage 2 — Local models on the GPU

- ☐ **(I)** Install Ollama and pull the full-size generation model (the 1.5B was a
  CPU stopgap; the 7B is the intended local generator):
  ```powershell
  winget install Ollama.Ollama
  ollama pull qwen2.5:7b-instruct
  ollama serve   # or let the tray app run it on :11434
  ```
- ☐ **(H)** The embedder/reranker configs `e5_bge_m3_local.yaml` and
  `e9_local_grounded.yaml` already point at **BGE-M3** + **bge-reranker-v2-m3** +
  **qwen2.5:7b-instruct**. These OOM/crawl on CPU by design — they are meant for
  this box. No code change needed; just run them (Stage 5).

## Stage 3 — Stand up the backing services (Docker)

- ☐ **(I)** Install Docker Desktop (needs WSL2 + admin + a reboot — that's why it's
  yours):
  ```powershell
  winget install Docker.DockerDesktop
  # reboot, launch Docker Desktop once, then:
  docker compose -f infra/docker-compose.yml up -d
  docker compose -f infra/docker-compose.yml ps    # all healthy?
  ```
- ☐ **(I)** Copy `.env.example` → `.env` and keep the docker defaults (Qdrant,
  Postgres, MinIO, MLflow, Phoenix URLs are pre-filled).

## Stage 4 — Point the engine at the stack

- ☐ Set these in `.env` (already present as defaults — confirm the host/IP):
  ```bash
  QDRANT_URL=http://localhost:6333
  MLFLOW_TRACKING_URI=http://localhost:5000
  LOCAL_LLM_BASE_URL=http://localhost:11434/v1
  ```
- ☐ Vector store: an experiment uses Qdrant by adding
  `vector_index: {name: qdrant, url: http://localhost:6333, collection: legalrag}`.
  MLflow tracking turns on **automatically** once `MLFLOW_TRACKING_URI` is set and
  `[tracking]` is installed — `resolve_tracker()` handles it; the local `runs/`
  manifest is still written and attached to each MLflow run.
- ☐ **(D, optional) Two-PC LAN:** to run services on the 3080 and query from the
  dev box, follow `docs/deployment.md` §"The two-PC pattern" (find the 3080's LAN
  IP, open ports 6333/5000 on the private firewall, set the two URLs to that IP on
  the dev box). This gives one shared experiment history across both machines.

## Stage 5 — Real data + full-scale runs (the actual research)

- ☐ **(D)** Get a free CourtListener API token
  (courtlistener.com/help/api) and put it in `.env` as `COURTLISTENER_API_TOKEN=`.
  Anonymous pulls give metadata + citation edges only; the token unlocks **full
  opinion text** needed for real retrieval. *(US-track code lives on the
  `jurisdiction/us` branch — check it out first.)*
- ☐ **(H)** Pull a real full-text slice and build the index:
  ```powershell
  git checkout jurisdiction/us
  python -m legalrag.jurisdictions.us.courtlistener --query "<topic>" --limit 500 --out-dir evalsets/us/<slice>
  ```
- ☐ **(H)** Climb the ablation ladder on real data and compare in MLflow. The rungs
  and their configs already exist:
  | Rung | Config | What it adds |
  |---|---|---|
  | E0 | `e0_naive_baseline.yaml` | deterministic baseline |
  | E1 | `e1_hybrid.yaml` | BM25 + dense fusion |
  | E2 | `e2_rerank.yaml` | BGE cross-encoder rerank |
  | E5 | `e5_bge_m3_local.yaml` | BGE-M3 embeddings (GPU) |
  | E7 | `e7_graph.yaml` | citation-graph expansion |
  | E9 | `e9_local_grounded.yaml` | local 7B grounded generation |
  ```powershell
  legalrag eval -e e5_bge_m3_local
  legalrag eval -e e9_local_grounded
  legalrag compare   # paired-bootstrap significance across runs
  ```
- ☐ **(D)** Build a real US eval set (gold citations / relevance) beyond the
  hand-crafted `evalsets/us/mini/` — the synthetic fixture is a smoke aid, not a
  benchmark (ADR 0003). CLERC/COLIEE are the intended external benchmarks.

## Stage 6 — Remaining engine wiring (optional, not blocking)

These are the only engine adapters still stubbed; the JSONL corpora + JSON/MLflow
trackers work without them. The assistant can build these on request — none needs
your hardware, they were left as scope, not blockers:

- ☐ **Postgres** corpus/edge store adapter (documents, metadata, citation edges).
  Testable now against SQLite; ask the assistant to wire it.
- ☐ **MinIO / DVC** object store for raw PDFs + parsed JSON (data versioning).
- ☐ **Phoenix** tracing hooks for error-analysis runs.

## Stage 7 — Pakistan track (the headline research target)

Currently only the US prototype bed is built. The real contribution lives on
`jurisdiction/pakistan` and is **not started** — it's the largest remaining piece:

- ☐ **(D)** Source a Pakistan case-law corpus (PLD / SC & High Court judgments).
- ☐ **(D)** Build the `pk-cite` citation extractor (Pakistan reporter/citation
  format) — the analogue of `us_regex` on the US branch.
- ☐ **(D)** Assemble the **PakLegalQA** benchmark (the headline deliverable).
- ☐ Run the same E0→E9 ladder on Pakistani data via a `jurisdictions/pk/` pack.

---

## Definition of "done" — acceptance criteria for a final version

The system is a *running, tested final version* when:

1. ☐ Gate is green on the 3080 box (`pytest` + `ruff` + `mypy`) with GPU torch.
2. ☐ `docker compose ps` shows Qdrant + MLflow healthy; `legalrag eval` logs to
   MLflow and the run appears in the UI at `:5000`.
3. ☐ A real full-text slice (CourtListener token) is indexed in Qdrant and served.
4. ☐ The full ladder E0→E9 has run on a **real** eval set, with
   `hallucinated_citation_rate == 0` preserved at every rung (the RQ3 safety bar).
5. ☐ `legalrag compare` shows a statistically significant lift for at least one
   rung over E0 on real data.
6. ☐ (Research goal) the same pipeline runs on the Pakistan pack against PakLegalQA.

Items 1–5 are US-prototype completion and are unblocked the moment you're on the
3080 box with a token. Item 6 is the multi-month research contribution.
