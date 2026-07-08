# FutureverseAI — Setup & Run Guide

How to get the FutureverseAI RAG platform running from a clean checkout. For architecture and component details, see [`docs/RAG_PIPELINE.md`](docs/RAG_PIPELINE.md).

> This guide reflects the **file/path fixes** applied during the audit (see §6) — a clean checkout should now run on Linux, macOS, Windows, and Docker without machine-specific paths.

---

## 1. Prerequisites

**Runtimes**
- Python **3.11**
- Node **22** (engine cap is `<=22.x`)
- (Optional) NVIDIA GPU + CUDA 12.8 for the pinned `torch==2.11.0+cu128`; CPU works via the CPU wheel index.

**Backing services** (must be reachable from the backend)
| Service | Version | Purpose |
|---|---|---|
| Weaviate | 1.27+ (matches `weaviate-client` 4.x) | Vectors + BM25 |
| Neo4j | **5.x** (driver pinned to `5.28.*`) | Knowledge graph |
| MySQL | 8.x | Application data |
| Redis | 7.x | Sessions / cache |
| Ollama | latest | Local LLMs (KG extraction, contextual chunking, answers, embeddings) |

**Ollama models** — pull what your `.env` references, e.g.:
```bash
ollama pull llama3.1:8b      # KG_EXTRACT_MODEL
ollama pull qwen2.5:7b       # CONTEXTUAL_CHUNKING_MODEL
ollama pull qwen2.5:32b      # RAG_ANSWER_MODEL
ollama pull nomic-embed-text # if using the ollama embedding engine
```

---

## 2. Quick start — Docker (recommended)

The full stack (backend + Weaviate + Neo4j + MySQL + Redis + Ollama) is defined in `docker-compose.prod.yaml` with health checks, dependency ordering, GPU reservations, and Docker secrets.

```bash
# 1. Create the secret files referenced by the compose (gitignored):
mkdir -p secrets
openssl rand -hex 32          > secrets/webui_secret_key
printf '%s' 'your-neo4j-pass' > secrets/neo4j_password
printf '%s' 'your-weav-key'   > secrets/weaviate_api_key

# 2. Provide .env values used by the compose (MYSQL_PASSWORD, CORS_ALLOW_ORIGIN, etc.)
cp backend/.env.example backend/.env   # edit it (see §4)

# 3. Build & launch
docker compose -f docker-compose.prod.yaml up -d --build

# 4. Verify (see §5)
curl -fsS http://localhost:8080/health
```

---

## 2b. Run on GPU — Windows (RTX 5090 / Blackwell SM_120)

Your primary launcher is **`D:\FutureverseAI\start_backend.ps1`**. It sets the GPU
environment, starts Ollama, and launches uvicorn into `backend/`. Just run it:

```powershell
D:\FutureverseAI\start_backend.ps1
```

**How the models map to the GPU** (Blackwell SM_120 needs a split):

| Model | Runtime | Device | Driven by |
|---|---|---|---|
| Embedding (`bge-large-en-v1.5`) | sentence-transformers | **CUDA** | `DEVICE_TYPE=cuda` |
| Reranker (`bge-reranker-v2-m3`) | CrossEncoder | **CUDA** | `DEVICE_TYPE=cuda` + `RAG_RERANKING_MODEL` |
| KG / chunking / answer LLMs | Ollama | NVIDIA **Vulkan** | `VK_ICD_FILENAMES`, `GGML_VK_VISIBLE_DEVICES=1` |
| Whisper / KeyBERT | PyTorch | **CUDA** (auto) | `DEVICE_TYPE` / auto-detect |

**Why Ollama uses Vulkan, not CUDA:** Ollama's CUDA build doesn't yet support
Blackwell SM_120, so the script forces the NVIDIA Vulkan ICD. PyTorch
(`torch 2.11.0+cu128`) does support SM_120, so embeddings + reranker stay on CUDA.

**What the corrected script fixes vs. the previous one:**
- **Reranker now actually loads** — `RAG_RERANKING_MODEL=BAAI/bge-reranker-v2-m3` was
  unset, so hybrid search silently fell back to cosine-similarity ordering. It now runs on CUDA.
- **`OLLAMA_GPU_LAYERS` is set before `ollama serve`** (it was set after → never applied),
  so all layers offload to the GPU.
- A **stable `WEBUI_SECRET_KEY`** persists to `D:\FutureverseAI\.webui_secret_key`, so
  logins survive restarts.
- The **NVIDIA Vulkan ICD is auto-discovered** (no hardcoded driver-hash path), and
  vestigial vars (`ATLAS_RAG_PATH`, `RAG_EMBEDDING_DEVICE`) were removed.

**Verify GPU use while it runs:** `nvidia-smi` (PyTorch VRAM for embed/rerank) and
`ollama ps` (the LLM should show GPU, not 100% CPU). The startup banner also prints
the embedding/reranker/LLM names.

> Note: keep `RAG_EMBEDDING_MODEL` = `bge-large-en-v1.5` — it must match what your
> data was ingested with. Switching to multilingual `bge-m3` (better for Hindi) needs a full re-index.

## 3. Manual start (development)

### 3a. Backend
```bash
cd backend

# environment
cp .env.example .env            # then edit — see §4
# start the supporting services (Weaviate, Neo4j, MySQL, Redis, Ollama) however you run them

# dependencies  (CPU torch shown; for GPU use the cu128 index in requirements.txt)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt

# RUN — important: launch from the backend/ directory so the `futureverse_ai`
# package resolves via cwd. No PYTHONPATH needed (see §6).
uvicorn futureverse_ai.main:app --host 0.0.0.0 --port 8080
#   or:  bash start.sh            (Linux/macOS — generates WEBUI_SECRET_KEY, then runs uvicorn)
#   or:  start_windows.bat        (Windows)
```

### 3b. Frontend
```bash
# from the repo root
npm ci
npm run dev        # dev server (http://localhost:5173)
# or
npm run build      # production build into build/  (served by the backend)
```

---

## 4. Environment configuration

Copy `backend/.env.example` → `backend/.env` and set at minimum:

| Variable | Example | Required |
|---|---|---|
| `WEBUI_SECRET_KEY` | `openssl rand -hex 32` | **Yes** — and **shared/stable across replicas** (don't auto-generate per container) |
| `DATABASE_URL` | `mysql+pymysql://futureverseai:PASS@localhost:3306/futureverseai` | Yes |
| `VECTOR_DB` | `weaviate` | Yes |
| `WEAVIATE_HTTP_HOST` / `WEAVIATE_HTTP_PORT` / `WEAVIATE_API_KEY` | `localhost` / `8080` / … | Yes |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | `bolt://localhost:7687` / `neo4j` / … | If KG enabled |
| `ENABLE_NEO4J` | `true` | To turn on KG retrieval |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Yes (local LLMs) |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | **Must match between ingest and query** |
| `CORS_ALLOW_ORIGIN` | `http://localhost:5173` | Lock down for prod (default `*` warns) |

Feature flags that improve quality (documented at the bottom of `.env.example`):
`ENABLE_RAG_HYBRID_SEARCH` (default **true** now), `RAG_RERANKING_MODEL`, `RAG_STRICT_GROUNDING`, `ENABLE_RAG_EMBEDDING_CACHE`, `RAG_DEBUG_LOG`.

> **PYTHONPATH is not required.** Launch with `uvicorn futureverse_ai.main:app` from the `backend/` directory and the package resolves via cwd. The `.env.example` no longer ships a hardcoded path (§6).

---

## 5. Verify it's running

```bash
curl -fsS http://localhost:8080/health        # {"status": true}
curl -fsS http://localhost:8080/health/db      # database reachable
curl -fsS http://localhost:8080/health/ready   # per-dependency readiness (Weaviate/Neo4j/Ollama)
```
`/health/ready` returns HTTP 503 (with a per-dependency breakdown) until the critical services are up — use it as your orchestrator's readiness probe.

---

## 6. File/path issues found & fixed

These were corrected so a clean checkout runs anywhere (no machine-specific paths):

| Issue | Location | Fix |
|---|---|---|
| **Hardcoded debug-log path** `D:\FutureverseAI\logs`, written every query | `retrieval/utils.py` | Now portable (`./logs` default) and **opt-in** via `RAG_DEBUG_LOG` / `RAG_DEBUG_LOG_DIR` |
| **Hardcoded/incorrect `PYTHONPATH`** (`YOUR_PROJECT_ROOT\FutureverseAI`, pointed at repo root not `backend/`, and read too late to matter) | `backend/.env.example` | Removed; documented that cwd-based launch needs no `PYTHONPATH` |
| **Missing `__init__.py`** in `routers`, `models`, `utils`, `socket`, `storage`, `internal`, `retrieval/{vector,loaders,web,models}`, `utils/{images,mcp,telemetry}` (worked only as namespace packages) | `backend/futureverse_ai/**` | Added package markers for consistency/packaging |
| **`.gitattributes` had no `*.py` rule** → CRLF/LF drift on Windows checkouts | `.gitattributes` | Added `*.py text eol=lf` (run `git add --renormalize .` once) |

**Still flagged (not auto-fixed — dead/optional code):**
- `retrieval/graph/pipeline.py` is **unused dead code** with broken imports (`from config import …`, `from extractor import …`) and a cwd-relative `CHUNKS_DIR` default. Delete it, or fix the imports to `futureverse_ai.*` and the path before using it.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'futureverse_ai'` | Launch from the `backend/` directory (or set `PYTHONPATH=…/backend`). |
| A `D:\FutureverseAI\logs` or odd `logs` folder appears | The old hardcoded debug log; ensure you're on the patched `utils.py`. Leave `RAG_DEBUG_LOG=false`. |
| Retrieval returns nothing / dimension errors | `RAG_EMBEDDING_MODEL` must be identical for ingestion and query; changing it requires re-indexing. |
| Hybrid search not helping | Confirm Weaviate is up; `ENABLE_RAG_HYBRID_SEARCH` defaults on and falls back to vector-only if BM25 fails. |
| KG context empty | Set `ENABLE_NEO4J=true` + `NEO4J_*`; confirm Neo4j is reachable (`/health/ready`). |
| Ollama errors | `ollama pull <model>` for each of `KG_EXTRACT_MODEL`, `CONTEXTUAL_CHUNKING_MODEL`, `RAG_ANSWER_MODEL`. |
| `torch` installs CPU build / fails on GPU box | Install torch from the matching index (`--index-url …/cu128` for GPU, `…/cpu` otherwise) **before** `pip install -r requirements.txt`. Note: the Dockerfile pins `torch<=2.9.1`, which currently conflicts with `requirements.txt` (`2.11.0+cu128`) — reconcile for your hardware. |
| JWT/login breaks across replicas | `WEBUI_SECRET_KEY` must be one shared, stable value for all instances. |

---

*Companion docs: [`docs/RAG_PIPELINE.md`](docs/RAG_PIPELINE.md) (architecture, config, API) and the phase audit reports in the `RAG Pipeline/` folder.*

---

## 8. Project structure

```
FutureverseAI/                       # repo root - run from D:\FutureverseAI\FutureverseAI
  backend/
    futureverse_ai/                  # the app package: uvicorn futureverse_ai.main:app
      main.py                        # FastAPI app + lifespan + /health /health/ready
      env.py  config.py              # device (DEVICE_TYPE) + settings (PersistentConfig)
      models/                        # SQLAlchemy models (knowledge, files, users, chats)
      routers/                       # API endpoints (retrieval, knowledge, auths, ...)
      retrieval/                     # RAG core
        graph/                       #   Neo4j KG: extractor, graph_rag, semantic_promoter
        vector/                      #   Weaviate + other vector-DB connectors
        loaders/  web/               #   document loaders + web search
        utils.py                     #   hybrid search, embeddings, context injection
        pipeline_agents.py           #   ingestion agent/checker pipeline
        contextual_chunking.py       #   [CONTEXT] enrichment
        embedding_cache.py           #   optional Redis query-embedding cache (Phase 3)
      migrations/  internal/         # Alembic + internal DB
      socket/ storage/ tools/ utils/
    requirements.txt                 # PRIMARY dependency source (pip) - complete
    requirements-dev.txt             # dev/test tooling (Phase 2)
    start-backend.ps1 / start.sh     # launchers
    tests/                           # unit + integration scaffolding (Phase 4)
  src/  build/  static/              # SvelteKit frontend (dev: npm run dev -> :5173)
  weaviate/docker-compose.yml        # Weaviate service
  docker-compose.prod.yaml           # full prod stack (Phase 4)
  pyproject.toml  uv.lock            # SECONDARY dep source (uv) - incomplete, see Sec.9
  Dockerfile  .github/workflows/     # container + CI
  SETUP.md  docs/RAG_PIPELINE.md     # this guide + architecture
```

The GPU launcher `D:\FutureverseAI\start_backend.ps1` lives one level up (parent of the repo).

---

## 9. Dependency management & upgrades

**`requirements.txt` is the authoritative source** (complete; installed by the start scripts via pip). `pyproject.toml` + `uv.lock` are secondary and currently **incomplete** - they are missing `neo4j`, `python-igraph`, and `leidenalg`, so a `uv sync` install would break the knowledge-graph feature. Decide one before relying on uv:

- **Keep pip / `requirements.txt` primary** (recommended): treat `pyproject.toml`/`uv.lock` as build metadata only, or remove them to avoid confusion.
- **Move fully to uv:** add the missing deps to `pyproject.toml` (`neo4j==5.28.*`, `python-igraph`, `leidenalg`, + the rest), then `uv lock`.

**Controlled upgrade (on a machine with Python 3.11):**
```bash
# pip path (current)
cd backend
pip list --outdated            # see what's behind
# edit requirements.txt, then:
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/unit              # validate
# uv path (only after pyproject is completed)
uv lock --upgrade --dry-run    # preview
uv lock --upgrade && uv sync
```
Upgrade in small batches (LangChain family together; `transformers`+`sentence-transformers` together) and run a smoke ingest + query after each. Exact-pinned (`==`) packages don't move on `--upgrade`; the unpinned ones (`cryptography`, `redis`, `tiktoken`, `openai`, `anthropic`, `accelerate`, `psutil`, `sentencepiece`, `pymongo`, `python-igraph`, `leidenalg`, ...) will - pin them once validated. (aiohttp/openai/anthropic are now pinned in both files.)

---

## 10. Recent runtime fixes (from your last run)

- **Knowledge base "No content found"** - the file-list queries selected every large JSON column and then `ORDER BY`, overflowing MySQL `sort_buffer_size` (error 1038); the `except` returned an empty list. Fixed by deferring those columns (`models/knowledge.py`). Safety net: raise `sort_buffer_size = 4M` in MySQL `my.cnf`.
- **Frontend (5173) vs backend (8080) differed** - same backend, intermittent failing query. With the fix both are consistent. Running them separately, set `CORS_ALLOW_ORIGIN=http://localhost:5173` (not `*`) for clean credentialed CORS.
- **Log warnings are benign** - `pyasn1`, `websockets.legacy`, `pydub`/ffmpeg, `grpcio-status`, Pydantic V1 notices are third-party deprecation warnings, not errors. Optional: `pip install grpcio-status`, install `ffmpeg` for audio.
