# FutureverseAI — Models Guide

Every AI model the platform uses, **what each one does**, **how it flows through the pipeline**, **alternatives**, and **which is best for this project** (UPSC/NCERT study assistant — English-primary with Hindi/Devanagari content, running on a single GPU).

> This reflects the *active runtime* config (your `start_backend.ps1`), which overrides the code defaults. Where they differ, it's noted.

---

## 1. Model roster (what you're running)

| Slot | Model in use | Runtime | Device | Config var | Dim/Size |
|---|---|---|---|---|---|
| **Embedding** | `BAAI/bge-large-en-v1.5` | sentence-transformers | GPU (CUDA) | `RAG_EMBEDDING_MODEL` | 1024-dim |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | CrossEncoder | GPU (CUDA) | `RAG_RERANKING_MODEL` | cross-encoder |
| **KG extraction** | `qwen2.5:7b` | Ollama | GPU | `KG_EXTRACT_MODEL` | 7B |
| **Contextual chunking** | `qwen2.5:7b` | Ollama | GPU | `CONTEXTUAL_CHUNKING_MODEL` | 7B |
| **Query expansion** | `qwen2.5:7b` | Ollama | GPU | (multi-query stage) | 7B |
| **Answer generation** | `qwen2.5:32b` | Ollama | GPU | chat model / `RAG_ANSWER_MODEL` | 32B |
| **Quiz / notes (fast)** | `qwen2.5:14b` | Ollama | GPU | `FAST_MODEL` | 14B |
| *Optional* Keyword (KeyBERT) | `all-MiniLM-L6-v2` | sentence-transformers | GPU | `KEYWORD_EXTRACTION_METHOD=keybert` | 384-dim |
| *Optional* Audio (STT) | `faster-whisper base` | faster-whisper | GPU | `WHISPER_MODEL` | 74M |
| *Optional* Aux embedding | `TaylorAI/bge-micro-v2` | sentence-transformers | GPU | `AUXILIARY_EMBEDDING_MODEL` | 384-dim |

**Config inconsistencies found (worth cleaning up):**
- `KG_EXTRACT_MODEL` has **three different defaults** across the code: `config.py` → `deepseek-r1:32b`, `extractor.py` → `llama3.1:8b`, and your env → `qwen2.5:7b`. The **live KG path reads the env**, so `qwen2.5:7b` wins — but the stray defaults are confusing. Pick one.
- `EMBED_MODEL=nomic-embed-text` is only referenced by the **dead** LangGraph `pipeline.py`; the real embedding is `bge-large-en-v1.5` via sentence-transformers. Ignore `EMBED_MODEL`.
- The **answer model** in normal chat is whatever you select in the UI model dropdown (typically `qwen2.5:32b`); `RAG_ANSWER_MODEL` only feeds the dead standalone pipeline.

---

## 2. The full process — which model runs where

### A) Ingestion (when a document is uploaded)
```
PDF/text
  → 1. Chunking            (no model — hierarchical text splitter)
  → 2. Keyword extraction  (regex default; optional KeyBERT all-MiniLM or llama3)
  → 3. Contextual enrich   → qwen2.5:7b  writes a [CONTEXT] topic line per chunk
  → 4. Embedding           → bge-large-en-v1.5  → 1024-dim vectors → Weaviate (HNSW + BM25)
  ── in parallel ──
  → 5. KG extraction       → qwen2.5:7b (via atlas_rag/Ollama) → entities + relations → Neo4j
```

### B) Query (when a user asks a question)
```
Question
  → 1. Query expansion   → qwen2.5:7b  generates 3–4 query variations (multi-query)
  → 2. Embedding         → bge-large-en-v1.5  embeds the query (must match ingest model)
  → 3. Hybrid retrieval  → Weaviate BM25 (keywords) + vector (semantic), fused
  → 4. Reranking         → bge-reranker-v2-m3  cross-encoder re-orders the candidates
  → 5. KG injection      → graph_rag pulls related entities/paths from Neo4j
  → 6. Answer            → qwen2.5:32b  writes the grounded answer with [id] citations
```

Each stage's model is independent — you can swap any one without touching the others (except embedding, which must be **identical at ingest and query**, and changing it requires a full re-index).

---

## 3. Each slot: current choice, alternatives, and the better pick

### 3.1 Embedding model  →  turns text into search vectors
**Current:** `BAAI/bge-large-en-v1.5` (1024-dim, **English-only**).

| Alternative | Dim | Languages | Notes |
|---|---|---|---|
| **`BAAI/bge-m3`** | 1024 | **Multilingual** (incl. Hindi) | Dense+sparse+ColBERT; strong on long text |
| `intfloat/multilingual-e5-large` | 1024 | Multilingual | Excellent multilingual retrieval |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | English | Tiny/fast, lower accuracy (the code default) |
| OpenAI `text-embedding-3-large` | 3072 | Multilingual | Hosted (cost + data leaves box) |

**Better for you: `bge-m3`.** Your NCERT/UPSC content contains **Hindi/Devanagari**, and `bge-large-en-v1.5` is English-only — Hindi passages embed poorly, so Hindi queries under-retrieve. `bge-m3` is multilingual, same 1024-dim, same GPU cost. **Caveat:** switching embedding models **requires re-ingesting all documents** (old vectors are incompatible). If your corpus is truly English-only, keep `bge-large-en-v1.5`.

### 3.2 Reranker  →  precision re-ordering of retrieved chunks
**Current:** `BAAI/bge-reranker-v2-m3` (multilingual cross-encoder).

| Alternative | Notes |
|---|---|
| `BAAI/bge-reranker-large` | English-focused, strong |
| `jinaai/jina-reranker-v2-base-multilingual` | Multilingual, fast |
| `mixedbread-ai/mxbai-rerank-large-v1` | Top English quality |
| Cohere Rerank 3 (hosted) | Excellent, but cost + data leaves box |

**Better for you: keep `bge-reranker-v2-m3`.** It's multilingual (matches Hindi content), high quality, and runs locally on your GPU. This is already the right choice — the only fix needed was that it wasn't being *loaded* (now set via `RAG_RERANKING_MODEL`).

### 3.3 KG-extraction LLM  →  pulls entities/relations for the knowledge graph
**Current:** `qwen2.5:7b`.

| Alternative | Notes |
|---|---|
| **`qwen2.5:14b`** | Noticeably better entity/relation quality; still fast on 48 GB |
| `llama3.1:8b` | Comparable to 7B, solid extraction |
| `deepseek-r1:14b`/`32b` | Reasoning models extract cleaner relations, but slower |
| Hosted (GPT-4o-mini / Claude Haiku) | Best structured-JSON reliability, but cost + cloud |

**Better for you: `qwen2.5:14b`.** KG quality directly affects graph retrieval, and the jump from 7B→14B meaningfully improves entity atomicity and relation typing (fewer of the "empty Entity" extraction misses you saw). On a 48 GB GPU it's affordable. If ingestion speed matters more than graph richness, `qwen2.5:7b` is a fine balance.

### 3.4 Contextual-chunking LLM  →  writes a 1–2 line topic per chunk
**Current:** `qwen2.5:7b`.

| Alternative | Notes |
|---|---|
| **`qwen2.5:3b` / `llama3.2:3b`** | This is a light task; a 3B model is ~2× faster at similar quality |
| `qwen2.5:7b` | Current — good quality, moderate speed |
| Skip enrichment | Fastest ingest, but loses the [CONTEXT] grounding benefit |

**Better for you: `qwen2.5:3b`.** Generating a short topic summary is easy; a 3B model does it well and **halves the per-chunk ingest time** (this stage runs once per chunk and dominates ingestion latency). Keep 7B only if you notice topic quality dropping.

### 3.5 Query-expansion LLM  →  rewrites the question into variants
**Current:** `qwen2.5:7b`.

**Better for you: `qwen2.5:7b` (keep), or `3b` for latency.** Expansion runs on the hot path of every query; a smaller model shaves response time. Quality difference is small for query rewriting. If answer latency matters, drop to `qwen2.5:3b`.

### 3.6 Answer-generation LLM  →  writes the final cited answer
**Current:** `qwen2.5:32b` (selected in the chat UI).

| Alternative | VRAM (Q4) | Notes |
|---|---|---|
| **`qwen2.5:32b`** | ~20 GB | Best local quality; strong multilingual (Hindi) — great for Mains-style answers |
| `qwen2.5:14b` | ~9 GB | ~2× faster, still very good — best if you're on a 24 GB GPU |
| `llama3.1:8b` | ~5 GB | Fast, weaker reasoning/citations |
| `gemma2:27b` | ~16 GB | Strong alternative to Qwen |
| Hosted GPT-4o / Claude | — | Highest quality + reliability, but cost and data leaves your box |

**Better for you: `qwen2.5:32b` on the 48 GB GPU** for answer quality (UPSC answers reward depth + accuracy), **or `qwen2.5:14b`** if you deploy on a 24 GB instance or need faster responses. Qwen2.5 is the right family — it handles English + Hindi well and follows the citation/MCQ template reliably.

### 3.7 Optional models
- **Keyword (KeyBERT `all-MiniLM-L6-v2`)** — only if `KEYWORD_EXTRACTION_METHOD=keybert`; default is regex (free, no model). Keep regex unless BM25 recall is poor.
- **Audio (`faster-whisper base`)** — only for voice input. Alternatives: `small`/`medium` (more accurate, slower), or `large-v3` (best). Use `base` unless transcription quality matters.
- **Aux embedding (`bge-micro-v2`)** — used in the evaluations path; leave as-is.

---

## 4. Recommendation summary (best-for-your-case)

| Slot | Keep or change | Recommended | Why |
|---|---|---|---|
| Embedding | **Change** (needs re-index) | `bge-m3` | Multilingual — fixes Hindi retrieval; same dim/cost |
| Reranker | Keep | `bge-reranker-v2-m3` | Already ideal (multilingual, local) |
| KG extraction | Change | `qwen2.5:14b` | Cleaner entities/relations → better graph retrieval |
| Contextual chunking | Change | `qwen2.5:3b` | Light task; 2× faster ingest |
| Query expansion | Keep/opt | `qwen2.5:7b` (or `3b`) | Small task; drop to 3b for latency |
| Answer | Keep | `qwen2.5:32b` (48 GB) / `14b` (24 GB) | Best local quality; multilingual |

**Overall verdict for this project:** the LLM family (**Qwen2.5**) is well chosen — strong, multilingual, template-obedient. The one impactful change is the **embedding model: move to `bge-m3`** so Hindi content is actually retrievable (with a one-time re-ingest). Bumping **KG extraction to 14B** and **contextual chunking down to 3B** trades a little VRAM for better graph quality and faster ingestion. Reranker and answer model are already correct.

---

## 5. How to change a model (no code edits)

All models are env-driven — set in `start_backend.ps1` (local) or `backend/.env` (Docker), then restart:

```ini
RAG_EMBEDDING_MODEL=BAAI/bge-m3            # (re-index required after changing)
RAG_RERANKING_MODEL=BAAI/bge-reranker-v2-m3
KG_EXTRACT_MODEL=qwen2.5:14b
CONTEXTUAL_CHUNKING_MODEL=qwen2.5:3b
RAG_ANSWER_MODEL=qwen2.5:32b
```
For Ollama models, pull them first: `ollama pull qwen2.5:14b` etc. For sentence-transformers models (embedding/reranker), they download from HuggingFace on first load (or pre-cache them; you run `HF_HUB_OFFLINE=1`, so pull them once before enabling offline mode).

> Reminder: changing `RAG_EMBEDDING_MODEL` invalidates every stored vector — you must **re-ingest all documents** afterward. Changing any LLM (answer/KG/chunking) is safe and takes effect immediately.

---

---

## 6. Caching — what's cached now, and where to add it

Caching in a RAG system buys two things: **speed** (skip GPU/DB work you already did) and **consistency** (identical questions get identical answers). It also *indirectly improves results* — the latency you save lets you afford heavier retrieval (wider `k`, more query expansions, reranking a bigger pool) within the same response-time budget.

### 6.1 What's already cached (in the code today)

| Layer | What it caches | Where | Effect |
|---|---|---|---|
| **Model weights** | Embedder + reranker loaded once, kept in VRAM | `app.state.ef` / `app.state.rf` (startup) | No per-request model load |
| **LLM keep-alive** | Ollama keeps the model in VRAM for 5 min | `OLLAMA_KEEP_ALIVE=5m` | No reload between close-together queries |
| **Per-request query embeds** | Unique expanded queries embedded once per request | `utils.query_collection_with_hybrid_search` (`_embedding_cache`) | Avoids N× embeds for multi-query |
| **Query-embedding cache (Redis)** | `hash(model+query) → vector`, cross-request | `retrieval/embedding_cache.py` — **built, but OFF by default** | Skips re-embedding repeated questions |
| **Neo4j schema cache** | Which relationship types exist | `extractor._rel_type_exists` / `_schema_rel_cache` | Fewer schema round-trips |
| **Ingestion dedup** | SHA-256 of file/chunk content | `pipeline_agents` ChunkDedup | Skips re-embedding identical content |
| **Model files on disk** | HF / tiktoken downloads | `HF_HOME`, `TIKTOKEN_CACHE_DIR` | No re-download |
| **DB internal caches** | Weaviate vector/query cache, Neo4j page cache | Weaviate / Neo4j | Faster repeat queries at the DB layer |

Redis is present (used for sessions), and `aiocache` is already a dependency — but **neither is used to cache RAG results yet**.

### 6.2 Where to ADD caching (biggest wins first)

| # | Cache | Stores (key → value) | Speeds up | Better results? | Effort | Status |
|---|---|---|---|---|---|---|
| 1 | **Query-embedding cache** | `sha256(model+query)` → vector | Every query's GPU embed | Consistency | **Flip a flag** | Built — just enable |
| 2 | **Answer cache** | `sha256(query+kb+model)` → final answer | The **32B generation** (slowest step) | High consistency | Medium | To build |
| 3 | **Retrieval-result cache** | `sha256(query+kb+k)` → assembled sources | Whole hybrid-search + rerank + KG | Consistency | Medium | To build |
| 4 | **KG-query cache** | `sha256(query+scope)` → graph context | The ~100 Neo4j round-trips/query | — | Medium | To build |
| 5 | **Reranker-score cache** | `(query_hash, chunk_hash)` → score | Cross-encoder (per-candidate GPU) | — | Medium | To build |
| 6 | **Contextual-chunk topic cache** | `chunk_hash` → topic line | `qwen2.5` call per chunk at **ingest** | — | Low | To build |
| 7 | **Doc-summary cache** | `file_hash` → summary | Wasted per-doc LLM summary call | — | Low | To build |

**Why these, concretely for your workload (UPSC/MCQ practice):**
- Students hit the **same questions** repeatedly (MCQ sets, common topics). **#1 + #2 + #3** turn a repeat question from "embed → search → rerank → 32B generate" (seconds) into a Redis lookup (milliseconds). This is the single biggest user-facing win.
- **#4** removes the query-time Neo4j fan-out (the ~100 round-trips in `get_graph_source`) for repeated/similar queries — the cheapest way to cut KG latency without the risky query rewrite (M-14).
- **#6 + #7** speed **re-ingestion** (e.g., after the chunk-cap fix you're re-adding files): unchanged chunks skip the per-chunk LLM topic call.

### 6.3 Recommended order

1. **Enable #1 now** (free): set `ENABLE_RAG_EMBEDDING_CACHE=true` + `REDIS_URL=...`. Fail-open, model-keyed, zero risk.
2. **Add #2 (answer) + #3 (result)** — the biggest latency drop for repeated questions. *Only cache deterministic answers (temperature 0), which your RAG prompt already uses.*
3. **Add #4 (KG cache)** — cuts the read-path Neo4j load.
4. **Add #5–#7** as incremental polish.

### 6.4 How to wire it (safely)

- Use **`aiocache` with the Redis backend** (already a dependency) so the same `@cached` decorators work in dev (in-memory) and prod (Redis).
- **Key hygiene:** include the **embedding model + dim** in embedding/result keys and the **answer model** in answer keys, so a model change auto-invalidates (avoids serving vectors/answers from a different model).
- **Invalidation is the hard part:** when a knowledge base changes (file add/remove/**reindex**), **purge the result + answer caches for that KB** (e.g., a `rag:kb:{id}:*` key namespace). This matters right after the chunk-cap fix — once you re-add files, stale cached answers must be dropped so users see the newly-recovered content.
- **TTLs:** embedding cache ~24h; result/answer caches shorter (5–60 min) unless you have solid invalidation; KG cache ~15 min.
- **Never cache errors or empty results** (the `__NO_RESULTS__` sentinel) — only successful, non-empty responses.

*Companion docs: [`RAG_PIPELINE.md`](RAG_PIPELINE.md) (architecture), [`../SETUP.md`](../SETUP.md) (config), [`DEPLOY_EC2.md`](DEPLOY_EC2.md) (deployment).*
