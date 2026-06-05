# Complete RAG Pipeline: Ingestion → Query → Answer
## FutureverseAI / Infinity Foundation NCERT Corpus
### SHC-CE Pipeline v2 — Post-Audit Specification

> **Generated from live code.** Every step below was verified against the
> actual implementation before being documented. File:line references are
> current as of this audit session.

---

## CHANGELOG — Audit & Fix Session

| Issue | Files | Change | Status |
|-------|-------|--------|--------|
| **Issue 1** | `utils.py`, `weaviate.py`, `retrieval.py` | Strategy B context headers via revived `ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS` flag; `_build_context_header()` helper; `header_1`/`header_2`/`breadcrumb` schema properties added | ✅ Done |
| **Issue 2** | `utils.py` | Parent dedup after reranking (two-pass, max-score-wins per `parent_id`); `get_sources_from_items` last-write-wins → max-score-wins | ✅ Done |
| **Issue 3** | `retrieval.py` | uuid4 chunk_id deferred — no active corruption; tech debt comment added at both assignment sites | 🔵 Deferred |
| **Issue 4** | `utils.py`, `config.py` | Sigmoid normalization of cross-encoder logits; `RAG_RELEVANCE_THRESHOLD` default 0.0 → 0.3 (**BREAKING**) | ✅ Done |
| **Issue 5** | `utils.py` | Explicit 3-step pipeline (ensemble → SHA256 dedup → compressor) replaces `ContextualCompressionRetriever`; dedup moved before reranking | ✅ Done |
| **Issue 6** | `utils.py` | numpy cosine query dedup (threshold 0.95) replaces pure-Python Jaccard; single BLAS matrix call | ✅ Done |
| **Issue 7** | `contextual_chunking.py` | Exponential backoff retry (2 retries, 1s/2s); circuit breaker after 5 consecutive failures; `OLLAMA_NUM_PARALLEL` startup warning; `FALLBACK_COUNTER` metric | ✅ Done |
| **Bonus** | `utils.py` | `[CONTEXT]...[CONTENT]` block stripped from `parent_text` unconditionally before header injection | ✅ Done |

---

```
╔══════════════════════════════════════════════════════════════╗
║              PHASE 1 — INGESTION                             ║
║   save_docs_to_vector_db()  ·  retrieval.py:1619             ║
╚══════════════════════════════════════════════════════════════╝
```

### Step 1 — Document Upload & Raw Text Extraction

User uploads a file (PDF / DOCX / MD / TXT) via the UI to a Knowledge Base
or directly into a chat session. The configured content extraction engine
(Marker / Docling / PyPDF / Tika) converts the file to raw markdown text,
preserving `#`, `##`, `###` headers.

```
retrieval.py → process_file() → save_docs_to_vector_db()
```

`full_doc_text` is captured immediately from joined `page_content` of all
input docs — **before any splitting** — and passed to contextual chunking as
the document-level context window.

```python
# retrieval.py:1672
full_doc_text = " ".join(doc.page_content for doc in docs)
```

---

### Step 2 — Markdown Header Split (Layer 1)

`MarkdownHeaderTextSplitter` splits on six header levels → `parent_docs`.
Each resulting doc carries header metadata in its `doc.metadata` dict.

```python
# retrieval.py:1683-1709
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#",      "Header 1"),
        ("##",     "Header 2"),
        ("###",    "Header 3"),
        ("####",   "Header 4"),
        ("#####",  "Header 5"),
        ("######", "Header 6"),
    ],
    strip_headers=False,
)
```

Headers are preserved in `page_content` (not stripped) so the full section
text including its title reaches the LLM. After splitting, an optional
`merge_docs_to_target_size()` pass merges sections below `CHUNK_MIN_SIZE_TARGET`
to prevent fragment chunks.

---

### Step 3 — Hierarchical Parent–Child Split (Layer 2)

**Active strategy: `"hierarchical"`** (set via `CHUNKING_STRATEGY` env var or
Admin → Documents → Chunking Strategy).

A second splitter (`RecursiveCharacterTextSplitter`, child size ≈ `CHUNK_SIZE // 4`)
further splits each markdown section (parent) into indexed child chunks.

```python
# retrieval.py:1760-1794  (simplified)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=child_chunk_size,
    chunk_overlap=child_overlap,
)

child_docs = []
for parent in parent_docs:
    pid  = str(uuid.uuid4())     # parent group ID — see Step 5
    ptxt = parent.page_content
    crumbs = [parent.metadata[h]
               for h in ("Header 1","Header 2","Header 3","Header 4")
               if parent.metadata.get(h)]
    breadcrumb = " > ".join(crumbs)
    for child in child_splitter.split_documents([parent]):
        child.metadata["parent_id"]   = pid
        child.metadata["parent_text"] = ptxt
        if breadcrumb:
            child.metadata["breadcrumb"] = breadcrumb
        child_docs.append(child)
```

After this step each child doc carries:
- `parent_id` — shared UUID linking siblings
- `parent_text` — full section text (4× child size)
- `breadcrumb` — `"H1 > H2 > H3"` display label

---

### Step 4 — Keyword Extraction

```python
# retrieval.py:1860-1895
kw = _extract_chunk_keywords(raw, doc.metadata, method=_kw_method, ollama_url=_ollama_url)
doc.metadata["keywords"] = kw
```

Three methods dispatched by `_kw_method`:
- `"regex"` — fast NER-lite, no model (default)
- `"keybert"` — sentence-transformers semantic keywords
- `"llama3"` — Ollama Llama-3 extraction (~1s/chunk, highest quality)

> **Note:** Keyword extraction currently runs **before** contextual chunking
> (Step 5). Ideally it should run after, so keywords can be extracted from the
> enriched text. Not refactored in this audit session.

---

### Step 5 — Layer 4 MGI Metadata Assignment ✨ NEW

Pre-assigns stable UUIDs and computes all sibling/child relationship fields
before Weaviate insert so cross-chunk links are consistent.

**Step A** — assign `chunk_id` to every child:

```python
# retrieval.py:1800-1812
for doc in child_docs:
    # chunk_id / parent_id: uuid4 (random per ingest cycle).
    # Safe because overwrite=True purges the full collection before rebuild
    # and no external index currently keys on these IDs.
    # Switch to uuid5(NAMESPACE_DNS, f"{file_id}::{...}::{sha256(raw_text)[:16]}")
    # when a persistent cross-session reference to chunk_id is introduced.
    # Validate raw_text byte-stability before doing so.
    doc.metadata["chunk_id"] = str(uuid.uuid4())
```

> 🔵 **DEFERRED — Issue 3:** `chunk_id` remains `uuid4()` (random per ingest).
> **Rationale:** `overwrite=True` purges the entire Weaviate collection before
> rebuild, so old uuid4 objects are never left as orphans. Neo4j does not use
> `chunk_id` as a node key. No active data corruption exists. Switching to
> `uuid5` is required only when a persistent cross-session external index keyed
> on `chunk_id` is introduced.

**Step B** — group children by parent:

```python
# retrieval.py:1814-1821
_p2c: dict = _defaultdict(list)
for doc in child_docs:
    pid_ = doc.metadata.get("parent_id")
    if pid_:
        _p2c[pid_].append(doc.metadata["chunk_id"])
```

**Step C** — write all relationship + positional + navigation fields: ✨ NEW

```python
# retrieval.py:1823-1847
for doc in child_docs:
    pid_  = doc.metadata.get("parent_id", "")
    cid_  = doc.metadata["chunk_id"]
    h1_   = doc.metadata.get("Header 1", "")
    fid_  = doc.metadata.get("file_id", "")
    ch_id_ = str(uuid5(NAMESPACE_DNS, f"{fid_}::{h1_}")) if h1_ else ""
    siblings_all = _p2c.get(pid_, [cid_])
    pos_         = siblings_all.index(cid_) if cid_ in siblings_all else 0
    doc.metadata.update({
        "document_id":         fid_,
        "chapter_id":          ch_id_,          # deterministic uuid5 per H1
        "section_id":          pid_,
        "sibling_chunk_ids":   ",".join(s for s in siblings_all if s != cid_),
        "child_chunk_ids":     ",".join(siblings_all),
        "position_in_section": pos_,             # 0-indexed
        "content_type":        "body_text",
        "token_count":         max(1, len(doc.page_content) // 4),
        # Sanitised header labels — "Header 1" contains a space → invalid
        # Weaviate property name → silently dropped. Store as header_1/header_2.
        "header_1":            doc.metadata.get("Header 1", ""),
        "header_2":            doc.metadata.get("Header 2", ""),
    })
    # breadcrumb was written in Step 3 and flows through doc.metadata automatically
```

---

### Step 6 — Contextual Chunking (Layer 3) ✨ NEW

```python
# retrieval.py:1921
docs = _apply_contextual_chunking(docs, full_doc_text, ollama_url=ollama_url)
```

`apply_contextual_chunking()` in `contextual_chunking.py:301`.

**6a — OLLAMA_NUM_PARALLEL startup warning** ✨ NEW

```python
# contextual_chunking.py:334-343
_ollama_parallel = int(os.environ.get("OLLAMA_NUM_PARALLEL", "1"))
if _MAX_WORKERS > 1 and _ollama_parallel <= 1:
    log.warning("OLLAMA_NUM_PARALLEL=%d but contextual chunking uses %d workers. "
                "Requests queue serially. Set OLLAMA_NUM_PARALLEL=%d ...",
                _ollama_parallel, _MAX_WORKERS, _MAX_WORKERS)
```

**6b — Sequential preceding_text precomputation** ✨ NEW

```python
# contextual_chunking.py:346-349
# CONSTRAINT: must be precomputed sequentially before parallel workers start.
# Workers run out-of-order; computing preceding_text inside a worker risks
# reading stale/empty content from an unfinished sibling.
preceding_texts: list[str] = [""]
for i in range(1, total):
    prev_raw = docs[i-1].metadata.get("raw_text") or docs[i-1].page_content
    preceding_texts.append(prev_raw)
```

**6c — `_call_ollama` with retry + circuit breaker** ✨ NEW

```python
# contextual_chunking.py:76-126
_CONSECUTIVE_FAILURES: list[int] = [0]     # module-level, thread-safe via CPython GIL
_CIRCUIT_OPEN_THRESHOLD: int     = 5

def _call_ollama(prompt, url, model, timeout, retries=2) -> Optional[str]:
    if _CONSECUTIVE_FAILURES[0] >= _CIRCUIT_OPEN_THRESHOLD:
        log.debug("Ollama circuit open (%d failures) — skipping", ...)
        return None                          # ~1ms, prevents 25-min fallback storm

    for attempt in range(retries + 1):      # up to 3 total attempts
        try:
            resp = requests.post(f"{url}/api/generate", ...)
            resp.raise_for_status()
            _CONSECUTIVE_FAILURES[0] = 0    # reset on success
            return resp.json().get("response", "").strip() or None
        except Exception as exc:
            if attempt < retries:
                wait_s = 2 ** attempt       # 1s, then 2s
                time.sleep(wait_s)

    _CONSECUTIVE_FAILURES[0] += 1
    return None
```

**6d — `_enrich_chunk` worker + FALLBACK_COUNTER** ✨ NEW

```python
# contextual_chunking.py:224-296
FALLBACK_COUNTER: list[int] = [0]   # incremented on every fallback — visible metric

def _enrich_chunk(idx, doc, full_doc, url, model, position, total, preceding_text):
    topic = _call_ollama(_topic_prompt(...), url, model, _CHUNK_TIMEOUT)
    if topic:
        context_block = _build_context_block(doc, position, total, topic, preceding_text)
        # context_block format:
        # [CONTEXT]
        # Document: {title}
        # Chapter:  {H1}
        # Section:  {H2}
        # Position: paragraph {n} of {total}
        # Topic:    {LLM 1-2 sentence summary}
        # Preceding idea: {first sentence of previous chunk}
        #
        # [CONTENT]
        return (idx, Document(
            page_content=f"{context_block}\n{raw_text}",
            metadata={**doc.metadata,
                      "raw_text":        raw_text,
                      "context_summary": context_block,
                      "has_context":     True},
        ), elapsed, True)

    # Fallback path
    FALLBACK_COUNTER[0] += 1          # every fallback is counted
    return (idx, Document(
        page_content=raw_text,
        metadata={**doc.metadata,
                  "raw_text":        raw_text,
                  "context_summary": "",
                  "has_context":     False},
    ), elapsed, False)
```

Workers run in `ThreadPoolExecutor(max_workers=3)` via `as_completed()`.

---

### Step 7 — Topic & Key-Entities Extraction

After contextual chunking, extract the `topics` field from the `[CONTEXT]`
block stored in `context_summary`, and alias `keywords` → `key_entities`.

```python
# retrieval.py:1932-1944
import re as _re_mgi
for doc in docs:
    ctx = doc.metadata.get("context_summary", "")
    if ctx:
        _m = _re_mgi.search(r'^Topic:\s*(.+)$', ctx, _re_mgi.MULTILINE)
        if _m:
            doc.metadata["topics"] = _m.group(1).strip()
    kw = doc.metadata.get("keywords", "")
    if kw:
        doc.metadata["key_entities"] = kw
```

If `has_context=False` (Ollama fallback), `context_summary=""` and `topics`
is never set.

---

### Step 8 — Embedding

```python
# retrieval.py:1994-2010  (simplified)
embedding_function = get_embedding_function(
    engine="sentence_transformers",
    model="BAAI/bge-m3",
    ...
)
embeddings = await embedding_function(texts, RAG_EMBEDDING_CONTENT_PREFIX)
```

- Model: `BAAI/bge-m3`, loaded from local HuggingFace cache (`HF_HUB_OFFLINE=1`)
- Device: CUDA (RTX 5090), FP16 via `SENTENCE_TRANSFORMERS_MODEL_KWARGS`
- Batch size: 4 (`RAG_EMBEDDING_BATCH_SIZE`)
- **What is embedded:** `page_content` — includes the `[CONTEXT]` prefix —
  so the embedding captures document position and topic, not just local meaning
- Model is loaded once and cached in `get_embedding_function._st_cache`

---

### Step 9 — Weaviate Schema & Insert ✨ NEW (schema additions)

`VECTOR_DB_CLIENT.insert()` → `weaviate.py:242`

**Schema properties** (`_create_collection`, `weaviate.py:128`):

```
Core text fields:
  text              TEXT   — embedded page_content (with [CONTEXT] prefix)
  raw_text          TEXT   — original chunk text; used for BM25 and display
  context_summary   TEXT   — the full [CONTEXT] block
  has_context       BOOL   — True if contextual enrichment succeeded

Hierarchical navigation:
  parent_id         TEXT   skip_vectorization=True
  parent_text       TEXT   — full parent section returned to LLM

Document structure / navigation labels:  ✨ NEW
  header_1          TEXT   skip_vectorization=True — sanitised "Header 1"
  header_2          TEXT   skip_vectorization=True — sanitised "Header 2"
  breadcrumb        TEXT   skip_vectorization=True — "H1 > H2 > H3" label

Layer 4 MGI:
  chunk_id          TEXT   skip_vectorization=True — Weaviate object UUID
  document_id       TEXT   skip_vectorization=True — file_id
  chapter_id        TEXT   skip_vectorization=True — uuid5(file_id + H1)
  section_id        TEXT   skip_vectorization=True — alias of parent_id
  sibling_chunk_ids TEXT   skip_vectorization=True — comma-separated
  child_chunk_ids   TEXT   skip_vectorization=True — comma-separated
  position_in_section INT  skip_vectorization=True — 0-based index
  token_count       INT    skip_vectorization=True — chars // 4
  content_type      TEXT   skip_vectorization=True — "body_text"

Semantic metadata:
  topics            TEXT   — extracted from Topic: line in [CONTEXT]
  key_entities      TEXT   — alias of keywords
  keywords          TEXT   — regex/KeyBERT/Llama3 keywords for BM25
```

Insert call:
```python
# weaviate.py:264
batch.add_object(
    properties=properties,   # all fields above via process_metadata()
    uuid=item_uuid,          # = chunk_id (or uuid4 if chunk_id absent)
    vector=item["vector"],   # bge-m3 embedding
)
```

---

```
╔══════════════════════════════════════════════════════════════╗
║              PHASE 2 — QUERY                                 ║
║   get_sources_from_items()  ·  utils.py:1287                 ║
╚══════════════════════════════════════════════════════════════╝
```

### Step 10 — User Query Entry

User submits a query in the chat UI. `middleware.py` intercepts, detects
RAG collections are attached, and routes through `get_sources_from_items()`.

---

### Step 11 — Multi-Query Expansion (middleware)

If `ENABLE_RAG_WEB_SEARCH_MULTI_QUERY` is enabled, the LLM generates 3–4
diverse rephrasings of the original query before retrieval.

Additionally, `get_sources_from_items` runs a lightweight keyword distillation
pass — strips MCQ option labels, extracts 4+ character content words, appends
a compact keyword query variant for BM25 signal:

```python
# utils.py:1329-1341
for q in list(queries)[:2]:
    clean = re.sub(r'\([a-d]\)\s*[\w\s]{0,30}', ' ', q)
    words = [w.lower() for w in re.findall(r'[A-Za-z]{4,}', clean)
             if w.lower() not in STOP]
    if len(words) >= 2:
        keyword_q = ' '.join(words[:4])
        if keyword_q not in expanded:
            expanded.append(keyword_q)
queries = expanded[:6]
```

---

### Step 12 — Cosine Query Deduplication ✨ NEW (replaces Jaccard)

Semantically near-identical queries are dropped before retrieval to avoid
redundant embedding calls and inflated candidate pools.

```python
# utils.py:1352-1379
if len(queries) > 1:
    try:
        import numpy as _np
        _q_embeds = await embedding_function(queries, RAG_EMBEDDING_QUERY_PREFIX)
        _arr   = _np.array(_q_embeds, dtype=_np.float32)      # (N, D)
        _norms = _np.linalg.norm(_arr, axis=1, keepdims=True)
        _arr_n = _arr / (_norms + 1e-9)                       # unit vectors
        _sim_mx = _arr_n @ _arr_n.T                           # (N×N) BLAS call

        _keep = [True] * len(queries)
        for _qi in range(1, len(queries)):
            for _qj in range(_qi):
                if _keep[_qj] and float(_sim_mx[_qi, _qj]) > 0.95:
                    _keep[_qi] = False
                    break
        queries = [q for q, k in zip(queries, _keep) if k]
    except Exception:
        pass   # graceful fallback — numpy unavailable or embed failed
```

Threshold 0.95: only near-perfect paraphrases are dropped; diverse
multi-query expansions with different vocabulary are preserved.

---

### Step 13 — Per-Query Per-Collection Hybrid Retrieval

`query_doc_with_hybrid_search()` called once per (query, collection) pair.
`utils.py:329`

**13a — BM25 Weight Decision**

```python
# utils.py:375-389
words = query.split()
has_entity = any(w[0].isupper() for w in words if len(w) > 3)
if len(words) <= 4 or has_entity:
    hybrid_bm25_weight = 0.65   # entity/keyword queries → BM25 leads
else:
    hybrid_bm25_weight = 0.35   # conceptual queries → vector leads

is_hierarchical = any(m.get("parent_id") for m in collection_sample.metadatas[0])
if is_hierarchical:
    hybrid_bm25_weight = max(hybrid_bm25_weight, 0.65)  # always BM25-heavy
```

**13b — Retriever Construction**

```python
# utils.py:362-403
bm25_retriever = WeaviateBM25Retriever(
    collection_name=collection_name,
    top_k=min(k, 15),
    query_properties=["raw_text", "keywords"],   # never searches [CONTEXT] prefix
)
vector_search_retriever = VectorSearchRetriever(
    collection_name=collection_name,
    embedding_function=embedding_function,
    top_k=k,
)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_search_retriever],
    weights=[hybrid_bm25_weight, 1.0 - hybrid_bm25_weight],
)
```

`EnsembleRetriever` fuses results using Reciprocal Rank Fusion (RRF).

---

### Step 14 — Pre-Rerank SHA256 Deduplication ✨ NEW

```python
# utils.py:421-436
raw_candidates = await ensemble_retriever.ainvoke(query)

_seen_ids: set = set()
_deduped_candidates: list = []
for _dc in raw_candidates:
    _raw_dc = _dc.metadata.get("raw_text") or _dc.page_content
    _h_dc = hashlib.sha256(_raw_dc.encode()).hexdigest()
    if _h_dc not in _seen_ids:
        _seen_ids.add(_h_dc)
        _deduped_candidates.append(_dc)
```

Single pass, keyed on `raw_text`. Keeps **first occurrence** (highest RRF
score). BM25 and vector arms frequently return the same chunk; deduplicating
before the cross-encoder prevents wasted GPU inference on identical inputs.

> Previously this was done inside `ContextualCompressionRetriever` (wrong
> abstraction layer) and also again **after** reranking (two passes with
> conflicting winner selection). Both removed.

---

### Step 15 — Cross-Encoder Reranking with Sigmoid Normalisation ✨ NEW ⚠️ BREAKING

`RerankCompressor.acompress_documents()` — `utils.py:2035`

**15a — Cross-encoder scoring**

```python
# utils.py:2047-2050
def _rerank_on_gpu(query, docs):
    with torch.cuda.device(0):
        return self.reranking_function(query, docs)
scores = await asyncio.to_thread(_rerank_on_gpu, query, documents)
# model: bge-reranker-v2-m3, FP16, output logits ∈ ~[-10, +10]
```

**15b — Sigmoid normalisation** ✨ NEW ⚠️ BREAKING

```python
# utils.py:2083-2096
if reranking:
    scores_norm = [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]
    log.debug("Reranker raw logits  : min=%.3f  max=%.3f  mean=%.3f", ...)
    log.debug("Reranker sigmoid-norm: min=%.3f  max=%.3f  mean=%.3f", ...)
else:
    scores_norm = raw_scores   # cosine path already in [-1, 1]
```

> ⚠️ **BREAKING:** `RAG_RELEVANCE_THRESHOLD` is now interpreted on the `[0,1]`
> sigmoid scale. Default changed `0.0 → 0.3`.
> **Migration:** `new_threshold = 1 / (1 + exp(-old_threshold))`
> Example: old `2.0` → new `0.88`

**15c — Threshold filter**

```python
# utils.py:2106-2119
effective_floor = max(
    self.r_score if self.r_score else 0.0,
    0.3 if reranking else 0.05,
)
docs_with_scores = [(d, s) for d, s in docs_with_scores if s >= effective_floor]
log.debug("Reranker threshold %.2f (%s): kept %d/%d docs",
          effective_floor, "sigmoid-norm" if reranking else "cosine", ...)
```

**15d — Metadata bonuses**

```python
# utils.py:2126-2151
# Metadata bonuses — deliberately calibrated as [0,1]-scale boost signals
# for NCERT corpus quality indicators. These are NOT noise compensation;
# after sigmoid normalization each field contributes a meaningful ~3–5%
# increment. Max combined bonus = 0.05+0.05+0.03+0.04 = +0.17.
#
# Design invariant: the max +0.17 bonus cannot rescue a truly irrelevant
# chunk over the 0.3 threshold on its own. A chunk needs a base sigmoid
# score ≥ 0.13 (raw logit ≥ -1.95) before metadata can push it over 0.3.
# Math: 0.13 + 0.17 = 0.30 (just meets threshold). Below 0.13 → no rescue.
bonus = 0
if metadata.get("chapter"):    bonus += 0.05
if metadata.get("subject"):    bonus += 0.05
if metadata.get("class_level"):bonus += 0.03
if metadata.get("title"):      bonus += 0.04
doc_score = doc_score + bonus
metadata["score"] = doc_score   # final score stored for downstream use
```

---

### Step 16 — Parent Deduplication ✨ NEW

After reranking, multiple children from the same parent all expand to the
same `parent_text` body. Without this step the LLM would see the same section
text repeated once per matching child.

```python
# utils.py:440-479

# Pass 1: find best-scored child per parent_id
_parent_best: dict = {}
for _d in result:
    _pid = _d.metadata.get("parent_id")
    if not _pid:
        continue
    _sc = _d.metadata.get("score", 0.0)
    if _pid not in _parent_best or _sc > _parent_best[_pid].metadata.get("score", 0.0):
        _parent_best[_pid] = _d

# Pass 2: walk result in reranker rank order, emit one entry per parent
# Design: parent surfaces at position of its first-encountered child
# (retrieval salience), represented by the highest-scoring child's metadata.
_result_deduped: list = []
_seen_pids: set = set()
_parent_dedup_collapsed = 0
for _d in result:
    _pid = _d.metadata.get("parent_id")
    if _pid:
        if _pid not in _seen_pids:
            _seen_pids.add(_pid)
            _result_deduped.append(_parent_best[_pid])
        else:
            _parent_dedup_collapsed += 1
    else:
        _result_deduped.append(_d)   # no parent_id → pass through unchanged

if _parent_dedup_collapsed:
    log.debug("Parent dedup: collapsed %d sibling chunk(s) into %d unique parent(s)"
              " — kept highest reranker score per parent_id",
              _parent_dedup_collapsed, len(_seen_pids))
result = _result_deduped
```

Comparison key: `metadata["score"]` = sigmoid-normalised reranker score +
metadata bonuses. **Not** RRF distance. **Not** BM25 score.

---

### Step 17 — Body Building with Strategy B Headers ✨ NEW

For each doc in `result`, the body sent to the LLM is assembled as follows.

**17a — Content priority**

```python
# utils.py:497-501
body = (
    d.metadata.get("parent_text")   # hierarchical: full section (4× child size)
    or d.metadata.get("raw_text")   # contextual: original chunk without prefix
    or d.page_content               # fallback: embedded text (includes [CONTEXT])
)
```

**17b — Strip embedded [CONTEXT] block** ✨ NEW

```python
# utils.py:505
body = re.sub(r'\[CONTEXT\].*?\[CONTENT\]\n?', '', body, flags=re.DOTALL).strip()
```

Runs **unconditionally** on every body. Protects both enriched and
non-enriched paths. Prevents the LLM seeing the raw embedding prefix when
`parent_text` was stored after contextual chunking had already prepended the
`[CONTEXT]` block to `page_content`.

**17c — Strategy B header injection** ✨ NEW

Controlled by `ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS` (config.py:2802).

```python
# utils.py:506-527
if enable_enriched_texts:
    if d.metadata.get("has_context"):
        header = _build_context_header(d.metadata)
        if header:
            body = f"{header}\n{body}"
            log.debug("[ContextHeader] chunk pos=%s header_bytes=%d", ...)
        else:
            _header_skipped += 1
    else:
        _header_skipped += 1   # old chunk without contextual enrichment

if enable_enriched_texts and _header_skipped:
    log.debug("[ContextHeader] context_header_skipped_count=%d ...", _header_skipped)
```

**`_build_context_header(meta)`** — `utils.py:218`

Source label fallback order:

| Priority | Condition | Source label |
|----------|-----------|--------------|
| 1 | `header_1` AND `header_2` both present | `"H1 > H2"` |
| 2 | `breadcrumb` present | `"H1 > H2 > H3"` (already formatted) |
| 3 | `header_1` only | `"H1"` |
| 4 | `name`/`title`/`source` | filename |
| 5 | nothing | `return ""` — header skipped |

```python
# utils.py:236-270
h1         = (meta.get("Header 1") or meta.get("header_1") or "").strip()
h2         = (meta.get("Header 2") or meta.get("header_2") or "").strip()
breadcrumb = (meta.get("breadcrumb") or "").strip()
filename   = (meta.get("name") or meta.get("title") or meta.get("source") or "").strip()

if h1 and h2:       source_label = f"{h1} > {h2}"
elif breadcrumb:    source_label = breadcrumb
elif h1:            source_label = h1
elif filename:      source_label = filename
else:               return ""

pos         = meta.get("position_in_section") or 0
display_pos = int(pos) + 1

child_ids_str = (meta.get("child_chunk_ids") or "").strip()
if child_ids_str:
    total   = len([c for c in child_ids_str.split(",") if c.strip()])
    pos_str = f"paragraph {display_pos} of {total}"
else:
    pos_str = f"paragraph {display_pos}"

topics     = (meta.get("topics") or "").strip()
topic_line = f"[Topic: {topics}]" if topics else ""
# returns:
# [Source: Chapter 6 > 6.1 Nutrition, paragraph 3 of 8]
# [Topic: Photosynthesis converts light energy into chemical energy stored as glucose.]
```

---

### Step 18 — Sources Merge & Global Deduplication

Results from all (query × collection) pairs are merged into a single
`sources` list in `get_sources_from_items`.

**18a — SHA256 exact dedup** ✨ NEW (max-score-wins)

```python
# utils.py:1820-1843
# Keep the copy with the highest reranker score (metadata["score"]).
# Last-write-wins was wrong: multiple sources may surface the same parent_text
# with different scores; the highest-scoring copy should be the survivor.
unique_docs = {}
for s in sources:
    for i, doc in enumerate(s["document"]):
        key    = hashlib.sha256(doc.encode()).hexdigest()
        _meta_i   = s["metadata"][i]
        _incoming = float(_meta_i.get("score", 0.0)) if isinstance(_meta_i, dict) else 0.0
        if key not in unique_docs:
            unique_docs[key] = (doc, _meta_i)
        else:
            _existing = float(unique_docs[key][1].get("score", 0.0)) \
                        if isinstance(unique_docs[key][1], dict) else 0.0
            if _incoming > _existing:
                unique_docs[key] = (doc, _meta_i)
```

> Previously: `unique_docs[key] = (doc, meta)` — last-write-wins, arbitrary
> winner when same parent_text appeared from multiple queries.

**18b — Jaccard semantic dedup**

Word-set overlap dedup (threshold 0.75, words > 3 chars) removes
near-duplicate passages that passed SHA256 dedup due to minor text
differences.

**18c — KG and UPSC signal blocks**

Knowledge Graph and UPSC signal sources are partitioned out before
dedup/rerank and re-injected at the top of the final context unchanged.

---

### Step 19 — Post-Merge Semantic Rerank

After dedup, a second rerank pass over the merged candidate pool from all
collections and queries is run using the same `RerankCompressor`, giving a
globally-ranked final context window.

---

```
╔══════════════════════════════════════════════════════════════╗
║              PHASE 3 — ANSWER                                ║
║   apply_source_context_to_messages()  ·  middleware.py:803   ║
╚══════════════════════════════════════════════════════════════╝
```

### Step 20 — RAG Prompt Construction ✨ NEW

Context string is assembled from all surviving document bodies:

```python
# middleware.py:819-830
for source in sources:
    for doc, meta in zip(source.get("document",[]), source.get("metadata",[])):
        src_id   = meta.get("source") or source.get("source",{}).get("id") or "N/A"
        src_name = source.get("source",{}).get("name")
        context_string += (
            f'<source id="{citation_idx[src_id]}"'
            + (f' name="{src_name}"' if src_name else "")
            + f">{doc}</source>\n"
        )
```

Each `doc` here is the fully-assembled body from Step 17:
```
[Source: Chapter 6 > 6.1 Nutrition, paragraph 3 of 8]
[Topic: Photosynthesis converts light energy...]
<clean parent_text — no [CONTEXT] prefix>
```

The RAG template is applied once and reused:

```python
# middleware.py:836-837
_rag_prompt = rag_template(
    request.app.state.config.RAG_TEMPLATE, context_string, user_message
)
```

---

### Step 21 — Debug Prompt Logging ✨ NEW

```python
# middleware.py:843-853
if log.isEnabledFor(logging.DEBUG):
    log.debug(
        "[RAG_PROMPT_DEBUG] Query: %s\n"
        "[RAG_PROMPT_DEBUG] Final prompt sent to LLM:\n%s\n"
        "[RAG_PROMPT_DEBUG] ---END--- (prompt_chars=%d, context_chars=%d)",
        user_message[:300],
        _rag_prompt[:6000],
        len(_rag_prompt),
        len(context_string),
    )
```

Gated at `DEBUG` level — silent in production. Enable with `LOG_LEVEL=DEBUG`
in `start_backend.ps1` or `.env`.

**Expected log output after full verification:**
```
[ContextHeader] chunk pos=0 header_bytes=91
[ContextHeader] chunk pos=1 header_bytes=88
Reranker raw logits  : min=-3.241  max=1.872  mean=-0.814
Reranker sigmoid-norm: min=0.038   max=0.867  mean=0.307
Parent dedup: collapsed 2 sibling chunk(s) into 1 unique parent(s)
[RAG_PROMPT_DEBUG] Query: what is photosynthesis
[RAG_PROMPT_DEBUG] Final prompt sent to LLM:
<source id="1" name="CLASS_12_Bio">
[Source: Chapter 6 — Life Processes > 6.1 Nutrition, paragraph 3 of 8]
[Topic: Photosynthesis converts light energy into chemical energy stored as glucose.]
Photosynthesis is the process by which ...
</source>
[RAG_PROMPT_DEBUG] ---END--- (prompt_chars=4821, context_chars=3204)
```

---

### Step 22 — LLM Inference

Final prompt is sent to the configured model:

```python
# middleware.py:855-858
if RAG_SYSTEM_CONTEXT:
    return add_or_update_system_message(_rag_prompt, messages, append=True)
else:
    return add_or_update_user_message(_rag_prompt, messages, append=False)
```

The LLM reads the structured context with `[Source:]` / `[Topic:]` headers,
generates an answer, and cites section names from the `[Source:]` labels.

---

## Re-Ingestion Checklist

All changes to the Weaviate schema (`header_1`, `header_2`, `breadcrumb`)
only apply to **newly created collections**. Existing collections lack these
properties.

| Action | Why required |
|--------|-------------|
| Delete all existing Weaviate collections for affected documents | Schema additions only apply on `_create_collection`; old collections never store `header_1`/`header_2`/`breadcrumb` |
| Re-ingest with `overwrite=True` | Forces `delete_collection()` + fresh insert with new schema |
| Ollama must be running during re-ingest | `topics` only populated when contextual chunking succeeds; empty `context_summary` = no `[Topic:]` line |
| Run `start_backend.ps1` (not base conda) | Uses `py -3.11` with CUDA torch; base conda Python 3.13 has CPU-only torch |
| Set `ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS=true` | Already in `start_backend.ps1`; gates Strategy B header injection |

---

## Configuration Reference

| Variable | Default | Notes |
|----------|---------|-------|
| `RAG_RELEVANCE_THRESHOLD` | `0.3` | ⚠️ BREAKING: now on sigmoid `[0,1]` scale. Migrate: `new = 1/(1+exp(-old))` |
| `ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS` | `False` | Set `true` to activate Strategy B headers |
| `ENABLE_RAG_HYBRID_SEARCH` | `False` | Must be `true` for hybrid BM25+vector path |
| `CHUNKING_STRATEGY` | `"hierarchical"` | Primary strategy; `"paragraph"` is secondary |
| `OLLAMA_NUM_PARALLEL` | `1` | Set ≥ `_MAX_WORKERS` (3) for true parallel contextual chunking |
| `LOG_LEVEL` | `INFO` | Set `DEBUG` to see `[ContextHeader]`, `[RAG_PROMPT_DEBUG]`, sigmoid logs |

---

## Tech Debt Register

| Item | Location | Note |
|------|----------|------|
| `chunk_id` is uuid4 (random per ingest) | `retrieval.py:1780, 1800` | Switch to `uuid5(NAMESPACE_DNS, f"{file_id}::{...}::{sha256(raw_text)[:16]}")` when external index keyed on `chunk_id` is introduced. Validate raw_text byte-stability first. |
| Keyword extraction runs before contextual chunking | `retrieval.py:1860-1895` | Should run after Step 6 so keywords can be extracted from enriched text including the Topic. No functional breakage today. |
| `get_enriched_texts()` is defined but has zero call sites | `utils.py:273` | Legacy BM25 enrichment intent, never deployed. Safe to remove in a future cleanup PR. |
