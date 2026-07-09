# HIERARCHICAL + CONTEXTUAL

## RAG Corpus Chunking Strategy Manual

─────────────────────────────────────────────
Version 1.0 · FutureVerse Technologies · Previous (Pre-Foundation) Specification · "SHC-CE" Pipeline
Governing Code: `retrieval/pipeline_agents.py` · `retrieval/contextual_chunking.py` · `retrieval/utils.py` · `docs/rag_pipeline_architecture.md`
Audience: Corpus Engineer · Backend Engineer · Technical Architect · Chief Architect

> **Structure note.** This manual mirrors the *Infinity Foundation — NCERT Corpus
> Chunking Strategy Manual* **section-for-section** (same Parts, Chapters,
> sub-section numbers, and Appendices A–E). Each section addresses the **same
> topic** as the Foundation manual, but the content describes the **previous,
> hierarchical** strategy (`CHUNKING_STRATEGY = "hierarchical"`), not the
> paragraph-level Foundation strategy.

## Purpose and Scope

This manual is the single source of truth for the **Hierarchical + Contextual**
chunking strategy — the general-purpose RAG chunker that predates the Foundation
paragraph strategy and still ships as `CHUNKING_STRATEGY = "hierarchical"`.

Chunking strategy here **is** a generic RAG engineering problem. The retrieval
purpose is the ordinary one: given a user query, return the most semantically
**relevant** passages. Every chunking decision traces to one of three downstream
requirements: the **parent-expansion requirement** (a matched small child must
expand to its larger parent for context), the **context-enrichment requirement**
(each chunk's embedding should capture document position and topic), and the
**hybrid-relevance requirement** (chunks must be findable by both keyword/BM25
and vector similarity).

This manual is advisory engineering specification, not a validation gate. Quality
is enforced by **automated checkers** in the pipeline, not by a human Academic
Lead, and chunks are not gated behind an `academic_lead_validated` flag.

---

# PART I — Foundations

## Chapter 1 — Why This Chunking Strategy Works the Way It Does

### 1.1 The Retrieval Purpose Is Relevance

In this strategy the retrieval goal is **relevance**: given a query, return the
passages whose meaning is closest to it. BM25, embedding similarity, and their
hybrid fusion are all optimised for exactly this. There is no notion of "the
right chunk for *this student right now*" — every query is answered with the
globally most-similar context. (This is the precise opposite of Foundation's
cognitive-state retrieval.)

### 1.2 No Verbatim Guarantee — Context-Enriched Embeddings

This strategy makes **no verbatim guarantee.** Before embedding, each chunk is
prepended with an LLM-generated `[CONTEXT]` block (document/chapter/section/
position/topic). The embedded text is therefore *not* the raw NCERT text — it is
enriched so the vector captures position and topic, not just local meaning. At
answer time the `[CONTEXT]` prefix is stripped and the parent section text is
sent to the LLM, but there is no self-consistency check that every served claim
is verbatim from source.

### 1.3 No Sentence-Level Trim — Full Parent Expansion

There is **no sentence-level trimming.** When a child chunk matches, the whole
**parent section** (`parent_text`, ≈4× a child) is returned to the LLM. The unit
of serving is the parent section, not a trimmed set of sentences. Consequently
there is no `proposition_sentence_map`; the only sub-chunk structure is the
`position_in_section` index and the parent/child links.

### 1.4 What Happens If Chunking Is Wrong

| Chunking Error | Immediate Effect | Downstream Consequence |
|---|---|---|
| Child boundary mid-sentence | Child begins/ends mid-clause | Embedding noise; parent expansion usually recovers context, but BM25 match quality drops |
| `[CONTEXT]` prefix not stripped | Prefix bleeds into the LLM body | Model reads pseudo-metadata as content; citations get muddy |
| `parent_id` / `parent_text` lost | Child cannot expand | LLM receives a 250-char fragment with no surrounding context |
| Header text not stripped | Headings inside `text` | BM25/embedding noise; "Chapter N" appears as content |
| `file_id` dropped during split | Chunk cannot be filtered to its file | `ChunkingChecker` fails the stage |

### 1.5 The Corpus at Scale

The strategy is corpus-agnostic — any uploaded document (PDF/MD/TXT/DOCX) is
chunked the same way. At `CHUNK_SIZE = 1000` / child ≈ 250, a typical NCERT
chapter (~28k chars) yields a few hundred child chunks. Weaviate (HNSW + BM25)
scales to millions of child vectors; there is no per-subject Q-matrix gating —
documents are ingested on demand.

## Chapter 2 — The Primary Strategy: Hierarchical Parent–Child Chunking

### 2.1 Why Fixed-Size Parent–Child

The primary unit is a **fixed-size character chunk**, in two levels. Small
**child** chunks give precise vector/BM25 matching; the larger **parent** section
they belong to gives the LLM enough context to answer. This "small-to-search,
large-to-read" design is the core idea. Boundaries are decided by character
count, *not* by sentence or proposition — the explicit difference from Foundation.

### 2.2 The Chunk Size Specification

| Level | Splitter | Size (defaults) | Overlap |
|---|---|---|---|
| **Parent** | `RecursiveCharacterTextSplitter` | `CHUNK_SIZE` = **1000** chars | `CHUNK_OVERLAP` = **100** |
| **Child** | `RecursiveCharacterTextSplitter` | `max(100, CHUNK_SIZE // 4)` = **250** chars | `max(20, OVERLAP // 2)` = **50** |

The child is the embedded/searched unit; the parent is the served unit.

### 2.3 The (Absent) Proposition-Split Constraint

Unlike Foundation, this strategy has **no proposition-split constraint.** A child
boundary may fall mid-sentence or mid-proposition. The mitigation is structural,
not preventive: a matched child always expands to its full `parent_text`, so the
LLM still sees the complete surrounding section even when the child fragment is
broken.

### 2.4 Structural Coherence: Markdown Header Split

`ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER` (default **true**) runs a
`MarkdownHeaderTextSplitter` over six heading levels (`#`…`######`,
`strip_headers=False`) *before* the parent/child split, so chunks stay within
section boundaries and carry `Header 1/2/3` metadata. An optional
`merge_docs_to_target_size()` pass merges sections under `CHUNK_MIN_SIZE_TARGET`
(default 0 = off).

### 2.5 When Chunks Are Split (Recursive Character Splitting)

`RecursiveCharacterTextSplitter` splits greedily on a separator hierarchy
(paragraph → line → sentence → word) until each piece is ≤ the size limit, with
overlap to preserve cross-boundary context. There is no topic-shift or
definitional-claim logic — purely size-driven.

### 2.6 When Small Sections Are Merged

Sections smaller than `CHUNK_MIN_SIZE_TARGET` may be merged by
`merge_docs_to_target_size()` to avoid tiny fragment chunks. With the default of
`0` this is off, so very short sections can become standalone small chunks
(parent expansion compensates at retrieval time).

---

# PART II — Content-Type Handling and Extraction

## Chapter 3 — Content Types (Uniform Treatment)

### 3.1 Why Content Type Is NOT Differentiated

This strategy treats **all extracted text uniformly.** It does not classify body
text vs sidebars vs exercises vs captions; every element flows through the same
parent/child splitter and is tagged `content_type = "body_text"`. (Content-type
awareness is a Foundation-only capability.) The trade-off: enrichment boxes,
exercises, and table fragments can all enter the searchable pool.

### 3.2 Body Text

Narrative prose is the common case and is chunked exactly per Chapter 2.

### 3.3 Sidebar Boxes

Not distinguished — sidebar text is split and indexed like body text. "Did you
know?" / "Let's recall" boxes can therefore surface in retrieval.

### 3.4 Tables

Not specially handled. Tables are split by character count like prose; rows are
**not** chunked one-per-row with a repeated header. Table fragments may appear
without their column headers.

### 3.5 Diagrams and Flowcharts

Only whatever text the extractor produced (captions/labels) is chunked, as
ordinary text. No diagram placeholder records.

### 3.6 Chapter-End Exercises

Not excluded. Question text is chunked and indexed like any other text and can be
retrieved (there is no "exercise pool" routing).

### 3.7 Margin Notes and Captions

Not separated into a context-note field; if the extractor emits them inline they
are chunked as ordinary text.

## Chapter 4 — PDF Extraction Challenges and Solutions

### 4.1 Why PDFs Are Structurally Complex

The same six extraction challenges as the Foundation manual apply (multi-column
layout, header/footer contamination, hyphenation/line-breaks, bullet lists,
table cell ordering, footnotes). This strategy relies on the configured
`CONTENT_EXTRACTION_ENGINE` (Marker / Docling / PyPDF / Tika) to produce text and
does **not** add the manual's dedicated cleanup steps — extraction artefacts that
slip through are carried into chunks (mitigated only by overlap + parent context).

### 4.2 The Extraction Quality Checklist

There is no enforced pre-chunking quality checklist. The pipeline's
`ChunkingChecker` performs lightweight post-hoc checks instead: at least one chunk
produced, empty-chunk rate < 10%, and `file_id` preserved (see Chapter 8).

---

# PART III — Enrichment and Metadata

## Chapter 5 — The Post-Chunk Enrichment Pipeline

### 5.1 Overview

After chunking, every chunk passes through enrichment stages before indexing —
the rough analogue of Foundation's two-pass tagging, but producing **search
fields**, not Q-matrix attributes. Order (in `run_ingestion_pipeline`):
`KeywordAgent` → `ContextualEnrichmentAgent` → dedup → `EmbeddingAgent` → indexing.
Each stage is **agent-executed and checker-validated** (no human gate).

### 5.2 Pass 1 — Keyword Extraction (BM25)

`KeywordAgent` extracts entity-anchored keywords into `metadata["keywords"]` for
Weaviate's BM25 index. Method = `KEYWORD_EXTRACTION_METHOD`: `regex` (default,
no model), `keybert` (all-MiniLM), or `llama3` (Ollama, ~1s/chunk). Structural
fields (breadcrumb/chapter/subject) are appended for precise BM25 anchoring.

### 5.3 Pass 2 — Contextual Enrichment (`[CONTEXT]`)

`ContextualEnrichmentAgent` (`contextual_chunking.py`, `ENABLE_CONTEXTUAL_CHUNKING`
default **true**, model **qwen2.5:7b**) prepends a structured block per chunk:

```
[CONTEXT]
Document: {title}   Chapter: {H1}   Section: {H2}
Position: paragraph {n} of {total}
Topic: {LLM 1–2 sentence summary}
Preceding idea: {first sentence of previous chunk}
[CONTENT]
{chunk text}
```

The enriched text is what gets embedded. Resilience: 2-retry exponential backoff
+ circuit breaker (`ThreadPoolExecutor`, `max_workers=3`); on failure the chunk
falls back to raw text with `has_context=False` (`FALLBACK_COUNTER`).

### 5.4 Mapping to Searchable Fields

Where Foundation maps propositions → `attribute_tags`, this strategy maps
enrichment → **searchable Weaviate fields**: `keywords` (BM25), `topics`
(from the `Topic:` line), `key_entities` (alias of keywords), `context_summary`
(the full `[CONTEXT]` block), `has_context`. There are no attribute IDs; the
"index" a chunk is retrievable by is its embedding + BM25 keywords.

### 5.5 Cognitive Frame Assignment (Not Used)

This strategy assigns **no cognitive_frame.** There is no foundational /
integrative / analytical distinction and no depth-aware serving. The only
positional metadata is `position_in_section` and the parent/child links
(Section 5.6). Depth control, if any, is left to the LLM and the relevance ranker.

### 5.6 The MGI Navigation Map (parent/child links)

In place of a `proposition_sentence_map`, each chunk carries **multi-granularity
index (MGI)** metadata so siblings and parents resolve consistently:
`parent_id`, `parent_text`, `chapter_id` (`uuid5(file_id + H1)`), `section_id`
(= `parent_id`), `sibling_chunk_ids`, `child_chunk_ids`, `position_in_section`,
`breadcrumb`, `header_1/header_2`, `token_count`, `content_type = "body_text"`.

---

## Chapter 6 — Multi-Topic Chunks and Boundary Cases

### 6.1 The Multi-Topic Chunk Problem

A fixed-size chunk can bundle two unrelated topics (a 250-char window doesn't
respect topic boundaries). This strategy does **not** detect or split on topic
shifts. The reliance is on (a) overlap, (b) parent expansion giving full context,
and (c) the reranker filtering off-topic candidates. There is no per-topic
sentence map, so an off-topic sentence inside a matched chunk is still shown.

### 6.2 Context Leakage (`[CONTEXT]` Prefix Bleeding into Body)

The insidious failure here is the **`[CONTEXT]` block leaking into the served
body.** The body builder strips `[CONTEXT]…[CONTENT]` unconditionally; if that
strip is skipped or a stored `parent_text` already contains the prefix, the LLM
reads pseudo-metadata as fact. (Analogue of Foundation's "integration attribute
leakage.")

### 6.3 Cross-Section Chunks

A chunk may span two markdown sections if the header split didn't fire (un-marked
headings). It then carries the breadcrumb of only one section. Retrieval still
works by relevance, but the `breadcrumb`/`header_1` label may be inaccurate.

### 6.4 Multi-Document Coverage

The same topic across multiple uploaded documents produces independent chunk sets
in separate Weaviate collections. Retrieval can query multiple collections and
fuses results via RRF; there is no multi-class coverage index.

---

# PART IV — Retrieval Architecture and Serving

## Chapter 7 — Relevance-Based Hybrid Retrieval

### 7.1 The Retrieval Pipeline

At query time (`utils.py`):
1. **Multi-query expansion** (optional): LLM rephrases the query 3–4 ways + a
   keyword-distilled BM25 variant.
2. **Cosine query dedup**: near-identical variants (cos > 0.95) dropped.
3. **Per-(query × collection) hybrid search**: `EnsembleRetriever` fuses
   `WeaviateBM25Retriever` + `VectorSearchRetriever` via **Reciprocal Rank
   Fusion**. BM25 weight = **0.65** (short/entity), **0.35** (conceptual), forced
   ≥ **0.65** for hierarchical collections.
4. **SHA256 pre-rerank dedup** on `raw_text`.

Unlike Foundation, there is **no attribute filter and no cognitive-frame filter** —
ranking *is* the selection criterion (relevance), not a stage after filtering.

### 7.2 The Conflict Resolution Rule (Parent Dedup)

When several sibling children of the same parent rank, **parent dedup** collapses
them to one entry per `parent_id`, keeping the **highest reranker score**. This
parallels Foundation's "fewest words" conflict rule, but the goal is avoiding
duplicate parent sections in the context, not minimising trim surface.

### 7.3 Body Building at Serving Time (No Trimming)

For each surviving doc the body = `parent_text` → else `raw_text` → else
`page_content`, with any `[CONTEXT]…[CONTENT]` block **stripped**. Optional
Strategy-B `[Source:]`/`[Topic:]` headers
(`ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS`). The **full parent section** is sent
to the LLM — there is **no sentence-level trimming.** Cross-encoder rerank
(`BAAI/bge-reranker-v2-m3`, FP16) → sigmoid-normalised scores → floor at
`RAG_RELEVANCE_THRESHOLD` (default **0.3**) + small metadata bonuses; Top K = 3.

### 7.4 Retrieval Latency

No hard SLA is enforced. Latency = Weaviate hybrid query (HNSW + BM25) +
optional multi-query embedding + cross-encoder rerank on the candidate pool.
Reranking is the dominant cost; it is bounded by the candidate count, not by
corpus size.

---

# PART V — Validation, Quality Assurance, and Maintenance

## Chapter 8 — Automated Checker Validation

### 8.1 What the Checkers Validate

There is **no Academic Lead gate.** Each agent is paired with a **checker** that
validates its output before the pipeline advances; a checker returning `ok=False`
aborts ingestion. Checkers: `ChunkingChecker`, `KeywordChecker`,
`ContextualEnrichmentChecker`, `ChunkDeduplicationChecker`, `EmbeddingChecker`,
`SemanticDeduplicationChecker`, `IndexingChecker`.

### 8.2 Validation Procedure Per Stage

- **ChunkingChecker:** ≥1 chunk produced (hard fail if zero); empty-chunk rate
  < 10%; `file_id` not dropped.
- **KeywordChecker:** chunk count unchanged; `page_content` not mutated.
- **EnrichmentChecker:** no `raw_text` lost; counts enriched vs fallback.
- **EmbeddingChecker:** every chunk has a vector; dimension matches (1024).
- **IndexingChecker:** post-insert Weaviate count equals chunk count.

### 8.3 Throughput

Throughput is machine-bound, not human-bound: chunking + keyword (regex) are
sub-second; contextual enrichment is ~1 Ollama call per chunk (the slow stage);
embedding ≈ 10–17 ms/chunk on GPU; Weaviate insert is batched. A 47-chunk file
ingests in ~10 s end to end.

### 8.4 Pipeline Quality Metrics

| Metric | Target | Action if below |
|---|---|---|
| Chunks produced | ≥ 1 | hard fail (`ChunkingChecker`) |
| Empty-chunk rate | < 10% | warn / fail (splitter issue) |
| `file_id` preserved | 100% | fail (metadata loss) |
| Embedding coverage | 100%, dim = 1024 | fail (`EmbeddingChecker`) |
| Enrichment fallback rate | low (`FALLBACK_COUNTER`) | check Ollama / `OLLAMA_NUM_PARALLEL` |
| Indexed == produced | 100% | fail (`IndexingChecker`) |

---

## Chapter 9 — Re-Ingestion, Overwrite, and Maintenance

### 9.1 The Overwrite Policy (no Never-Delete)

This strategy uses **`overwrite=True`**, the opposite of Foundation's Never-Delete.
Re-ingesting a file `delete_collection()`s and rebuilds it; old chunks are purged,
not deprecated. There is no `deprecated_at` / `successor_chunk_id` audit trail and
`chunk_id` is `uuid4` (random per ingest), which is safe precisely because the old
collection is purged each rebuild.

### 9.2 Re-Ingestion on Schema/Model Change

Schema additions (`header_1`, `breadcrumb`, etc.) or an embedding-model change
apply **only to newly created collections**; existing collections must be deleted
and re-ingested with `overwrite=True` to pick them up. (This is why the Re-Ingestion
checklist in `rag_pipeline_architecture.md` exists.)

### 9.3 Adding New Documents

Uploading a file via the Knowledge page runs the full pipeline and creates/updates
its Weaviate collection. No vocabulary prerequisite — documents are added on
demand. (Note: the add-to-knowledge-base step pulls a file's chunks with an
explicit `limit=10000` to avoid the vector-DB default-limit truncation.)

---

# PART VI — Failure Modes and Appendices

## Chapter 10 — Chunking Failure Modes

### 10.1 Overview

Like Foundation, these are mostly **silent failures** — wrong retrieval behaviour
at runtime, not errors at ingest. Ten modes for this strategy:

- **Failure 1 — Mid-sentence / mid-proposition split.** Fixed-size cut leaves an
  orphan pronoun at a child boundary. Mitigated (not prevented) by parent
  expansion + overlap.
- **Failure 2 — `[CONTEXT]` prefix served to the LLM.** Strip step skipped → model
  reads pseudo-metadata as content.
- **Failure 3 — Stale/duplicate `parent_text`.** Same parent surfaces from multiple
  queries with conflicting scores (handled by max-score parent dedup).
- **Failure 4 — Broken parent link.** `parent_id`/`parent_text` lost → child served
  as a 250-char fragment with no context.
- **Failure 5 — Keyword over-extraction.** Noisy `keywords` field → BM25 returns
  off-topic chunks.
- **Failure 6 — Keyword under-extraction.** Sparse `keywords` → BM25 coverage gap;
  entity queries miss relevant chunks.
- **Failure 7 — Header contamination in `text`.** Running headers / headings not
  stripped → BM25 + embedding noise.
- **Failure 8 — Bullet list split into tiny chunks.** Each bullet becomes a
  decontextualised mini-chunk.
- **Failure 9 — Table fragment without header row.** Cells split by character count
  lose their column headers.
- **Failure 10 — Collection truncation / dim mismatch.** Default query limit caps
  a file's chunks (the fixed `limit=10000`), or an embedding-model change leaves a
  collection with mixed vector dimensions.

---

## Appendix A — Weaviate Chunk Schema Reference

Fields written by `IndexingAgent` (Weaviate, not Postgres). M = always set.

| Field | Type | Req | Description |
|---|---|---|---|
| `chunk_id` | UUID | M | Weaviate object UUID; uuid4 per ingest |
| `text` | text | M | embedded content (**includes** `[CONTEXT]` prefix) |
| `raw_text` | text | M | original chunk text (BM25 + display) |
| `context_summary` | text | O | full `[CONTEXT]` block |
| `has_context` | bool | M | true if enrichment succeeded |
| `parent_id` | text | M | links sibling children of one parent |
| `parent_text` | text | M | full parent section (served to LLM) |
| `header_1` / `header_2` | text | O | sanitised heading labels |
| `breadcrumb` | text | O | `"H1 > H2 > H3"` label |
| `document_id` | text | M | source `file_id` |
| `chapter_id` | text | O | `uuid5(file_id + H1)` |
| `section_id` | text | M | alias of `parent_id` |
| `sibling_chunk_ids` / `child_chunk_ids` | text | O | comma-separated lists |
| `position_in_section` | int | M | 0-based child index |
| `token_count` | int | M | `len(text) // 4` |
| `content_type` | text | M | always `"body_text"` |
| `keywords` / `key_entities` | text | O | BM25 keyword field |
| `topics` | text | O | from the `Topic:` line |
| `vector` | vector(1024) | M | bge embedding of `text` |

## Appendix B — Automated Checker Checklist

For each stage, the checker confirms before the pipeline advances:

**B.1 — Chunking** • ≥1 chunk produced • empty-chunk rate < 10% • `file_id` preserved.
**B.2 — Keyword** • chunk count unchanged • `page_content` unmutated • no empty `keywords` strings.
**B.3 — Enrichment** • `raw_text` present on every chunk • fallback count recorded.
**B.4 — Embedding** • one vector per chunk • dimension = 1024.
**B.5 — Indexing** • post-insert Weaviate object count == chunk count.

## Appendix C — Strategy Dispatch Decision Tree

`ChunkingAgent` selects the splitter from `CHUNKING_STRATEGY` / `TEXT_SPLITTER`:

| `CHUNKING_STRATEGY` | Splitter | Notes |
|---|---|---|
| `hierarchical` | markdown split → parent (1000) → child (250) | this manual |
| `character` / (TEXT_SPLITTER `""`) | `RecursiveCharacterTextSplitter` | single level |
| `token` | `TokenTextSplitter` (tiktoken) | uses `TIKTOKEN_ENCODING_NAME` |
| `fixed` | single-level via `TEXT_SPLITTER` | no parent/child |
| `foundation` | (separate manual) | paragraph-level, bypasses this pipeline |

All non-foundation strategies still run keyword + contextual enrichment + embedding + Weaviate indexing.

## Appendix D — MGI Metadata JSON Schema

```jsonc
{
  "chunk_id": "<uuid4>",
  "parent_id": "<uuid4>",          // shared by sibling children
  "parent_text": "<full section text>",
  "document_id": "<file_id>",
  "chapter_id": "<uuid5(file_id + H1)>",
  "section_id": "<= parent_id>",
  "sibling_chunk_ids": "<csv of chunk_ids>",
  "child_chunk_ids": "<csv of chunk_ids>",
  "position_in_section": 0,         // 0-based
  "breadcrumb": "H1 > H2 > H3",
  "header_1": "...", "header_2": "...",
  "content_type": "body_text",
  "token_count": 62,
  "has_context": true,
  "context_summary": "[CONTEXT] ... [CONTENT]",
  "keywords": "Finance Commission, Article 280, ...",
  "topics": "..."
}
```

## Appendix E — Glossary

| Term | Definition |
|---|---|
| **Relevance retrieval** | Return the most semantically similar chunks to the query (BM25 + vector + rerank). The retrieval model of this strategy. |
| **Parent / child** | Child (~250 char) = embedded/searched unit; parent (~1000 char section) = served unit. |
| **Parent expansion** | Returning a matched child's full `parent_text` to the LLM for context. |
| **MGI metadata** | Multi-granularity index links (parent/child/sibling/position) on each chunk. |
| **Contextual enrichment** | LLM-generated `[CONTEXT]` block prepended before embedding. |
| **`has_context`** | Whether enrichment succeeded for a chunk (false ⇒ raw-text fallback). |
| **Hybrid search / RRF** | Reciprocal-Rank-Fusion of BM25 and vector retrievers. |
| **Cross-encoder rerank** | `bge-reranker-v2-m3` re-scores candidates; sigmoid-normalised; floor `RAG_RELEVANCE_THRESHOLD`. |
| **Parent dedup** | Collapse sibling children to one entry per `parent_id`, max reranker score. |
| **Overwrite policy** | Re-ingest purges + rebuilds a collection (no Never-Delete, no deprecation). |
| **Checker** | Automated per-stage validator (replaces the Academic Lead gate). |
