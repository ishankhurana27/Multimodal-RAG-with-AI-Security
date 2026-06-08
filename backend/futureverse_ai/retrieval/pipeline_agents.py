"""
Ingestion pipeline agents — Steps 3–7 of the SHC-CE pipeline.

Each agent owns exactly one stage:
  ChunkingAgent             — Step 3: SHC-CE hierarchical / character / token chunking
  KeywordAgent              — Step 5: BM25 keyword extraction (runs before enrichment)
  ContextualEnrichmentAgent — Step 4: LLM [CONTEXT]/[CONTENT] prefix via Ollama
  EmbeddingAgent            — Step 6: bge-m3 GPU vector generation
  IndexingAgent             — Step 7: Weaviate HNSW + BM25 dual-index insert

Each agent is paired with a Checker that validates its output before the
pipeline advances to the next stage. A checker returning ok=False aborts the
pipeline with ValueError. Warnings are logged but never abort.

Orchestrator: run_ingestion_pipeline() — drives all agents + checkers in order.

Concept contract: every agent contains the EXACT same logic as the inline code
it replaces. No algorithm change, no config change, no quality change.
"""

import logging
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Agent result envelope
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Uniform return type for every agent and checker."""
    agent:     str
    ok:        bool
    data:      Any
    elapsed_s: float
    warnings:  list = field(default_factory=list)

    def __str__(self) -> str:
        status = "OK" if self.ok else "FAIL"
        warn   = f"  warnings={self.warnings}" if self.warnings else ""
        return f"[{status}] {self.agent}  ({self.elapsed_s:.2f}s){warn}"


def _log_stage(agent_result: AgentResult, check_result: AgentResult) -> None:
    """Log both agent and checker results; escalate each warning individually."""
    log.info(str(agent_result))
    log.info(str(check_result))
    for w in agent_result.warnings + check_result.warnings:
        log.warning("  ⚠  %s", w)


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD EXTRACTION HELPERS
# Moved from routers/retrieval.py — only used by KeywordAgent.
# Logic is byte-for-byte identical to the original inline functions.
# ══════════════════════════════════════════════════════════════════════════════

def _extract_chunk_keywords_regex(text: str, metadata: dict) -> list:
    """Regex/NER-lite keyword extraction — fast, no model required."""
    words = text.split()
    keyword_parts = []
    i = 0
    while i < len(words):
        clean = re.sub(r"[^a-zA-Z0-9'-]", "", words[i])
        if clean and clean[0].isupper() and len(clean) > 3:
            phrase = [clean]
            j = i + 1
            while j < min(i + 4, len(words)):
                nxt = re.sub(r"[^a-zA-Z0-9'-]", "", words[j])
                if nxt and nxt[0].isupper() and len(nxt) > 2:
                    phrase.append(nxt)
                    j += 1
                else:
                    break
            keyword_parts.append(" ".join(phrase))
            i = j
        else:
            i += 1
    return keyword_parts


def _extract_chunk_keywords_keybert(text: str) -> list:
    """KeyBERT-based semantic keyword extraction using sentence-transformers locally."""
    try:
        import pathlib
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
        _kb = getattr(_extract_chunk_keywords_keybert, "_model", None)
        if _kb is None:
            _cache = (
                pathlib.Path.home()
                / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
            )
            _snapshots = sorted(_cache.iterdir()) if _cache.exists() else []
            _model_path = str(_snapshots[0]) if _snapshots else "all-MiniLM-L6-v2"
            st = SentenceTransformer(_model_path, local_files_only=True)
            _kb = KeyBERT(model=st)
            _extract_chunk_keywords_keybert._model = _kb
        results = _kb.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=10,
            use_mmr=True,
            diversity=0.5,
        )
        return [kw for kw, _ in results]
    except Exception as e:
        log.warning("KeyBERT extraction failed, falling back to regex: %s", e)
        return []


def _extract_chunk_keywords_llama3(text: str, ollama_url: str) -> list:
    """Llama-3 keyword extraction via Ollama — high quality, ~1s per chunk."""
    import json as _json
    import urllib.request

    prompt = (
        "Extract 5 to 10 high-value keywords or key phrases from the following text. "
        "Focus on named entities, proper nouns, technical terms, and domain-specific concepts. "
        "Return only a comma-separated list of keywords, nothing else.\n\nText:\n"
        + text[:1500]
    )
    payload = _json.dumps({
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 100},
    }).encode()
    try:
        req = urllib.request.Request(
            f"{ollama_url.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())
        raw = data.get("response", "").strip()
        keywords = [kw.strip() for kw in raw.split(",") if kw.strip()]
        return keywords[:15]
    except Exception as e:
        log.warning("Llama-3 keyword extraction failed, falling back to regex: %s", e)
        return []


def _extract_chunk_keywords(
    text: str,
    metadata: dict,
    method: str = "regex",
    ollama_url: str = "",
) -> str:
    """Dispatch keyword extraction to the configured method; return comma-separated string."""
    if method == "keybert":
        keyword_parts = _extract_chunk_keywords_keybert(text)
        if not keyword_parts:
            keyword_parts = _extract_chunk_keywords_regex(text, metadata)
    elif method == "llama3":
        keyword_parts = _extract_chunk_keywords_llama3(
            text, ollama_url or "http://localhost:11434"
        )
        if not keyword_parts:
            keyword_parts = _extract_chunk_keywords_regex(text, metadata)
    else:
        keyword_parts = _extract_chunk_keywords_regex(text, metadata)

    # Always append structural metadata fields for precise BM25 anchoring
    for field_name in ("breadcrumb", "chapter", "subject", "class_level", "name"):
        val = metadata.get(field_name)
        if val and isinstance(val, str) and len(val) > 2:
            keyword_parts.append(val)

    # Deduplicate preserving order, cap at 15
    seen: set = set()
    unique: list = []
    for kw in keyword_parts:
        kl = kw.lower()
        if kl not in seen and len(kw) > 3:
            seen.add(kl)
            unique.append(kw)

    return ", ".join(unique[:15])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CHUNKING AGENT + CHECKER
# ══════════════════════════════════════════════════════════════════════════════

class ChunkingAgent:
    """
    Step 3 — SHC-CE 4-Layer Hierarchical Chunking.

    Applies markdown header splitting (L1→L2 structural boundary), then the
    configured chunking strategy:
      "hierarchical" — fixed-size parent→child (L3→L4) with MGI metadata
      "character"/"" — RecursiveCharacterTextSplitter (single level)
      "token"        — TokenTextSplitter via tiktoken (single level)

    Attaches multi-granularity metadata to every child chunk:
      parent_id, parent_text, chunk_id, chapter_id, section_id,
      sibling_chunk_ids, child_chunk_ids, position_in_section,
      content_type, token_count, breadcrumb, header_1, header_2.

    Zero concept change — identical logic to the inline code it replaces.
    merge_fn is passed in to avoid a circular import with routers.retrieval.
    """

    def run(
        self,
        docs: list[Document],
        request,
        merge_fn: Optional[Callable] = None,
    ) -> AgentResult:
        t0       = time.monotonic()
        warnings = []

        # ── Markdown header pre-split ──────────────────────────────────────────
        if request.app.state.config.ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER:
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
            split_docs = []
            for doc in docs:
                split_docs.extend([
                    Document(
                        page_content=chunk.page_content,
                        metadata={**doc.metadata},
                    )
                    for chunk in markdown_splitter.split_text(doc.page_content)
                ])
            docs = split_docs
            print(f"  [1/4] Markdown split   → {len(docs)} sections", flush=True)

            if merge_fn and request.app.state.config.CHUNK_MIN_SIZE_TARGET > 0:
                docs = merge_fn(request, docs)
                print(f"        After merge    → {len(docs)} sections", flush=True)

        # ── Chunking strategy ──────────────────────────────────────────────────
        chunking_strategy = getattr(request.app.state.config, "CHUNKING_STRATEGY", "fixed")

        if chunking_strategy == "hierarchical":
            docs = self._hierarchical_split(docs, request)

        elif request.app.state.config.TEXT_SPLITTER in ["", "character"]:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=request.app.state.config.CHUNK_SIZE,
                chunk_overlap=request.app.state.config.CHUNK_OVERLAP,
                add_start_index=True,
            )
            docs = text_splitter.split_documents(docs)
            print(
                f"  [2/4] Character split  → {len(docs)} chunks"
                f"  (size={request.app.state.config.CHUNK_SIZE},"
                f" overlap={request.app.state.config.CHUNK_OVERLAP})",
                flush=True,
            )

        elif request.app.state.config.TEXT_SPLITTER == "token":
            import tiktoken
            tiktoken.get_encoding(str(request.app.state.config.TIKTOKEN_ENCODING_NAME))
            text_splitter = TokenTextSplitter(
                encoding_name=str(request.app.state.config.TIKTOKEN_ENCODING_NAME),
                chunk_size=request.app.state.config.CHUNK_SIZE,
                chunk_overlap=request.app.state.config.CHUNK_OVERLAP,
                add_start_index=True,
            )
            docs = text_splitter.split_documents(docs)
            print(
                f"  [2/4] Token split      → {len(docs)} chunks"
                f"  (size={request.app.state.config.CHUNK_SIZE},"
                f" overlap={request.app.state.config.CHUNK_OVERLAP},"
                f" enc={request.app.state.config.TIKTOKEN_ENCODING_NAME})",
                flush=True,
            )

        else:
            from futureverse_ai.constants import ERROR_MESSAGES
            raise ValueError(ERROR_MESSAGES.DEFAULT("Invalid text splitter"))

        if not docs:
            warnings.append("No chunks produced after splitting")

        return AgentResult(
            agent="ChunkingAgent",
            ok=bool(docs),
            data=docs,
            elapsed_s=time.monotonic() - t0,
            warnings=warnings,
        )

    def _hierarchical_split(self, docs: list[Document], request) -> list[Document]:
        """
        Fixed-size parent→child hierarchical split with MGI metadata.
        Identical logic to the inline hierarchical block — no change.
        """
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.app.state.config.CHUNK_SIZE,
            chunk_overlap=request.app.state.config.CHUNK_OVERLAP,
        )
        parent_docs = parent_splitter.split_documents(docs)

        child_chunk_size = max(100, request.app.state.config.CHUNK_SIZE // 4)
        child_overlap    = max(20,  request.app.state.config.CHUNK_OVERLAP // 2)
        child_splitter   = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
        )

        child_docs: list[Document] = []
        for parent in parent_docs:
            pid  = str(uuid.uuid4())
            ptxt = parent.page_content
            crumbs = [
                parent.metadata[h]
                for h in ("Header 1", "Header 2", "Header 3", "Header 4")
                if parent.metadata.get(h)
            ]
            breadcrumb = " > ".join(crumbs)
            for child in child_splitter.split_documents([parent]):
                child.metadata["parent_id"]   = pid
                child.metadata["parent_text"] = ptxt
                if breadcrumb:
                    child.metadata["breadcrumb"] = breadcrumb
                child_docs.append(child)

        # Step A — pre-assign stable UUIDs
        for doc in child_docs:
            doc.metadata["chunk_id"] = str(uuid.uuid4())

        # Step B — group children by parent to compute relationships
        _p2c: dict = defaultdict(list)
        for doc in child_docs:
            pid_ = doc.metadata.get("parent_id")
            if pid_:
                _p2c[pid_].append(doc.metadata["chunk_id"])

        # Step C — write relationships + positional + content-type metadata
        for doc in child_docs:
            pid_   = doc.metadata.get("parent_id", "")
            cid_   = doc.metadata["chunk_id"]
            h1_    = doc.metadata.get("Header 1", "")
            fid_   = doc.metadata.get("file_id", "")
            ch_id_ = (
                str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{fid_}::{h1_}")) if h1_ else ""
            )
            siblings_all = _p2c.get(pid_, [cid_])
            pos_         = siblings_all.index(cid_) if cid_ in siblings_all else 0
            doc.metadata.update({
                "document_id":         fid_,
                "chapter_id":          ch_id_,
                "section_id":          pid_,
                "sibling_chunk_ids":   ",".join(s for s in siblings_all if s != cid_),
                "child_chunk_ids":     ",".join(siblings_all),
                "position_in_section": pos_,
                "content_type":        "body_text",
                "token_count":         max(1, len(doc.page_content) // 4),
                "header_1":            doc.metadata.get("Header 1", ""),
                "header_2":            doc.metadata.get("Header 2", ""),
            })

        print(
            f"  [2/4] Hierarchical     → {len(parent_docs)} parents"
            f" → {len(child_docs)} children"
            f"  (parent≤{request.app.state.config.CHUNK_SIZE},"
            f" child={child_chunk_size}, overlap={child_overlap})"
            f"  [MGI: chunk_id + sibling/child links assigned]",
            flush=True,
        )
        return child_docs


class ChunkingChecker:
    """
    Validates ChunkingAgent output before the pipeline proceeds.

    Checks:
      - At least one chunk was produced (hard fail if zero)
      - Empty chunk rate is below 10% (>10% suggests splitter failure)
      - file_id metadata was not silently dropped during splitting
    """

    def check(
        self,
        input_docs: list[Document],
        result: AgentResult,
    ) -> AgentResult:
        t0       = time.monotonic()
        warnings = list(result.warnings)
        ok       = result.ok
        chunks   = result.data or []

        if not chunks:
            return AgentResult(
                agent="ChunkingChecker",
                ok=False,
                data=result.data,
                elapsed_s=time.monotonic() - t0,
                warnings=["No chunks produced — pipeline cannot continue"],
            )

        empty_chunks = [i for i, d in enumerate(chunks) if not d.page_content.strip()]
        if empty_chunks:
            warnings.append(
                f"{len(empty_chunks)} empty chunks at indices {empty_chunks[:5]}"
            )
            if len(empty_chunks) > len(chunks) * 0.1:
                ok = False
                warnings.append(">10% empty chunks — possible splitter failure")

        has_fid_in  = any(d.metadata.get("file_id") for d in input_docs)
        has_fid_out = any(d.metadata.get("file_id") for d in chunks)
        if has_fid_in and not has_fid_out:
            ok = False
            warnings.append("file_id metadata lost during chunking")

        print(
            f"  [CHECK] ChunkingChecker"
            f" → {len(chunks)} chunks, {len(empty_chunks)} empty"
            f"  {'OK' if ok else 'FAIL'}",
            flush=True,
        )
        return AgentResult(
            agent="ChunkingChecker",
            ok=ok,
            data=result.data,
            elapsed_s=time.monotonic() - t0,
            warnings=warnings,
        )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — KEYWORD AGENT + CHECKER
# (runs before contextual enrichment — same order as the existing inline code)
# ══════════════════════════════════════════════════════════════════════════════

class KeywordAgent:
    """
    Step 5 — BM25 Keyword Generation.

    Extracts entity-anchored keywords from each chunk's raw text via the
    configured method (regex / keybert / llama3). Populates
    doc.metadata["keywords"] for Weaviate's native BM25 inverted index.

    Execution order: runs BEFORE contextual enrichment so that the keywords
    field is available when the enricher annotates key_entities. This
    preserves the exact ordering of the original inline code.
    """

    def run(
        self,
        docs: list[Document],
        method: str,
        ollama_url: str,
        enabled: bool = True,
    ) -> AgentResult:
        t0       = time.monotonic()
        warnings = []

        if not enabled:
            print("  [KW] Keyword extraction  SKIPPED (disabled)", flush=True)
            return AgentResult(
                agent="KeywordAgent",
                ok=True,
                data=docs,
                elapsed_s=time.monotonic() - t0,
            )

        kw_count = 0
        for doc in docs:
            raw = doc.metadata.get("raw_text") or doc.page_content
            kw  = _extract_chunk_keywords(
                raw, doc.metadata, method=method, ollama_url=ollama_url
            )
            if kw:
                doc.metadata["keywords"] = kw
                kw_count += 1

        print(
            f"  [KW] Keyword extraction ({method})"
            f" → {kw_count}/{len(docs)} chunks labelled",
            flush=True,
        )
        if kw_count == 0:
            warnings.append("No keywords extracted — BM25 coverage will be degraded")

        return AgentResult(
            agent="KeywordAgent",
            ok=True,
            data=docs,
            elapsed_s=time.monotonic() - t0,
            warnings=warnings,
        )


class KeywordChecker:
    """
    Validates KeywordAgent output.

    Checks:
      - Chunk count is unchanged (keyword extraction must not alter list length)
      - page_content was not mutated (keywords go in metadata only)
      - No chunks have a keywords field that is an empty string
    """

    def check(
        self,
        input_docs: list[Document],
        result: AgentResult,
    ) -> AgentResult:
        t0       = time.monotonic()
        warnings = list(result.warnings)
        ok       = True
        docs     = result.data or []

        if len(docs) != len(input_docs):
            return AgentResult(
                agent="KeywordChecker",
                ok=False,
                data=result.data,
                elapsed_s=time.monotonic() - t0,
                warnings=[
                    f"Chunk count changed: {len(input_docs)} → {len(docs)}"
                ],
            )

        bad_kw = [
            i for i, d in enumerate(docs)
            if "keywords" in d.metadata and not d.metadata["keywords"].strip()
        ]
        if bad_kw:
            warnings.append(f"{len(bad_kw)} chunks have empty keywords field")

        corrupted = [
            i for i in range(len(docs))
            if docs[i].page_content != input_docs[i].page_content
        ]
        if corrupted:
            ok = False
            warnings.append(
                f"page_content mutated in {len(corrupted)} chunks during keyword extraction"
            )

        print(
            f"  [CHECK] KeywordChecker"
            f" → {len(docs)} chunks, {len(bad_kw)} empty-kw"
            f"  {'OK' if ok else 'FAIL'}",
            flush=True,
        )
        return AgentResult(
            agent="KeywordChecker",
            ok=ok,
            data=result.data,
            elapsed_s=time.monotonic() - t0,
            warnings=warnings,
        )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CONTEXTUAL ENRICHMENT AGENT + CHECKER
# ══════════════════════════════════════════════════════════════════════════════

class ContextualEnrichmentAgent:
    """
    Step 4 — Contextual Enrichment (Anthropic paper approach).

    Calls apply_contextual_chunking() — same logic as the inline code.
    Each chunk gets a structured [CONTEXT]/[CONTENT] prefix generated by
    qwen2.5:7b. The enriched text is used for embedding; the original raw
    text is stored in metadata["raw_text"] for LLM display (search small,
    read big).

    After enrichment, extracts Layer 4 post-contextual fields:
      topics      — from "Topic:" line in the [CONTEXT] block
      key_entities — alias of keywords field (schema alignment)

    Graceful fallback: raw text is always preserved if Ollama is unavailable.
    Never blocks an upload — failures are logged as warnings only.
    """

    def run(
        self,
        docs: list[Document],
        full_doc_text: str,
        ollama_url: str,
        enabled: bool = True,
        available: bool = True,
    ) -> AgentResult:
        t0       = time.monotonic()
        warnings = []

        if not enabled or not available:
            reason = "disabled" if not enabled else "module unavailable"
            print(f"  [3/4] Contextual chunking  SKIPPED ({reason})", flush=True)
            self._extract_post_contextual_fields(docs)
            return AgentResult(
                agent="ContextualEnrichmentAgent",
                ok=True,
                data=docs,
                elapsed_s=time.monotonic() - t0,
            )

        try:
            from futureverse_ai.retrieval.contextual_chunking import (
                apply_contextual_chunking,
            )
            print("  [3/4] Contextual chunking  ENABLED — calling Ollama ...", flush=True)
            docs = apply_contextual_chunking(docs, full_doc_text, ollama_url=ollama_url)

            ctx_ok  = sum(1 for d in docs if d.metadata.get("has_context"))
            elapsed = time.monotonic() - t0
            print(
                f"  [3/4] Contextual chunking  DONE"
                f" — {ctx_ok}/{len(docs)} enriched  ({elapsed:.1f}s)",
                flush=True,
            )
            if ctx_ok < len(docs) * 0.5:
                warnings.append(
                    f"Only {ctx_ok}/{len(docs)} chunks enriched"
                    " — Ollama may be overloaded or OLLAMA_NUM_PARALLEL too low"
                )
        except Exception as e:
            print(
                f"  [3/4] Contextual chunking  ERROR — raw fallback: {e}",
                flush=True,
            )
            log.warning("Contextual enrichment error: %s", e)
            warnings.append(f"Contextual enrichment failed: {e}")

        # Layer 4 post-contextual: extract topics + key_entities from context_summary
        self._extract_post_contextual_fields(docs)

        return AgentResult(
            agent="ContextualEnrichmentAgent",
            ok=True,
            data=docs,
            elapsed_s=time.monotonic() - t0,
            warnings=warnings,
        )

    def _extract_post_contextual_fields(self, docs: list[Document]) -> None:
        """
        Extract topics and key_entities from context metadata.
        Identical to the inline Layer 4 post-contextual block.
        """
        for doc in docs:
            ctx = doc.metadata.get("context_summary", "")
            if ctx:
                m = re.search(r'^Topic:\s*(.+)$', ctx, re.MULTILINE)
                if m:
                    doc.metadata["topics"] = m.group(1).strip()
            kw = doc.metadata.get("keywords", "")
            if kw:
                doc.metadata["key_entities"] = kw


class ContextualEnrichmentChecker:
    """
    Validates ContextualEnrichmentAgent output.

    Checks:
      - Chunk count is unchanged (enrichment never adds or removes chunks)
      - Every enriched chunk (has_context=True) has raw_text in metadata —
        without it the LLM would see the [CONTEXT] prefix directly (display bug)
    """

    def check(
        self,
        input_docs: list[Document],
        result: AgentResult,
    ) -> AgentResult:
        t0       = time.monotonic()
        warnings = list(result.warnings)
        ok       = True
        docs     = result.data or []

        if len(docs) != len(input_docs):
            return AgentResult(
                agent="ContextualEnrichmentChecker",
                ok=False,
                data=result.data,
                elapsed_s=time.monotonic() - t0,
                warnings=[
                    f"Chunk count changed: {len(input_docs)} → {len(docs)}"
                ],
            )

        missing_raw = [
            i for i, d in enumerate(docs)
            if d.metadata.get("has_context") and "raw_text" not in d.metadata
        ]
        if missing_raw:
            ok = False
            warnings.append(
                f"{len(missing_raw)} enriched chunks missing raw_text — "
                "LLM will see [CONTEXT] prefix directly (display corruption)"
            )

        enriched_count = sum(1 for d in docs if d.metadata.get("has_context"))
        print(
            f"  [CHECK] EnrichmentChecker"
            f" → {enriched_count}/{len(docs)} enriched,"
            f" {len(missing_raw)} missing raw_text"
            f"  {'OK' if ok else 'FAIL'}",
            flush=True,
        )
        return AgentResult(
            agent="ContextualEnrichmentChecker",
            ok=ok,
            data=result.data,
            elapsed_s=time.monotonic() - t0,
            warnings=warnings,
        )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — EMBEDDING AGENT + CHECKER
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingAgent:
    """
    Step 6 — bge-m3 GPU Embedding (1024-dim vectors).

    Converts sanitised chunk texts to dense vectors using the configured
    embedding engine (sentence_transformers / openai / ollama / azure_openai).
    Bridges to the main asyncio event loop via run_coroutine_threadsafe —
    identical mechanism to the existing inline code.
    """

    def run(
        self,
        texts: list[str],
        embedding_function,
        prefix: str,
        main_loop,
        timeout: int,
        user,
    ) -> AgentResult:
        import asyncio
        t0 = time.monotonic()

        future = asyncio.run_coroutine_threadsafe(
            embedding_function(
                list(map(lambda x: x.replace("\n", " "), texts)),
                prefix=prefix,
                user=user,
            ),
            main_loop,
        )
        embeddings = future.result(timeout=timeout)
        elapsed    = time.monotonic() - t0
        per_ms     = (elapsed / len(embeddings) * 1000) if embeddings else 0

        print(
            f"  [4/4] Embedding        DONE"
            f" — {len(embeddings)} vectors  ({elapsed:.1f}s, {per_ms:.0f}ms/chunk)",
            flush=True,
        )
        return AgentResult(
            agent="EmbeddingAgent",
            ok=bool(embeddings),
            data=embeddings,
            elapsed_s=elapsed,
        )


class EmbeddingChecker:
    """
    Validates EmbeddingAgent output.

    Checks:
      - At least one embedding was produced (hard fail if zero)
      - Embedding count matches text count (embedding APIs may silently drop chunks)
      - Vector dimension is non-zero (catches model load failures)
      - No all-zero vectors (indicates a silent model error)
    """

    def check(
        self,
        texts: list[str],
        result: AgentResult,
        expected_dim: int = 0,
    ) -> AgentResult:
        t0         = time.monotonic()
        warnings   = list(result.warnings)
        ok         = True
        embeddings = result.data or []

        if not embeddings:
            return AgentResult(
                agent="EmbeddingChecker",
                ok=False,
                data=result.data,
                elapsed_s=time.monotonic() - t0,
                warnings=["No embeddings produced"],
            )

        if len(embeddings) != len(texts):
            warnings.append(
                f"Embedding count mismatch: {len(embeddings)} vectors"
                f" for {len(texts)} texts"
                f" — using min({len(embeddings)},{len(texts)})"
            )
            log.warning(
                "Embedding count mismatch: %d embeddings for %d texts. "
                "Using min(%d, %d) items.",
                len(embeddings), len(texts), len(embeddings), len(texts),
            )

        actual_dim = len(embeddings[0]) if embeddings else 0
        if actual_dim == 0:
            ok = False
            warnings.append("Zero-dimension vectors returned by embedding model")
        elif expected_dim > 0 and actual_dim != expected_dim:
            warnings.append(
                f"Unexpected embedding dim: got {actual_dim}, expected {expected_dim}"
            )

        zero_vecs = sum(
            1 for v in embeddings if v and all(abs(x) < 1e-9 for x in v[:10])
        )
        if zero_vecs:
            warnings.append(f"{zero_vecs} near-zero embedding vectors detected")

        print(
            f"  [CHECK] EmbeddingChecker"
            f" → {len(embeddings)}/{len(texts)} vectors, dim={actual_dim}"
            f"  {'OK' if ok else 'FAIL'}",
            flush=True,
        )
        return AgentResult(
            agent="EmbeddingChecker",
            ok=ok,
            data=result.data,
            elapsed_s=time.monotonic() - t0,
            warnings=warnings,
        )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — INDEXING AGENT + CHECKER
# ══════════════════════════════════════════════════════════════════════════════

class IndexingAgent:
    """
    Step 7 — Weaviate Dual-Index Insert.

    Manages collection lifecycle (overwrite / skip-if-populated / re-insert-if-empty),
    then inserts items into Weaviate. Each item is indexed with:
      - HNSW vector index (for semantic search)
      - Inverted index     (for BM25 keyword search)
    Identical logic to the inline collection lifecycle + insert block.
    """

    def run(
        self,
        collection_name: str,
        items: list[dict],
        overwrite: bool,
        add: bool,
        vector_db_client,
    ) -> AgentResult:
        t0       = time.monotonic()
        warnings = []

        if vector_db_client.has_collection(collection_name=collection_name):
            log.info(f"collection {collection_name} already exists")

            if overwrite:
                vector_db_client.delete_collection(collection_name=collection_name)
                log.info(f"deleting existing collection {collection_name}")

            elif not add:
                existing = vector_db_client.get(collection_name=collection_name)
                has_data = bool(existing and existing.ids and existing.ids[0])

                if has_data:
                    log.info(
                        f"collection {collection_name} already exists with data, skipping"
                    )
                    print(
                        f"  [4/4] Embedding SKIPPED — collection already populated"
                        f" ({len(existing.ids[0])} items)",
                        flush=True,
                    )
                    return AgentResult(
                        agent="IndexingAgent",
                        ok=True,
                        data={"stored": len(existing.ids[0]), "skipped": True},
                        elapsed_s=time.monotonic() - t0,
                    )
                else:
                    log.info(
                        f"collection {collection_name} exists but is EMPTY,"
                        " deleting and re-inserting"
                    )
                    print(
                        f"  [4/4] Collection exists but is empty — re-embedding ...",
                        flush=True,
                    )
                    vector_db_client.delete_collection(collection_name=collection_name)

        log.info(f"adding to collection {collection_name}")
        vector_db_client.insert(collection_name=collection_name, items=items)

        return AgentResult(
            agent="IndexingAgent",
            ok=True,
            data={"stored": len(items), "skipped": False},
            elapsed_s=time.monotonic() - t0,
            warnings=warnings,
        )


class IndexingChecker:
    """
    Validates IndexingAgent output via post-insert Weaviate query.

    Weaviate may silently accept an insert call but store zero objects if
    the batch fails internally. This checker catches that by querying the
    collection after insert and comparing the stored count against expected.
    """

    def check(
        self,
        collection_name: str,
        expected: int,
        result: AgentResult,
        vector_db_client,
    ) -> AgentResult:
        t0       = time.monotonic()
        warnings = list(result.warnings)
        ok       = True

        if result.data and result.data.get("skipped"):
            return AgentResult(
                agent="IndexingChecker",
                ok=True,
                data=result.data,
                elapsed_s=time.monotonic() - t0,
            )

        verify = vector_db_client.get(collection_name=collection_name)
        stored = (
            len(verify.ids[0])
            if (verify and verify.ids and verify.ids[0])
            else 0
        )

        if stored == 0:
            ok = False
            log.error(
                "Post-insert verification FAILED: 0 objects in '%s' after"
                " inserting %d items. Batch likely failed silently.",
                collection_name, expected,
            )
            warnings.append(
                f"0/{expected} items stored in Weaviate — batch insert failed"
            )
        elif stored < expected:
            warnings.append(f"Partial insert: {stored}/{expected} items stored")

        result.data["verified_stored"] = stored
        print(
            f"  [CHECK] IndexingChecker"
            f" → {stored}/{expected} stored in {collection_name}"
            f"  {'OK' if ok else 'FAIL'}",
            flush=True,
        )
        return AgentResult(
            agent="IndexingChecker",
            ok=ok,
            data=result.data,
            elapsed_s=time.monotonic() - t0,
            warnings=warnings,
        )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5.5 — CHUNK DEDUPLICATION AGENT + CHECKER
# ══════════════════════════════════════════════════════════════════════════════

class ChunkDeduplicationAgent:
    """
    Step 5.5 — Chunk-level exact content deduplication (Layer 1).

    Runs BEFORE embedding (Step 6) so duplicate chunks never reach the GPU/API.

    Two-pass deduplication strategy:

    Pass 1 — file_id grouped:
      Chunks are grouped by their ``file_id`` metadata field.  For each unique
      file_id, ONE cheap query (limit=1) checks whether ANY chunk from that file
      is already stored in the collection.  If so, ALL chunks from that file_id
      are skipped — they were already fully indexed in a previous run.  This
      handles the most common re-ingestion case (same chapter/file uploaded again
      or KB rebuilt with unchanged documents) using only simple ``.equal()``
      filters that every vector-DB backend supports.

    Pass 2 — file-level hash fallback (re-upload detection):
      After the file_id pass, surviving chunks are grouped by the file-level
      ``hash`` field (SHA-256 of the full file text, set by retrieval.py).
      For each unique hash, ONE query checks whether that content hash already
      exists in the collection.  If it does, the file was re-uploaded under a
      new file_id — ALL chunks from that hash are skipped.  This prevents
      duplicates when a user deletes and re-uploads the exact same file.

    Additionally:
      SHA-256 of each chunk's raw_text is stamped as ``chunk_hash`` in the
      metadata so it is stored in the vector DB for potential future hash-level
      dedup queries.

    Behaviour when disabled: hashes are still stamped onto metadata (so future
    runs benefit), but no chunks are filtered — all pass through unchanged.
    Behaviour on query error: logs a warning, returns all chunks unchanged
    (fail-open — never block ingestion due to a dedup lookup failure).
    """

    def run(
        self,
        texts: list[str],
        metadatas: list[dict],
        collection_name: str,
        vector_db_client,
        enabled: bool = True,
    ) -> "AgentResult":
        import hashlib
        t0 = time.monotonic()

        # Always stamp chunk_hash so it lands in the vector store for future runs
        for i, m in enumerate(metadatas):
            raw = m.get("raw_text") or texts[i]
            m["chunk_hash"] = hashlib.sha256(raw.encode()).hexdigest()

        if not enabled:
            print(
                f"  [5.5] ChunkDedup       DISABLED"
                f" — {len(texts)} chunks passed through",
                flush=True,
            )
            return AgentResult(
                agent="ChunkDeduplicationAgent",
                ok=True,
                data={"texts": texts, "metadatas": metadatas, "skipped": 0},
                elapsed_s=time.monotonic() - t0,
            )

        # ── Strategy: group chunks by file_id, query once per unique file_id ──
        # This avoids a large $in filter and works with the existing .equal()
        # filter support. If a file_id already has ANY chunk in the collection,
        # ALL chunks from that file are skipped (they're already indexed).
        # Chunks with no file_id fall back to individual chunk_hash equality queries.
        new_texts:  list = []
        new_metas:  list = []
        skipped          = 0

        try:
            if not vector_db_client.has_collection(collection_name=collection_name):
                # Brand-new collection — nothing can be a duplicate
                new_texts  = list(texts)
                new_metas  = list(metadatas)
            else:
                # Group indices by file_id
                from collections import defaultdict as _dd
                fid_groups: dict = _dd(list)   # file_id → [idx, ...]
                no_fid_indices: list = []

                for idx, m in enumerate(metadatas):
                    fid = m.get("file_id")
                    if fid:
                        fid_groups[fid].append(idx)
                    else:
                        no_fid_indices.append(idx)

                # Per-file_id: one cheap query (limit=1) to detect existing chunks
                already_indexed: set = set()   # file_ids already in the collection
                for fid in fid_groups:
                    result = vector_db_client.query(
                        collection_name=collection_name,
                        filter={"file_id": fid},
                        limit=1,
                    )
                    if result and result.ids and result.ids[0]:
                        already_indexed.add(fid)
                        log.debug(
                            "[DEDUP-L1] file_id=%s already indexed in %s — "
                            "skipping %d chunks",
                            fid[:16], collection_name, len(fid_groups[fid]),
                        )

                for idx, (text, meta) in enumerate(zip(texts, metadatas)):
                    fid = meta.get("file_id")
                    if fid and fid in already_indexed:
                        skipped += 1
                    else:
                        new_texts.append(text)
                        new_metas.append(meta)

                # ── Hash-based fallback: catches re-uploads with a new file_id ──
                # When the same file is re-uploaded it gets a new file_id, so the
                # file_id-based check above won't catch it.  The file-level `hash`
                # field (SHA-256 of the full text, set in retrieval.py) is stable
                # across re-uploads, so we use it as a second-pass filter.
                # Only chunks that SURVIVED the file_id filter are examined here.
                if new_texts:
                    from collections import defaultdict as _ddhash
                    hash_groups: dict = _ddhash(list)   # file-level hash → [pos in new_texts]
                    for pos, m in enumerate(new_metas):
                        h = m.get("hash")   # file-level hash stamped by retrieval.py
                        if h:
                            hash_groups[h].append(pos)

                    if hash_groups:
                        already_hashed: set = set()
                        for h in hash_groups:
                            res2 = vector_db_client.query(
                                collection_name=collection_name,
                                filter={"hash": h},
                                limit=1,
                            )
                            if res2 and res2.ids and res2.ids[0]:
                                already_hashed.add(h)
                                log.debug(
                                    "[DEDUP-L1] file hash=%s already indexed in %s"
                                    " — skipping %d chunks"
                                    " (same content re-uploaded with new file_id)",
                                    h[:16], collection_name, len(hash_groups[h]),
                                )

                        if already_hashed:
                            final_texts: list = []
                            final_metas: list = []
                            for t, m in zip(new_texts, new_metas):
                                h = m.get("hash")
                                if h and h in already_hashed:
                                    skipped += 1
                                else:
                                    final_texts.append(t)
                                    final_metas.append(m)
                            new_texts = final_texts
                            new_metas = final_metas

        except Exception as exc:
            log.warning(
                "[DEDUP-L1] Dedup query failed (fail-open) — "
                "all %d chunks will be re-embedded: %s",
                len(texts), exc,
            )
            print(
                f"  [5.5] ChunkDedup       WARN — query failed, all chunks forwarded",
                flush=True,
            )
            return AgentResult(
                agent="ChunkDeduplicationAgent",
                ok=True,
                data={"texts": texts, "metadatas": metadatas, "skipped": 0},
                elapsed_s=time.monotonic() - t0,
                warnings=[f"Dedup query failed (fail-open): {exc}"],
            )

        print(
            f"  [5.5] ChunkDedup       "
            f"{len(texts)} → {len(new_texts)} chunks"
            f"  ({skipped} exact duplicates skipped)",
            flush=True,
        )
        return AgentResult(
            agent="ChunkDeduplicationAgent",
            ok=True,
            data={"texts": new_texts, "metadatas": new_metas, "skipped": skipped},
            elapsed_s=time.monotonic() - t0,
        )


class ChunkDeduplicationChecker:
    """
    Validates ChunkDeduplicationAgent output.

    Hard-fails only if the agent returned None data (internal crash).
    A skipped=all result is valid and expected for fully re-ingested documents
    — emits a warning but never aborts the pipeline.
    """

    def check(
        self,
        original_count: int,
        result: "AgentResult",
    ) -> "AgentResult":
        t0       = time.monotonic()
        warnings = list(result.warnings)
        ok       = True

        if result.data is None:
            return AgentResult(
                agent="ChunkDeduplicationChecker",
                ok=False,
                data=result.data,
                elapsed_s=time.monotonic() - t0,
                warnings=["ChunkDeduplicationAgent returned no data (internal error)"],
            )

        new_count = len(result.data.get("texts", []))
        skipped   = result.data.get("skipped", 0)

        if new_count == 0 and original_count > 0:
            warnings.append(
                f"All {original_count} chunks are exact duplicates — "
                "document fully skipped (normal for re-ingestion of unchanged files)"
            )

        print(
            f"  [CHECK] ChunkDedupChecker"
            f" → {new_count}/{original_count} new"
            f"  ({'OK' if ok else 'FAIL'})",
            flush=True,
        )
        return AgentResult(
            agent="ChunkDeduplicationChecker",
            ok=ok,
            data=result.data,
            elapsed_s=time.monotonic() - t0,
            warnings=warnings,
        )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6.5 — SEMANTIC DEDUPLICATION AGENT + CHECKER
# ══════════════════════════════════════════════════════════════════════════════

class SemanticDeduplicationAgent:
    """
    Step 6.5 — Semantic near-duplicate deduplication (Layer 2).

    Runs AFTER embedding (Step 6) and BEFORE indexing (Step 7).
    For each new embedding, queries the existing collection for its nearest
    neighbour. If cosine similarity >= threshold, the chunk is treated as a
    near-duplicate (paraphrase, lightly reformatted copy) and dropped.

    Disabled by default (ENABLE_SEMANTIC_DEDUP=false). Enable when content
    sources contain heavily reworded duplicates — e.g. textbook editions,
    scraped pages with shared boilerplate, or OCR re-scans of the same material.

    Fail-open: if a per-item similarity search throws, that item is kept.
    First-time ingestion (collection doesn't exist yet) is always a full pass.
    """

    def run(
        self,
        items: list[dict],
        collection_name: str,
        vector_db_client,
        threshold: float = 0.97,
        enabled: bool = False,
    ) -> "AgentResult":
        t0 = time.monotonic()

        if not enabled:
            print(
                f"  [6.5] SemanticDedup    DISABLED"
                f" — {len(items)} items passed through",
                flush=True,
            )
            return AgentResult(
                agent="SemanticDeduplicationAgent",
                ok=True,
                data=items,
                elapsed_s=time.monotonic() - t0,
            )

        if not vector_db_client.has_collection(collection_name=collection_name):
            print(
                f"  [6.5] SemanticDedup    SKIP (new collection)"
                f" — all {len(items)} items are new",
                flush=True,
            )
            return AgentResult(
                agent="SemanticDeduplicationAgent",
                ok=True,
                data=items,
                elapsed_s=time.monotonic() - t0,
            )

        new_items: list = []
        skipped         = 0

        for item in items:
            try:
                result = vector_db_client.search(
                    collection_name=collection_name,
                    query_vector=item["vector"],
                    limit=1,
                )
                if result and result.ids and result.ids[0]:
                    top_score = (
                        result.distances[0][0]
                        if (result.distances and result.distances[0])
                        else 0.0
                    )
                    if top_score >= threshold:
                        skipped += 1
                        log.debug(
                            "[DEDUP-L2] Near-duplicate skipped"
                            " — similarity=%.4f >= threshold=%.4f, chunk_id=%s",
                            top_score, threshold,
                            str(item["metadata"].get("chunk_id", item["id"]))[:16],
                        )
                        continue
            except Exception as exc:
                log.warning(
                    "[DEDUP-L2] Similarity search failed for item %s"
                    " — keeping it (fail-open): %s",
                    str(item["id"])[:16], exc,
                )
            new_items.append(item)

        print(
            f"  [6.5] SemanticDedup    "
            f"{len(items)} → {len(new_items)} items"
            f"  ({skipped} near-duplicates skipped, threshold={threshold})",
            flush=True,
        )
        return AgentResult(
            agent="SemanticDeduplicationAgent",
            ok=True,
            data=new_items,
            elapsed_s=time.monotonic() - t0,
        )


class SemanticDeduplicationChecker:
    """
    Validates SemanticDeduplicationAgent output.
    Never hard-fails — semantic dedup is best-effort and always fail-open.
    """

    def check(
        self,
        original_count: int,
        result: "AgentResult",
    ) -> "AgentResult":
        t0       = time.monotonic()
        warnings = list(result.warnings)
        ok       = True

        if result.data is None:
            return AgentResult(
                agent="SemanticDeduplicationChecker",
                ok=False,
                data=result.data,
                elapsed_s=time.monotonic() - t0,
                warnings=["SemanticDeduplicationAgent returned no data"],
            )

        new_count = len(result.data)

        if new_count == 0 and original_count > 0:
            warnings.append(
                f"All {original_count} items are semantic near-duplicates — "
                "document fully skipped"
            )

        print(
            f"  [CHECK] SemanticDedupChecker"
            f" → {new_count}/{original_count} unique"
            f"  ({'OK' if ok else 'FAIL'})",
            flush=True,
        )
        return AgentResult(
            agent="SemanticDeduplicationChecker",
            ok=ok,
            data=result.data,
            elapsed_s=time.monotonic() - t0,
            warnings=warnings,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_ingestion_pipeline(
    *,
    docs: list[Document],
    collection_name: str,
    full_doc_text: str,
    request,
    metadata: Optional[dict],
    overwrite: bool,
    split: bool,
    add: bool,
    user,
    pipeline_start: float,
    merge_fn: Optional[Callable] = None,
) -> bool:
    """
    Sync orchestrator for the Steps 3–7 ingestion pipeline.

    Execution order:
      Step 3    ChunkingAgent              → ChunkingChecker
      Step 5    KeywordAgent               → KeywordChecker        (before enrichment)
      Step 4    ContextualEnrichment       → EnrichmentChecker     (after keywords)
                Build texts + metadatas
      Step 5.5  ChunkDeduplicationAgent    → ChunkDeduplicationChecker
                  ↳ SHA-256 stamp on raw_text; two-pass dedup:
                    Pass 1: group by file_id → one query/file_id → skip
                      all chunks from files already in collection
                    Pass 2: group survivors by file-level hash → one
                      query/hash → skip re-uploads with new file_id
                    (fail-open on any query error)
      Step 6    EmbeddingAgent             → EmbeddingChecker      (smaller batch now)
                Build items list
      Step 6.5  SemanticDeduplicationAgent → SemanticDeduplicationChecker
                  ↳ cosine-similarity search per vector, drop near-duplicates
                    BEFORE writing to the vector store (off by default)
      Step 7    IndexingAgent              → IndexingChecker

    A checker with ok=False raises ValueError and aborts the pipeline.
    Checker warnings are logged but never abort (graceful degradation).

    Called from save_docs_to_vector_db() in routers/retrieval.py.
    merge_fn=merge_docs_to_target_size is passed in to avoid circular import.
    """
    # Deferred imports — keep at function level to avoid circular imports
    from futureverse_ai.config import RAG_EMBEDDING_CONTENT_PREFIX
    from futureverse_ai.constants import ERROR_MESSAGES
    from futureverse_ai.env import RAG_EMBEDDING_TIMEOUT
    from futureverse_ai.retrieval.utils import get_embedding_function
    from futureverse_ai.retrieval.vector.factory import VECTOR_DB_CLIENT
    from futureverse_ai.utils.misc import sanitize_text_for_db

    # ── Instantiate agents + checkers ─────────────────────────────────────────
    chunking_agent    = ChunkingAgent()
    chunking_checker  = ChunkingChecker()
    keyword_agent     = KeywordAgent()
    keyword_checker   = KeywordChecker()
    ctx_agent         = ContextualEnrichmentAgent()
    ctx_checker       = ContextualEnrichmentChecker()
    dedup_agent       = ChunkDeduplicationAgent()
    dedup_checker     = ChunkDeduplicationChecker()
    embed_agent       = EmbeddingAgent()
    embed_checker     = EmbeddingChecker()
    sem_dedup_agent   = SemanticDeduplicationAgent()
    sem_dedup_checker = SemanticDeduplicationChecker()
    index_agent       = IndexingAgent()
    index_checker     = IndexingChecker()

    if split:
        # ── Step 3: Chunking ──────────────────────────────────────────────────
        raw_input    = list(docs)
        chunk_result = chunking_agent.run(docs, request, merge_fn=merge_fn)
        chunk_check  = chunking_checker.check(raw_input, chunk_result)
        _log_stage(chunk_result, chunk_check)
        if not chunk_check.ok:
            raise ValueError(f"ChunkingAgent failed: {chunk_check.warnings}")
        docs = chunk_result.data

        if not docs:
            raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)

        # ── Step 5: Keyword Generation ────────────────────────────────────────
        _kw_enabled = getattr(request.app.state.config, "ENABLE_KEYWORD_EXTRACTION", True)
        _kw_method  = getattr(request.app.state.config, "KEYWORD_EXTRACTION_METHOD", "regex")
        _ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        pre_kw    = list(docs)
        kw_result = keyword_agent.run(docs, _kw_method, _ollama_url, enabled=_kw_enabled)
        kw_check  = keyword_checker.check(pre_kw, kw_result)
        _log_stage(kw_result, kw_check)
        if not kw_check.ok:
            log.warning(
                "KeywordChecker failed — continuing without full keyword coverage: %s",
                kw_check.warnings,
            )
        docs = kw_result.data

        # ── Step 4: Contextual Enrichment ─────────────────────────────────────
        try:
            from futureverse_ai.retrieval.contextual_chunking import (  # noqa: F401
                apply_contextual_chunking as _cc,
            )
            _ctx_available = True
        except Exception:
            _ctx_available = False

        _ctx_enabled = request.app.state.config.ENABLE_CONTEXTUAL_CHUNKING

        pre_ctx    = list(docs)
        ctx_result = ctx_agent.run(
            docs, full_doc_text, _ollama_url,
            enabled=_ctx_enabled, available=_ctx_available,
        )
        ctx_check  = ctx_checker.check(pre_ctx, ctx_result)
        _log_stage(ctx_result, ctx_check)
        if not ctx_check.ok:
            log.warning(
                "EnrichmentChecker failed — continuing with raw fallback: %s",
                ctx_check.warnings,
            )
        docs = ctx_result.data

    # ── Build texts + metadatas (both split=True and split=False merge here) ──
    if not docs:
        raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)

    texts     = [sanitize_text_for_db(doc.page_content) for doc in docs]
    metadatas = [
        {
            **doc.metadata,
            **(metadata if metadata else {}),
            "embedding_config": {
                "engine": request.app.state.config.RAG_EMBEDDING_ENGINE,
                "model":  request.app.state.config.RAG_EMBEDDING_MODEL,
            },
        }
        for doc in docs
    ]

    # ── Step 5.5: Chunk-level Exact Deduplication ────────────────────────────
    _chunk_dedup_enabled = getattr(
        request.app.state.config, "ENABLE_CHUNK_DEDUP", True
    )
    dedup_result  = dedup_agent.run(
        texts, metadatas, collection_name, VECTOR_DB_CLIENT,
        enabled=_chunk_dedup_enabled,
    )
    dedup_check   = dedup_checker.check(len(texts), dedup_result)
    _log_stage(dedup_result, dedup_check)
    if not dedup_check.ok:
        raise ValueError(f"ChunkDeduplicationAgent failed: {dedup_check.warnings}")

    texts     = dedup_result.data["texts"]
    metadatas = dedup_result.data["metadatas"]

    # All chunks already indexed — nothing new to embed or store
    if not texts:
        total_secs = time.monotonic() - pipeline_start
        print("━" * 62, flush=True)
        print(
            f"PIPELINE COMPLETE  fully deduplicated"
            f" → {collection_name}  ({total_secs:.1f}s)"
            f"  [all chunks were exact duplicates, skipped]",
            flush=True,
        )
        print("━" * 62, flush=True)
        return True

    # ── Step 6: Embedding ─────────────────────────────────────────────────────
    print(
        f"  [4/4] Embedding        {len(texts)} chunks"
        f" → {collection_name}"
        f"  (model={request.app.state.config.RAG_EMBEDDING_MODEL}) ...",
        flush=True,
    )

    embedding_function = get_embedding_function(
        request.app.state.config.RAG_EMBEDDING_ENGINE,
        request.app.state.config.RAG_EMBEDDING_MODEL,
        request.app.state.ef,
        (
            request.app.state.config.RAG_OPENAI_API_BASE_URL
            if request.app.state.config.RAG_EMBEDDING_ENGINE == "openai"
            else (
                request.app.state.config.RAG_OLLAMA_BASE_URL
                if request.app.state.config.RAG_EMBEDDING_ENGINE == "ollama"
                else request.app.state.config.RAG_AZURE_OPENAI_BASE_URL
            )
        ),
        (
            request.app.state.config.RAG_OPENAI_API_KEY
            if request.app.state.config.RAG_EMBEDDING_ENGINE == "openai"
            else (
                request.app.state.config.RAG_OLLAMA_API_KEY
                if request.app.state.config.RAG_EMBEDDING_ENGINE == "ollama"
                else request.app.state.config.RAG_AZURE_OPENAI_API_KEY
            )
        ),
        request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
        azure_api_version=(
            request.app.state.config.RAG_AZURE_OPENAI_API_VERSION
            if request.app.state.config.RAG_EMBEDDING_ENGINE == "azure_openai"
            else None
        ),
        enable_async=request.app.state.config.ENABLE_ASYNC_EMBEDDING,
    )

    embed_result = embed_agent.run(
        texts, embedding_function, RAG_EMBEDDING_CONTENT_PREFIX,
        request.app.state.main_loop, RAG_EMBEDDING_TIMEOUT, user,
    )
    embed_check  = embed_checker.check(texts, embed_result)
    _log_stage(embed_result, embed_check)
    if not embed_check.ok:
        raise ValueError(f"EmbeddingAgent failed: {embed_check.warnings}")

    embeddings = embed_result.data
    safe_count = min(len(embeddings), len(texts), len(metadatas))
    items = [
        {
            "id":       metadatas[idx].get("chunk_id") or str(uuid.uuid4()),
            "text":     texts[idx],
            "vector":   embeddings[idx],
            "metadata": metadatas[idx],
        }
        for idx in range(safe_count)
    ]

    # ── Step 6.5: Semantic Near-Duplicate Deduplication ──────────────────────
    _sem_dedup_enabled   = getattr(
        request.app.state.config, "ENABLE_SEMANTIC_DEDUP", False
    )
    _sem_dedup_threshold = float(getattr(
        request.app.state.config, "SEMANTIC_DEDUP_THRESHOLD", 0.97
    ))
    sem_result  = sem_dedup_agent.run(
        items, collection_name, VECTOR_DB_CLIENT,
        threshold=_sem_dedup_threshold,
        enabled=_sem_dedup_enabled,
    )
    sem_check   = sem_dedup_checker.check(len(items), sem_result)
    _log_stage(sem_result, sem_check)
    if not sem_check.ok:
        raise ValueError(f"SemanticDeduplicationAgent failed: {sem_check.warnings}")

    items = sem_result.data

    # All items were semantic near-duplicates — nothing to store
    if not items:
        total_secs = time.monotonic() - pipeline_start
        print("━" * 62, flush=True)
        print(
            f"PIPELINE COMPLETE  fully deduplicated (semantic)"
            f" → {collection_name}  ({total_secs:.1f}s)"
            f"  [all items were near-duplicates at threshold={_sem_dedup_threshold}]",
            flush=True,
        )
        print("━" * 62, flush=True)
        return True

    # ── Step 7: Weaviate Indexing ─────────────────────────────────────────────
    index_result = index_agent.run(
        collection_name, items, overwrite, add, VECTOR_DB_CLIENT
    )
    index_check  = index_checker.check(
        collection_name, len(items), index_result, VECTOR_DB_CLIENT
    )
    _log_stage(index_result, index_check)

    # ── Final pipeline summary ────────────────────────────────────────────────
    total_secs = time.monotonic() - pipeline_start
    d          = index_result.data or {}
    stored     = d.get("verified_stored", d.get("stored", 0))
    skipped    = d.get("skipped", False)

    print("━" * 62, flush=True)
    if stored > 0 or skipped:
        print(
            f"PIPELINE COMPLETE  {stored}/{len(items)} items stored"
            f" → {collection_name}  ({total_secs:.1f}s)",
            flush=True,
        )
    else:
        print(
            f"PIPELINE ERROR  0/{len(items)} items stored in Weaviate!"
            f" → {collection_name}  ({total_secs:.1f}s)",
            flush=True,
        )
    print("━" * 62, flush=True)

    if not index_check.ok:
        raise ValueError(f"IndexingAgent failed: {index_check.warnings}")

    return True
