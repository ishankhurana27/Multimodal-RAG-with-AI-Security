"""
Contextual chunking — Layer 3 of the SHC-CE pipeline.

For each post-split chunk, emits a structured context prefix:

    [CONTEXT]
    Document: {file title}
    Chapter:  {chapter or H1}
    Section:  {section or H2}
    Position: paragraph {n} of {total}
    Topic:    {LLM-generated 1-2 sentence topic summary}
    Preceding idea: {first sentence of the preceding chunk}

    [CONTENT]
    {actual chunk text}

What this solves:
  - Orphan pronouns resolved via Document/Chapter/Section context
  - Embeddings capture document position, not just local meaning
  - LLM at generation time has rich, structured grounding
  - "Preceding idea" repairs topic-split boundaries

Cost:  one Ollama call per chunk (topic summary only).
       Preceding idea is extracted from the preceding chunk's raw text —
       zero LLM cost.

Control env vars:
  OLLAMA_BASE_URL            — default: http://localhost:11434
  CONTEXTUAL_CHUNKING_MODEL  — default: qwen2.5:7b
"""

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from langchain_core.documents import Document

log = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_MODEL      = os.environ.get("CONTEXTUAL_CHUNKING_MODEL", "qwen2.5:7b")
_CHUNK_TIMEOUT      = 60   # covers Vulkan cold-load on NVIDIA; CUDA would be ~3-8s/chunk
_DOC_TIMEOUT        = 120   # GPU Vulkan: summary of ~3k chars takes <15s; 120s covers cold model load
_MAX_WORKERS        = 3
_MAX_DOC_CHARS      = 3000
_MAX_CHUNK_CHARS    = 500
_LOG_EVERY          = 5

# Issue 7: module-level fallback counter — surfaces silent quality degradation.
# Incremented each time a chunk falls back to raw text (no [CONTEXT] prefix).
# Access with: from futureverse_ai.retrieval.contextual_chunking import FALLBACK_COUNTER
# Value: FALLBACK_COUNTER[0]  (list for thread-safe mutation under CPython GIL)
FALLBACK_COUNTER: list[int] = [0]

# Issue 7: circuit breaker — prevents a 5× latency storm when Ollama is down.
# After _CIRCUIT_OPEN_THRESHOLD consecutive failures across ALL threads, _call_ollama
# returns None immediately without retrying. Resets on the next successful call.
# Threshold configurable here without touching the calling code.
_CONSECUTIVE_FAILURES: list[int]  = [0]
_CIRCUIT_OPEN_THRESHOLD: int      = 5


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bar(done: int, total: int, width: int = 28) -> str:
    pct    = done / total if total else 0
    filled = int(width * pct)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {done:>3}/{total} ({pct:5.1%})"


def _call_ollama(
    prompt: str,
    url: str,
    model: str,
    timeout: int,
    retries: int = 2,
) -> Optional[str]:
    """Call Ollama with exponential backoff and circuit breaker.

    retries=2 → up to 3 total attempts (backoff: 1s, 2s between attempts).

    Circuit breaker: after _CIRCUIT_OPEN_THRESHOLD consecutive failures across
    ALL threads, subsequent calls return None immediately without retrying.
    This prevents a 5× latency storm (3 workers × 3s retry time per chunk) when
    Ollama is genuinely down during a large ingestion. Resets on any success.
    """
    if _CONSECUTIVE_FAILURES[0] >= _CIRCUIT_OPEN_THRESHOLD:
        log.debug(
            "Ollama circuit open (%d consecutive failures ≥ threshold %d) — skipping",
            _CONSECUTIVE_FAILURES[0], _CIRCUIT_OPEN_THRESHOLD,
        )
        return None

    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip() or None
            _CONSECUTIVE_FAILURES[0] = 0  # reset circuit on success
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                wait_s = 2 ** attempt   # 1s after attempt 1, 2s after attempt 2
                log.debug(
                    "Ollama attempt %d/%d failed (%s), retrying in %ds",
                    attempt + 1, retries + 1, exc, wait_s,
                )
                time.sleep(wait_s)

    _CONSECUTIVE_FAILURES[0] += 1
    log.debug(
        "Ollama call failed after %d attempts: %s (consecutive_failures=%d)",
        retries + 1, last_exc, _CONSECUTIVE_FAILURES[0],
    )
    return None


def _first_sentence(text: str) -> str:
    """Extract first sentence from text (no LLM cost)."""
    text = text.strip()
    if not text:
        return ""
    m = re.search(r'(.+?[.!?])\s', text)
    if m:
        return m.group(1).strip()
    return text[:150].strip()


def _doc_summary_prompt(full_doc: str) -> str:
    return (
        f"<document>\n{full_doc[:_MAX_DOC_CHARS]}\n</document>\n"
        "In 2-3 sentences, describe what this document is about, its subject area, "
        "and key themes. Be concise."
    )


def _topic_prompt(full_doc: str, chunk_text: str, breadcrumb: str, position: int, total: int) -> str:
    """Prompt that generates just the Topic field (1-2 sentences)."""
    location = f"\nDocument location: {breadcrumb}" if breadcrumb else ""
    pos_info  = f"\nChunk position: paragraph {position} of {total}"
    return (
        f"<document>\n{full_doc[:_MAX_DOC_CHARS]}\n</document>\n"
        f"{location}{pos_info}\n"
        "Here is the chunk:\n"
        f"<chunk>\n{chunk_text[:_MAX_CHUNK_CHARS]}\n</chunk>\n"
        "In 1-2 sentences, describe what specific topic this paragraph covers "
        "within the document. Be specific about the subject matter. "
        "Answer only with the topic description and nothing else."
    )


def _build_context_block(
    doc: Document,
    position: int,
    total: int,
    topic: str,
    preceding_text: str,
) -> str:
    """
    Build the structured [CONTEXT] block from metadata + LLM topic.

    Uses breadcrumb, header metadata, and NCERT-specific fields to populate
    Document / Chapter / Section lines. Preceding idea is taken from the
    first sentence of the previous chunk — zero additional LLM cost.
    """
    meta = doc.metadata

    # ── Extract hierarchy labels ───────────────────────────────────────────────
    breadcrumb = meta.get("breadcrumb", "")
    crumbs     = [c.strip() for c in breadcrumb.split(" > ")] if breadcrumb else []

    # Document title: prefer file name, then H1, then first crumb
    doc_title = (
        meta.get("name")
        or meta.get("title")
        or meta.get("Header 1")
        or (crumbs[0] if crumbs else "")
    )

    # Chapter: NCERT chapter field, then H1, then second crumb
    chapter = (
        meta.get("chapter")
        or meta.get("Header 1")
        or (crumbs[1] if len(crumbs) > 1 else "")
    )

    # Section: H2/H3, then last crumb
    section = (
        meta.get("Header 2")
        or meta.get("Header 3")
        or (crumbs[-1] if len(crumbs) > 1 else "")
    )

    # Preceding idea: first sentence of previous chunk (free — no LLM call)
    preceding_idea = _first_sentence(preceding_text) if preceding_text else ""

    # ── Assemble block ─────────────────────────────────────────────────────────
    lines = ["[CONTEXT]"]
    if doc_title:     lines.append(f"Document: {doc_title}")
    if chapter:       lines.append(f"Chapter: {chapter}")
    if section and section != chapter:
                      lines.append(f"Section: {section}")
    lines.append(f"Position: paragraph {position} of {total}")
    if topic:         lines.append(f"Topic: {topic}")
    if preceding_idea:lines.append(f"Preceding idea: {preceding_idea}")
    lines.append("")
    lines.append("[CONTENT]")
    return "\n".join(lines)


# ── Per-chunk worker ───────────────────────────────────────────────────────────

def _enrich_chunk(
    idx:            int,
    doc:            Document,
    full_doc:       str,
    url:            str,
    model:          str,
    position:       int,
    total:          int,
    preceding_text: str,
) -> tuple:
    """
    Enrich one chunk. Returns (idx, Document, elapsed_s, success).

    LLM generates the Topic field only.
    Preceding idea is extracted from preceding_text without an LLM call.
    Falls back to raw text on any failure.
    """
    t0       = time.monotonic()
    raw_text = doc.page_content

    # For hierarchical/paragraph chunks: use parent_text as the context window
    # (more targeted than the full document)
    context_window = (
        doc.metadata.get("grandparent_text")
        or doc.metadata.get("parent_text")
        or full_doc
    )
    breadcrumb = doc.metadata.get("breadcrumb", "")

    try:
        topic = _call_ollama(
            _topic_prompt(context_window, raw_text, breadcrumb, position, total),
            url, model, _CHUNK_TIMEOUT,
        )
        elapsed = time.monotonic() - t0

        if topic:
            context_block = _build_context_block(doc, position, total, topic, preceding_text)
            embedded_text = f"{context_block}\n{raw_text}"
            return (
                idx,
                Document(
                    page_content=embedded_text,
                    metadata={
                        **doc.metadata,
                        "raw_text":        raw_text,
                        "context_summary": context_block,
                        "has_context":     True,
                    },
                ),
                elapsed,
                True,
            )
    except Exception as exc:
        log.warning("chunk %d context failed: %s", idx + 1, exc)

    # Issue 7: count every fallback so silent quality degradation is visible.
    FALLBACK_COUNTER[0] += 1
    elapsed = time.monotonic() - t0
    return (
        idx,
        Document(
            page_content=raw_text,
            metadata={
                **doc.metadata,
                "raw_text":        raw_text,
                "context_summary": "",
                "has_context":     False,
            },
        ),
        elapsed,
        False,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def apply_contextual_chunking(
    docs:       list,
    full_doc_text: str,
    ollama_url: str = _DEFAULT_OLLAMA_URL,
    model:      str = _DEFAULT_MODEL,
) -> list:
    """
    Enrich post-split chunks with the structured [CONTEXT]/[CONTENT] format.

    Computes preceding_text for every chunk before parallel execution so
    workers don't depend on each other's results.

    CONSTRAINT — preceding_texts must be precomputed sequentially before workers
    start. Do NOT move preceding_text computation into the worker threads: workers
    run out-of-order (ThreadPoolExecutor does not guarantee FIFO), so a worker for
    chunk N could read enriched page_content from chunk N-1 before N-1 is done,
    producing a stale or empty preceding_text. The current design (lines below)
    snapshots raw page_content before parallelism. Any future change that derives
    preceding_text from enriched text MUST restructure to a sequential second pass.

    Issue 7 — FALLBACK_COUNTER[0] is incremented for every chunk that falls back
    to raw text (no [CONTEXT] prefix). Monitor it to detect silent quality degradation
    when Ollama is overloaded or OLLAMA_NUM_PARALLEL is misconfigured.
    """
    total   = len(docs)
    wall_t0 = time.monotonic()

    if total == 0:
        return docs

    # Issue 7: warn if OLLAMA_NUM_PARALLEL=1 while using multiple workers.
    # A single-parallel Ollama instance queues requests serially — workers
    # don't speed things up and may accumulate timeouts under load.
    _ollama_parallel = int(os.environ.get("OLLAMA_NUM_PARALLEL", "1"))
    if _MAX_WORKERS > 1 and _ollama_parallel <= 1:
        _warn_msg = (
            f"OLLAMA_NUM_PARALLEL={_ollama_parallel} but contextual chunking "
            f"uses {_MAX_WORKERS} workers. Requests queue serially. "
            f"Set OLLAMA_NUM_PARALLEL={_MAX_WORKERS} in Ollama's environment "
            "for true parallelism."
        )
        log.warning(_warn_msg)
        print(f"  ⚠  {_warn_msg}", flush=True)

    # ── Pre-compute preceding texts (raw, no LLM) ─────────────────────────────
    preceding_texts: list[str] = [""]  # chunk 0 has no predecessor
    for i in range(1, total):
        prev_raw = docs[i - 1].metadata.get("raw_text") or docs[i - 1].page_content
        preceding_texts.append(prev_raw)

    # ── Header ────────────────────────────────────────────────────────────────
    print("", flush=True)
    print("╔" + "═" * 60 + "╗", flush=True)
    print("║  CONTEXTUAL CHUNKING  START" + " " * 32 + "║", flush=True)
    print(f"║  Chunks  : {total:<49}║", flush=True)
    print(f"║  Model   : {model} @ {ollama_url}"[:62].ljust(62) + "║", flush=True)
    print(f"║  Workers : {_MAX_WORKERS}   timeout/chunk: {_CHUNK_TIMEOUT}s" + " " * 36 + "║", flush=True)
    print(f"║  Format  : [CONTEXT] Document/Chapter/Section/Position/Topic/Preceding" + " " * 0 + "║", flush=True)
    print("╚" + "═" * 60 + "╝", flush=True)

    # ── Step 1: document summary (for logging only — context uses parent_text) ─
    print(f"\n  Step 1/2  Generating document summary ...", flush=True)
    t1          = time.monotonic()
    doc_summary = _call_ollama(_doc_summary_prompt(full_doc_text), ollama_url, model, _DOC_TIMEOUT, retries=0)
    summary_ms  = (time.monotonic() - t1) * 1000

    if doc_summary:
        print(f"  ✓ Summary  ({summary_ms:.0f}ms, {len(doc_summary)} chars)", flush=True)
        print(f"    \"{doc_summary[:120].replace(chr(10),' ')}...\"", flush=True)
    else:
        print(f"  ✗ Summary  FAILED ({summary_ms:.0f}ms)", flush=True)

    # ── Step 2: per-chunk enrichment ──────────────────────────────────────────
    print(f"\n  Step 2/2  Enriching {total} chunks (workers={_MAX_WORKERS}) ...", flush=True)

    enriched:       list  = [None] * total
    elapsed_by_idx: dict  = {}
    done_count  = 0
    ok_count    = 0
    fail_count  = 0

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        future_map = {
            executor.submit(
                _enrich_chunk,
                idx, doc, full_doc_text, ollama_url, model,
                idx + 1,            # 1-based position
                total,
                preceding_texts[idx],
            ): idx
            for idx, doc in enumerate(docs)
        }

        for future in as_completed(future_map):
            orig_idx = future_map[future]
            try:
                idx, enriched_doc, elapsed, success = future.result(timeout=_CHUNK_TIMEOUT + 5)
            except Exception as exc:
                idx          = orig_idx
                elapsed      = float(_CHUNK_TIMEOUT + 5)
                success      = False
                raw_text     = docs[orig_idx].page_content
                enriched_doc = Document(
                    page_content=raw_text,
                    metadata={
                        **docs[orig_idx].metadata,
                        "raw_text":        raw_text,
                        "context_summary": "",
                        "has_context":     False,
                    },
                )
                print(f"  ✗ chunk {orig_idx + 1:>3}  TIMEOUT/ERROR: {exc}", flush=True)

            enriched[idx]       = enriched_doc
            elapsed_by_idx[idx] = elapsed
            done_count         += 1
            ok_count           += success
            fail_count         += (not success)

            avg_ms = (sum(elapsed_by_idx.values()) / len(elapsed_by_idx)) * 1000
            eta_s  = (avg_ms / 1000) * (total - done_count) / _MAX_WORKERS

            if done_count % _LOG_EVERY == 0 or done_count == total:
                bar = _bar(done_count, total)
                print(
                    f"  {bar}  ok={ok_count:>3}  fail={fail_count:>2}"
                    f"  avg={avg_ms:.0f}ms  ETA~{max(eta_s,0):.0f}s",
                    flush=True,
                )

    # ── Safety net ────────────────────────────────────────────────────────────
    for i, doc in enumerate(enriched):
        if doc is None:
            raw_text = docs[i].page_content
            enriched[i] = Document(
                page_content=raw_text,
                metadata={
                    **docs[i].metadata,
                    "raw_text":        raw_text,
                    "context_summary": "",
                    "has_context":     False,
                },
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    wall_elapsed = time.monotonic() - wall_t0
    avg_chunk_ms = (sum(elapsed_by_idx.values()) / len(elapsed_by_idx) * 1000) if elapsed_by_idx else 0
    pct_ok       = (ok_count / total * 100) if total else 0

    print("", flush=True)
    print("  " + "─" * 58, flush=True)
    print(f"  CONTEXTUAL CHUNKING  COMPLETE", flush=True)
    print(f"  Total    : {total} chunks", flush=True)
    print(f"  Enriched : {ok_count}  ({pct_ok:.1f}%)", flush=True)
    print(f"  Fallback : {fail_count}  ({100-pct_ok:.1f}%)", flush=True)
    print(f"  Avg/chunk: {avg_chunk_ms:.0f} ms", flush=True)
    print(f"  Wall time: {wall_elapsed:.1f} s", flush=True)
    print("  " + "─" * 58, flush=True)

    log.info("Contextual chunking — %d/%d enriched  (%.1fs)", ok_count, total, wall_elapsed)

    # Per-chunk detail to log
    log.info("  IDX  STATUS  TIME(ms)  CONTEXT PREVIEW")
    for i, d in enumerate(enriched):
        status   = "OK  " if d.metadata.get("has_context") else "FAIL"
        ctx      = d.metadata.get("context_summary", "")
        preview  = (ctx[:80] if ctx else d.page_content[:80]).replace("\n", " ")
        chunk_ms = elapsed_by_idx.get(i, 0) * 1000
        log.info("  %3d  %s  %7.0fms  %s", i + 1, status, chunk_ms, preview)

    print("", flush=True)
    print(f"  {'IDX':>3}  {'STATUS':<6}  {'ms':>7}  CONTEXT PREVIEW", flush=True)
    print("  " + "-" * 56, flush=True)
    for i, d in enumerate(enriched):
        status   = "✓ OK  " if d.metadata.get("has_context") else "✗ FAIL"
        ctx      = d.metadata.get("context_summary", "")
        preview  = (ctx[:60] if ctx else d.page_content[:60]).replace("\n", " ")
        chunk_ms = elapsed_by_idx.get(i, 0) * 1000
        print(f"  {i+1:>3}  {status}  {chunk_ms:>7.0f}ms  {preview}", flush=True)

    print("", flush=True)
    return enriched
