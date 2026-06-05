# -*- coding: utf-8 -*-
"""
token_logger.py — Per-query LLM token usage logger.

Tracks input/output tokens for every LLM call in the UPSC RAG pipeline,
writes structured JSONL records, and produces a per-query cost summary.

Log file: TOKEN_LOG_DIR/token_usage.jsonl  (one JSON object per line)

Record types
────────────
  "call"          — one LLM invocation (stage + model + in/out tokens)
  "query_summary" — aggregated view across all stages for one user query

Usage
─────
    from futureverse_ai.retrieval.token_logger import log_call, log_query_summary, new_query_id

    qid = new_query_id()

    # After each LLM call:
    log_call(qid, stage="query_expansion",     model="qwen2.5:7b",  input_tokens=120, output_tokens=45,  query_text=query)
    log_call(qid, stage="answer_generation",   model="qwen2.5:32b", input_tokens=980, output_tokens=310, query_text=query)

    # Once per query (after all stages complete):
    log_query_summary(qid, query_text=query,
                      query_expansion_in=120,  query_expansion_out=45,  expansion_model="qwen2.5:7b",
                      answer_in=980,           answer_out=310,           answer_model="qwen2.5:32b")
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# ── Log file location ─────────────────────────────────────────────────────────
TOKEN_LOG_DIR  = os.environ.get("TOKEN_LOG_DIR", r"D:\FutureverseAI\logs")
TOKEN_LOG_FILE = os.path.join(TOKEN_LOG_DIR, "token_usage.jsonl")

# ── Cost table (USD per 1 million tokens) ─────────────────────────────────────
# Local Ollama models = $0 (capacity planning only).
# Add cloud model rates here if you switch providers.
_COST_PER_1M: dict[str, dict[str, float]] = {
    # ── Local Ollama ──────────────────────────────────────────────────────────
    "qwen2.5:7b":     {"input": 0.0, "output": 0.0},
    "qwen2.5:32b":    {"input": 0.0, "output": 0.0},
    "llama3.1:8b":    {"input": 0.0, "output": 0.0},
    # ── Cloud models (uncomment if you switch) ────────────────────────────────
    # "gpt-4o":                   {"input": 2.50,  "output": 10.00},
    # "gpt-4o-mini":              {"input": 0.15,  "output": 0.60},
    # "claude-sonnet-4-6":        {"input": 3.00,  "output": 15.00},
    # "claude-haiku-4-5":         {"input": 0.80,  "output": 4.00},
    # "gemini-1.5-flash":         {"input": 0.075, "output": 0.30},
}
_DEFAULT_COST = {"input": 0.0, "output": 0.0}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_log_dir() -> None:
    Path(TOKEN_LOG_DIR).mkdir(parents=True, exist_ok=True)


def _append(record: dict) -> None:
    """Atomically append one JSON record to the JSONL log file."""
    _ensure_log_dir()
    try:
        with open(TOKEN_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.warning(f"[TokenLogger] write failed: {exc}")


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _COST_PER_1M.get(model, _DEFAULT_COST)
    return (
        (input_tokens  / 1_000_000) * rates["input"] +
        (output_tokens / 1_000_000) * rates["output"]
    )


# ── Public API ────────────────────────────────────────────────────────────────

def new_query_id() -> str:
    """Generate a short unique ID for one user query session."""
    return str(uuid.uuid4())[:8]


def log_call(
    query_id:     str,
    stage:        str,   # "query_expansion" | "answer_generation"
    model:        str,
    input_tokens: int,
    output_tokens: int,
    query_text:   str = "",
) -> dict:
    """
    Log a single LLM call and return the written record.

    Args:
        query_id:      Unique ID for the parent user query (from new_query_id()).
        stage:         Pipeline stage — "query_expansion" or "answer_generation".
        model:         Ollama model name (e.g. "qwen2.5:7b").
        input_tokens:  Prompt tokens (prompt_eval_count from Ollama response).
        output_tokens: Completion tokens (eval_count from Ollama response).
        query_text:    Original user query (stored truncated to 200 chars).

    Returns:
        The dict record that was written to the log file.
    """
    total     = input_tokens + output_tokens
    cost_usd  = _compute_cost(model, input_tokens, output_tokens)

    record = {
        "type":          "call",
        "query_id":      query_id,
        "stage":         stage,
        "model":         model,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  total,
        "cost_usd":      round(cost_usd, 8),
        "query_text":    (query_text or "")[:200],
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }
    _append(record)
    log.info(
        f"[TokenUsage] id={query_id} stage={stage} model={model} "
        f"in={input_tokens} out={output_tokens} total={total}"
    )
    return record


def log_query_summary(
    query_id:             str,
    query_text:           str,
    query_expansion_in:   int,
    query_expansion_out:  int,
    answer_in:            int,
    answer_out:           int,
    expansion_model:      str = "",
    answer_model:         str = "",
) -> dict:
    """
    Write an aggregated summary record for one complete user query.

    Call this once after both LLM stages complete (query expansion + answer).

    Returns:
        The summary dict written to the log file.
    """
    exp_total   = query_expansion_in  + query_expansion_out
    ans_total   = answer_in           + answer_out
    grand_total = exp_total           + ans_total

    cost_usd = (
        _compute_cost(expansion_model, query_expansion_in,  query_expansion_out) +
        _compute_cost(answer_model,    answer_in,           answer_out)
    )

    record = {
        "type":                   "query_summary",
        "query_id":               query_id,
        "query_text":             (query_text or "")[:300],

        # ── Stage 1: query expansion ──────────────────────────────────────────
        "expansion_model":        expansion_model,
        "query_expansion_input":  query_expansion_in,
        "query_expansion_output": query_expansion_out,
        "query_expansion_total":  exp_total,

        # ── Stage 2: answer generation ────────────────────────────────────────
        "answer_model":           answer_model,
        "answer_input":           answer_in,
        "answer_output":          answer_out,
        "answer_total":           ans_total,

        # ── Totals ────────────────────────────────────────────────────────────
        "grand_total_tokens":     grand_total,
        "estimated_cost_usd":     round(cost_usd, 8),

        "timestamp":              datetime.now(timezone.utc).isoformat(),
    }
    _append(record)
    log.info(
        f"[TokenSummary] id={query_id} "
        f"expansion={exp_total} answer={ans_total} "
        f"grand_total={grand_total} cost_usd={cost_usd:.8f}"
    )
    return record


def get_recent_summaries(n: int = 20) -> list[dict]:
    """
    Read the last ``n`` query_summary records from the log file.
    Useful for building a usage dashboard or debugging costs.
    """
    _ensure_log_dir()
    if not os.path.exists(TOKEN_LOG_FILE):
        return []
    records = []
    try:
        with open(TOKEN_LOG_FILE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("type") == "query_summary":
                        records.append(rec)
                except json.JSONDecodeError:
                    pass
    except Exception as exc:
        log.warning(f"[TokenLogger] read failed: {exc}")
    return records[-n:]
