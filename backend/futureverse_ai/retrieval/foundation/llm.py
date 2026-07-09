"""
Ollama client for the two-pass tagging stage.

Mirrors the proven call pattern in contextual_chunking.py (requests → /api/generate,
exponential backoff, circuit breaker) but is tuned for deterministic tagging:
temperature 0 and a high token budget for proposition lists.

Model: qwen2.5:7b by default (FOUNDATION_TAGGING_MODEL env to override).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
TAGGING_MODEL   = os.environ.get("FOUNDATION_TAGGING_MODEL", "qwen2.5:7b")
_TIMEOUT        = int(os.environ.get("FOUNDATION_TAGGING_TIMEOUT", "120"))

_CONSECUTIVE_FAILURES: list[int] = [0]
_CIRCUIT_OPEN_THRESHOLD: int     = 5


def call_llm(
    prompt: str,
    *,
    model: str = "",
    url: str = "",
    timeout: int = 0,
    retries: int = 2,
    num_predict: int = 800,
) -> Optional[str]:
    """
    Send a prompt to Ollama and return the response text (or None on failure).

    Deterministic (temperature 0) so tagging output is stable across re-runs.
    Returns None — never raises — so the caller can fall back gracefully.
    """
    model = model or TAGGING_MODEL
    url = (url or OLLAMA_BASE_URL).rstrip("/")
    timeout = timeout or _TIMEOUT

    if _CONSECUTIVE_FAILURES[0] >= _CIRCUIT_OPEN_THRESHOLD:
        log.warning("Ollama circuit open (%d failures) — skipping tagging call",
                    _CONSECUTIVE_FAILURES[0])
        return None

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": num_predict},
    }

    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(f"{url}/api/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            text = resp.json().get("response", "").strip() or None
            _CONSECUTIVE_FAILURES[0] = 0
            return text
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(2 ** attempt)   # 1s, 2s

    _CONSECUTIVE_FAILURES[0] += 1
    log.warning("Ollama tagging call failed after %d attempts: %s", retries + 1, last_exc)
    return None


def is_available(model: str = "") -> bool:
    """Quick health/model-presence probe used by the CLI runner."""
    model = model or TAGGING_MODEL
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        return any(n == model or n.startswith(model.split(":")[0]) for n in names)
    except Exception:
        return False
