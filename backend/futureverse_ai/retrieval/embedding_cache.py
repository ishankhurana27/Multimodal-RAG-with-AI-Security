"""
Cross-request query-embedding cache (Phase 3, \u00a78).

Default OFF (ENABLE_RAG_EMBEDDING_CACHE=false) \u2014 zero behaviour change until enabled.
Fail-open: any Redis error is treated as a cache miss, so retrieval never breaks.
Keyed by the embedding model (read from env) so a model change auto-invalidates;
if you change the model via the UI/DB without changing RAG_EMBEDDING_MODEL, flush
the `rag:emb:*` keys.
"""
import hashlib
import json
import logging
import os

log = logging.getLogger(__name__)

CACHE_ENABLED = os.environ.get("ENABLE_RAG_EMBEDDING_CACHE", "false").lower() == "true"
_TTL = int(os.environ.get("RAG_EMBEDDING_CACHE_TTL", "86400"))  # 24h
_REDIS_URL = os.environ.get("REDIS_URL", "")
_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "")

_client = None
_init_done = False


def _redis():
    global _client, _init_done
    if _init_done:
        return _client
    _init_done = True
    if not CACHE_ENABLED:
        return None
    try:
        import redis
        _client = redis.Redis.from_url(_REDIS_URL) if _REDIS_URL else redis.Redis()
        _client.ping()
        log.info("RAG embedding cache: connected to Redis")
    except Exception as e:
        log.warning("RAG embedding cache disabled \u2014 Redis unavailable: %s", e)
        _client = None
    return _client


def _key(query):
    h = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return f"rag:emb:{_MODEL}:{h}"


def get_cached(queries):
    """Return {query: vector} for cache hits. Fail-open -> {} on any error."""
    c = _redis()
    if not c:
        return {}
    out = {}
    try:
        vals = c.mget([_key(q) for q in queries])
        for q, v in zip(queries, vals):
            if v:
                try:
                    out[q] = json.loads(v)
                except Exception:
                    pass
    except Exception as e:
        log.debug("embedding cache get failed (fail-open): %s", e)
    return out


def set_cached(mapping):
    """Store {query: vector}. Fail-open -> no-op on any error."""
    c = _redis()
    if not c or not mapping:
        return
    try:
        pipe = c.pipeline()
        for q, vec in mapping.items():
            pipe.setex(_key(q), _TTL, json.dumps(vec))
        pipe.execute()
    except Exception as e:
        log.debug("embedding cache set failed (fail-open): %s", e)
