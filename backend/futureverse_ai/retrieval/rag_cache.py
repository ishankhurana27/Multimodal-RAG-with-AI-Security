"""
Generic Redis-backed RAG caches (result / knowledge-graph / reranker layers).

Modeled on embedding_cache.py:
- Every layer is behind its own env flag, default OFF — with all flags off,
  behavior is byte-for-byte unchanged (no Redis connection is even opened).
- Fail-open: any cache/Redis error is treated as a miss (compute normally);
  a cache must never break a request.
- Keys include the relevant model name so a model change auto-invalidates.

Key format: rag:{namespace}:{sha256("|".join(map(str, key_parts)))}
Namespaces: result (final retrieval sources), kg (Neo4j graph context),
rerank (cross-encoder scores), kbidx (per-collection index sets used to
purge result keys when a knowledge base changes).
"""
import hashlib
import json
import logging
import os

log = logging.getLogger(__name__)


def _flag(name):
    return os.environ.get(name, "false").lower() == "true"


RESULT_CACHE_ENABLED = _flag("ENABLE_RAG_RESULT_CACHE")
KG_CACHE_ENABLED = _flag("ENABLE_RAG_KG_CACHE")
RERANK_CACHE_ENABLED = _flag("ENABLE_RAG_RERANK_CACHE")

RESULT_CACHE_TTL = int(os.environ.get("RAG_RESULT_CACHE_TTL", "900"))
KG_CACHE_TTL = int(os.environ.get("RAG_KG_CACHE_TTL", "900"))
RERANK_CACHE_TTL = int(os.environ.get("RAG_RERANK_CACHE_TTL", "86400"))

EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "")
RERANKING_MODEL = os.environ.get("RAG_RERANKING_MODEL", "")

_REDIS_URL = os.environ.get("REDIS_URL", "")
_ANY_ENABLED = RESULT_CACHE_ENABLED or KG_CACHE_ENABLED or RERANK_CACHE_ENABLED

_client = None
_init_done = False


def _redis():
    """Lazy singleton Redis client (same pattern as embedding_cache._redis).

    Returns None when no cache flag is on or Redis is unreachable, so every
    helper below degrades to a no-op / miss.
    """
    global _client, _init_done
    if _init_done:
        return _client
    _init_done = True
    if not _ANY_ENABLED:
        return None
    try:
        import redis
        _client = redis.Redis.from_url(_REDIS_URL) if _REDIS_URL else redis.Redis()
        _client.ping()
        log.info("RAG cache: connected to Redis")
    except Exception as e:
        log.warning("RAG cache disabled — Redis unavailable: %s", e)
        _client = None
    return _client


def make_key(namespace, key_parts):
    h = hashlib.sha256("|".join(map(str, key_parts)).encode("utf-8")).hexdigest()
    return f"rag:{namespace}:{h}"


def cache_get(namespace, key_parts):
    """JSON-decoded value on hit; fail-open -> None on miss or any error."""
    c = _redis()
    if not c:
        return None
    try:
        v = c.get(make_key(namespace, key_parts))
        if v is not None:
            return json.loads(v)
    except Exception as e:
        log.debug("RAG cache get failed (fail-open): %s", e)
    return None


def cache_set(namespace, key_parts, value, ttl):
    """JSON-encode with setex. Fail-open -> no-op on any error."""
    c = _redis()
    if not c:
        return
    try:
        c.setex(make_key(namespace, key_parts), int(ttl), json.dumps(value))
    except Exception as e:
        log.debug("RAG cache set failed (fail-open): %s", e)


def cache_get_many(namespace, key_parts_list):
    """MGET variant: list aligned with input, None per miss. Fail-open -> all None."""
    misses = [None] * len(key_parts_list)
    c = _redis()
    if not c or not key_parts_list:
        return misses
    try:
        vals = c.mget([make_key(namespace, kp) for kp in key_parts_list])
        out = []
        for v in vals:
            if v is None:
                out.append(None)
            else:
                try:
                    out.append(json.loads(v))
                except Exception:
                    out.append(None)
        return out
    except Exception as e:
        log.debug("RAG cache mget failed (fail-open): %s", e)
        return misses


def cache_set_many(namespace, items, ttl):
    """Pipelined setex for [(key_parts, value), ...]. Fail-open -> no-op."""
    c = _redis()
    if not c or not items:
        return
    try:
        pipe = c.pipeline()
        for key_parts, value in items:
            pipe.setex(make_key(namespace, key_parts), int(ttl), json.dumps(value))
        pipe.execute()
    except Exception as e:
        log.debug("RAG cache mset failed (fail-open): %s", e)


def kbidx_register(collection_names, cache_key, ttl):
    """Record a result-cache key under rag:kbidx:{collection} for each collection,
    so purge_collections() can invalidate it when that knowledge base changes.
    The index set outlives the entry TTL slightly; stale members DEL to nothing."""
    c = _redis()
    if not c or not collection_names:
        return
    try:
        pipe = c.pipeline()
        for col in collection_names:
            skey = f"rag:kbidx:{col}"
            pipe.sadd(skey, cache_key)
            pipe.expire(skey, max(int(ttl) * 4, 3600))
        pipe.execute()
    except Exception as e:
        log.debug("RAG cache kbidx register failed (fail-open): %s", e)


def purge_collections(collection_names):
    """Delete every result-cache key recorded for these collections, then the
    index sets themselves. Called after a knowledge-base mutation. Fail-open."""
    c = _redis()
    if not c or not collection_names:
        return
    try:
        for col in collection_names:
            skey = f"rag:kbidx:{col}"
            members = c.smembers(skey)
            pipe = c.pipeline()
            for m in members:
                pipe.delete(m)
            pipe.delete(skey)
            pipe.execute()
            if members:
                log.debug(
                    "RAG result cache: purged %d entries for collection %s",
                    len(members), col,
                )
    except Exception as e:
        log.debug("RAG cache purge failed (fail-open): %s", e)


def cache_purge_namespace(pattern):
    """SCAN + DEL by key pattern (e.g. 'rag:result:*'). Fail-open -> no-op."""
    c = _redis()
    if not c:
        return
    try:
        cursor = 0
        while True:
            cursor, keys = c.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                c.delete(*keys)
            if cursor == 0:
                break
    except Exception as e:
        log.debug("RAG cache namespace purge failed (fail-open): %s", e)
