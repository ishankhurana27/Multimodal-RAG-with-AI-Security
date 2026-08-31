"""
repair_vectors.py — find Weaviate collections whose objects have no vectors
and backfill the missing embeddings in place.

    # diagnose only (default) — writes nothing
    python qgen\\scripts\\repair_vectors.py

    # one collection, still read-only
    python qgen\\scripts\\repair_vectors.py --knowledge-id 30fcb31b-5e7a-4de2-b134-e676f2e0678a

    # do the repair, then verify a dense query comes back
    python qgen\\scripts\\repair_vectors.py --knowledge-id <kb-id> --apply --verify

WHY THIS EXISTS
───────────────
A Weaviate object inserted without a vector is still fully searchable by
BM25 and completely invisible to `nearVector`. Nothing errors — the dense
channel just returns nothing for that collection, so hybrid retrieval
silently degrades to keyword-only and every consumer (chat RAG, question
generation, the PYQ chunk extractor) quietly loses half its engine on that
corpus. Detected in this project on the knowledge base holding the full
Laxmikant book: 1427 objects, 1897 BM25 hits across a 240-query run, zero
dense hits.

WHAT IT DOES
────────────
Re-embeds ONLY the objects that lack a vector and patches the vector in
place, leaving every property untouched — no re-chunking, no re-running the
LLM keyword extraction or the [CONTEXT] enrichment, and chunk_ids, keywords
and breadcrumbs all survive exactly as they are. That makes it far cheaper
and far safer than re-ingesting the files.

EMBEDDING FIDELITY
──────────────────
The vector must be produced the same way ingestion produced it, or the
repaired objects sit in a different corner of the space than everything
else. Reading the pipeline (pipeline_agents.run_ingestion_pipeline):

    texts = [sanitize_text_for_db(doc.page_content) for doc in docs]
    embed_agent.run(texts, embedding_function, RAG_EMBEDDING_CONTENT_PREFIX, …)
    items = [{"id": chunk_id, "text": texts[i], "vector": embeddings[i], …}]

so the embedded string is exactly the stored `text` property — page_content
WITH its [CONTEXT] prefix, already sanitised — and EmbeddingAgent replaces
newlines with spaces before encoding. On the sentence-transformers engine
the prefix argument is ignored by `async_st_embedding`, and encoding uses
normalize_embeddings=True. This script reproduces that chain exactly: read
the stored `text`, newlines to spaces, same model, normalised, no prefix.
`--verify` then proves the repair by running a real dense query.

Stdlib + the backend's own vector client and sentence-transformers.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

SCRIPT_VERSION = "repair-vectors-1"


def find_repo_root(start: Path) -> Path:
    for cand in [start] + list(start.parents):
        if (cand / "backend" / "futureverse_ai").is_dir():
            return cand
    return start


ROOT = find_repo_root(Path(__file__).resolve().parent)


def load_env(root: Path) -> dict:
    """Read .env for model/Weaviate settings. Values are used, never logged."""
    env = {}
    path = root / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return env


def sanitize_collection_name(name: str) -> str:
    """Mirror of the Weaviate wrapper's own sanitiser, so a knowledge-base
    UUID maps to the same class name the wrapper would use."""
    col = re.sub(r"[^a-zA-Z0-9_]", "", str(name).replace("-", "_")).strip("_")
    if not col:
        return col
    return ("C" + col) if not col[0].isalpha() else (col[0].upper() + col[1:])


# ══════════════════════════════════════════════════════════════════════════
# inspection
# ══════════════════════════════════════════════════════════════════════════

def iter_objects(client, cname: str, with_vectors: bool = True):
    """Yield (uuid, properties, vector_or_None) for every object."""
    coll = client.client.collections.get(cname)
    for obj in coll.iterator(include_vector=with_vectors):
        vec = getattr(obj, "vector", None)
        if isinstance(vec, dict):          # named vectors: {"default": [...]}
            vec = (vec.get("default") if "default" in vec
                   else (next(iter(vec.values())) if vec else None))
        yield str(obj.uuid), (dict(obj.properties) if obj.properties else {}), vec


def inspect(client, cname: str, sample_limit: int = 0) -> dict:
    total = with_vec = 0
    dims = {}
    missing_ids = []
    empty_text = 0
    sample_text = ""
    for uid, props, vec in iter_objects(client, cname):
        total += 1
        text = (props.get("text") or "").strip()
        if not text:
            empty_text += 1
        elif not sample_text:
            sample_text = text[:160]
        if vec:
            with_vec += 1
            dims[len(vec)] = dims.get(len(vec), 0) + 1
        else:
            if not sample_limit or len(missing_ids) < sample_limit:
                missing_ids.append(uid)
    return {"collection": cname, "objects": total, "with_vector": with_vec,
            "missing_vector": total - with_vec, "dims": dims,
            "empty_text": empty_text, "sample_text": sample_text,
            "missing_ids": missing_ids}


def target_collections(client, env: dict, args) -> list:
    if args.collections:
        return [c.strip() for c in args.collections.split(",") if c.strip()]
    if args.knowledge_id:
        return [sanitize_collection_name(args.knowledge_id)]
    import urllib.request
    host = os.environ.get("WEAVIATE_HTTP_HOST", "localhost")
    port = os.environ.get("WEAVIATE_HTTP_PORT", "8080")
    secure = (os.environ.get("WEAVIATE_HTTP_SECURE", "false").lower()
              in ("1", "true", "yes"))
    url = f"{'https' if secure else 'http'}://{host}:{port}/v1/schema"
    req = urllib.request.Request(url)
    key = os.environ.get("WEAVIATE_API_KEY", "")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=30) as r:
        schema = json.loads(r.read().decode())
    return sorted(c["class"] for c in (schema.get("classes") or [])
                  if c.get("class"))


# ══════════════════════════════════════════════════════════════════════════
# embedding — must match ingestion exactly (see module docstring)
# ══════════════════════════════════════════════════════════════════════════

def make_encoder():
    model = os.environ.get("RAG_EMBEDDING_MODEL") \
        or "sentence-transformers/all-MiniLM-L6-v2"
    engine = (os.environ.get("RAG_EMBEDDING_ENGINE") or "").strip().lower()
    if engine not in ("", "sentence-transformers", "st", "local"):
        raise SystemExit(
            f"RAG_EMBEDDING_ENGINE={engine!r}: this repair only reproduces the "
            "sentence-transformers path faithfully. Re-ingest through the app "
            "instead, or extend this script for that engine.")
    from sentence_transformers import SentenceTransformer
    try:
        st = SentenceTransformer(model, local_files_only=True)
    except Exception:
        st = SentenceTransformer(model)

    def encode(texts: list) -> list:
        # EmbeddingAgent: texts are newline-flattened, then encoded with
        # normalize_embeddings=True. No prefix: async_st_embedding ignores it.
        flat = [t.replace("\n", " ") for t in texts]
        return st.encode(flat, normalize_embeddings=True,
                         show_progress_bar=False,
                         convert_to_numpy=True).tolist()

    return encode, f"sentence-transformers:{model}"


def rewrite_named_space(client, cname: str, space: str, limit: int = 0) -> dict:
    """Move each object's legacy vector into the NAMED vector space.

    The failure this fixes: the class was created with
    Configure.Vectors.self_provided(), which on weaviate-client >= 4.16
    declares a named space ("default"), while insert() handed
    batch.add_object() a flat list — and a flat list goes to the legacy
    unnamed slot. Reading an object shows a 1024-dim vector (the client
    surfaces the legacy one under the key "default"), yet the named space
    the query targets is empty, so nearVector returns nothing and says
    "vector not found for target: default".

    The stored vectors are perfectly good, so nothing is re-embedded: each
    one is read and written back as {space: vector}, which lands it in the
    space queries actually search. Properties are untouched.
    """
    coll = client.client.collections.get(cname)
    moved = failed = skipped = 0
    for uid, _props, vec in iter_objects(client, cname):
        if not vec:
            skipped += 1
            continue
        try:
            coll.data.update(uuid=uid, vector={space: list(vec)})
            moved += 1
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"      ! {uid}: {type(e).__name__}: {e}")
        if limit and moved >= limit:
            break
        if moved and moved % 200 == 0:
            print(f"      {moved} moved…", flush=True)
    return {"moved": moved, "failed": failed, "skipped_no_vector": skipped}


def patch_vector(client, cname: str, uid: str, vector: list) -> None:
    """Update ONLY the vector, leaving properties untouched.

    Deliberately not the wrapper's upsert(): that rebuilds properties from a
    metadata round-trip, which risks changing types or dropping a field. A
    vector-only patch cannot.
    """
    coll = client.client.collections.get(cname)
    coll.data.update(uuid=uid, vector=vector)


# ══════════════════════════════════════════════════════════════════════════
# repair
# ══════════════════════════════════════════════════════════════════════════

def repair(client, cname: str, encode, batch: int, force: bool,
           limit: int = 0) -> dict:
    todo = []
    for uid, props, vec in iter_objects(client, cname):
        if vec and not force:
            continue
        text = (props.get("text") or "").strip()
        if not text:
            continue
        todo.append((uid, props["text"]))
        if limit and len(todo) >= limit:
            break
    written = failed = 0
    t0 = time.time()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        vectors = encode([t for _u, t in chunk])
        for (uid, _t), vec in zip(chunk, vectors):
            try:
                patch_vector(client, cname, uid, vec)
                written += 1
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"      ! patch failed for {uid}: "
                          f"{type(e).__name__}: {e}")
        done = min(i + batch, len(todo))
        print(f"      {done}/{len(todo)} embedded+written "
              f"({time.time() - t0:.0f}s)", flush=True)
    return {"candidates": len(todo), "written": written, "failed": failed}


def diagnose_dense(client, cname: str, probes: int = 3) -> dict:
    """Why does this collection return nothing to a dense query?

    Objects can hold perfectly good vectors and still be unreachable by
    `nearVector`: the vectors may live in a NAMED vector space while the
    query targets the unnamed one, or the class may have been created with
    its vector index disabled, in which case Weaviate answers with zero
    objects and no error at all. The decisive test is to query with an
    object's OWN vector — if that object does not come back first, the
    stored vectors are not in the space being searched.
    """
    out = {"collection": cname}
    try:                                    # class-level vector config
        import urllib.request
        host = os.environ.get("WEAVIATE_HTTP_HOST", "localhost")
        port = os.environ.get("WEAVIATE_HTTP_PORT", "8080")
        secure = (os.environ.get("WEAVIATE_HTTP_SECURE", "false").lower()
                  in ("1", "true", "yes"))
        url = (f"{'https' if secure else 'http'}://{host}:{port}"
               f"/v1/schema/{cname}")
        req = urllib.request.Request(url)
        key = os.environ.get("WEAVIATE_API_KEY", "")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        with urllib.request.urlopen(req, timeout=30) as r:
            sch = json.loads(r.read().decode())
        out["vectorizer"] = sch.get("vectorizer")
        out["vectorIndexType"] = sch.get("vectorIndexType")
        vic = sch.get("vectorIndexConfig") or {}
        out["index_skip"] = vic.get("skip")
        out["distance"] = vic.get("distance")
        out["named_vectors"] = sorted((sch.get("vectorConfig") or {}).keys())
    except Exception as e:
        out["schema_error"] = f"{type(e).__name__}: {e}"

    # Several objects, spread through the collection — NOT just the first.
    # Judging a whole collection from one object is how a partially repaired
    # collection reads as fully fixed: probe the first object after moving
    # five of 1427 and it reports success for all of them.
    coll = client.client.collections.get(cname)
    samples = []                    # [(uuid, vector)]
    vector_names = []
    total = 0
    try:                            # object count, for evenly spaced probes
        agg = coll.aggregate.over_all(total_count=True)
        total = getattr(agg, "total_count", 0) or 0
    except Exception:
        try:
            total = sum(1 for _ in coll.iterator())
        except Exception:
            total = 0
    # First, middle and last — NOT the first three. A partial repair fixes a
    # contiguous run from the start, so adjacent probes would all pass and
    # report the whole collection healthy.
    if total > 0:
        want_at = sorted({0, total // 2, max(total - 1, 0)}) if probes >= 3 \
            else sorted({0, max(total - 1, 0)})
        want_at = want_at[:max(probes, 1)]
    else:
        want_at = [0]
    try:
        last = max(want_at)
        for idx, obj in enumerate(coll.iterator(include_vector=True)):
            if idx in want_at:
                v = getattr(obj, "vector", None)
                if isinstance(v, dict):
                    vector_names = sorted(v.keys()) or vector_names
                    vec = (v.get("default") if "default" in v
                           else (next(iter(v.values())) if v else None))
                else:
                    vec = v
                if vec:
                    samples.append((str(obj.uuid), vec))
            if idx >= last:
                break
    except Exception as e:
        out["iterate_error"] = f"{type(e).__name__}: {e}"
    out["objects"] = total
    out["vector_spaces_on_object"] = vector_names
    out["sample_vector_dim"] = len(samples[0][1]) if samples else 0

    # self-query: an object searched by its own vector must return itself
    if samples:
        found = hits = 0
        try:
            for uid, vec in samples:
                resp = coll.query.near_vector(near_vector=list(vec), limit=3)
                got = [str(o.uuid) for o in resp.objects]
                hits += len(got)
                found += 1 if uid in got else 0
            out["self_query_probes"] = len(samples)
            out["self_query_hits"] = hits
            out["self_query_found_itself"] = (found == len(samples))
            out["self_query_found_count"] = found
        except Exception as e:
            out["self_query_error"] = f"{type(e).__name__}: {e}"
        sample_uuid, sample_vec = samples[0]
    else:
        sample_uuid = sample_vec = None
    if sample_uuid:
        try:
            resp = coll.query.near_object(near_object=sample_uuid, limit=3)
            out["near_object_hits"] = len(resp.objects)
        except Exception as e:
            out["near_object_error"] = f"{type(e).__name__}: {e}"
        # and through the wrapper the app actually uses
        if sample_vec:
            try:
                res = client.search(collection_name=cname,
                                    vectors=[list(sample_vec)], limit=3)
                docs = (res.documents or [[]])[0] if res else []
                out["wrapper_search_hits"] = len(docs or [])
            except Exception as e:
                out["wrapper_search_error"] = f"{type(e).__name__}: {e}"
    return out


def verify(client, cname: str, encode, probe: str) -> dict:
    """Prove the repair with a real dense query."""
    emb = encode([probe])[0]
    res = client.search(collection_name=cname, vectors=[emb], limit=3)
    docs = (res.documents or [[]])[0] if res else []
    metas = (res.metadatas or [[]])[0] if res and res.metadatas else []
    out = []
    for d, m in zip(docs or [], metas or []):
        out.append(str((m or {}).get("name")
                       or (m or {}).get("breadcrumb") or "")[:48]
                   or (d or "")[:48])
    return {"probe": probe, "dense_hits": len(docs or []), "top": out}


# ══════════════════════════════════════════════════════════════════════════

def self_test() -> int:
    checks = 0
    fails = []

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(msg)

    # collection-name sanitising must match the wrapper's rule
    ok(sanitize_collection_name("30fcb31b-5e7a-4de2-b134-e676f2e0678a")
       == "C30fcb31b_5e7a_4de2_b134_e676f2e0678a",
       f"kb id → {sanitize_collection_name('30fcb31b-5e7a-4de2-b134-e676f2e0678a')}")
    ok(sanitize_collection_name("abc-def") == "Abc_def", "alpha-first name")
    ok(sanitize_collection_name("AkuKnowledge") == "AkuKnowledge", "unchanged")

    # a fake client: some objects have vectors, some do not
    class _Coll:
        def __init__(self, objs):
            self.objs = objs
            self.updated = {}

        def iterator(self, include_vector=False):
            class O:
                pass
            for uid, props, vec in self.objs:
                o = O()
                o.uuid, o.properties = uid, props
                o.vector = ({"default": vec} if vec else None)
                yield o

        class _Data:
            def __init__(self, outer):
                self.outer = outer

            def update(self, uuid, vector=None, **kw):
                self.outer.updated[uuid] = vector

        @property
        def data(self):
            return self._Data(self)

    class _Inner:
        def __init__(self, coll):
            self._c = coll
            self.collections = self

        def get(self, name):
            return self._c

    class _Client:
        def __init__(self, coll):
            self.client = _Inner(coll)

        def search(self, collection_name, vectors, limit=3, **kw):
            class R:
                documents = [["doc one", "doc two"]]
                metadatas = [[{"name": "ch23_parliament.md"},
                              {"name": "ch08_fundamental_rights.md"}]]
            return R()

    objs = [("u1", {"text": "Article 79 constitutes Parliament"}, None),
            ("u2", {"text": "The Rajya Sabha is the upper house"}, None),
            ("u3", {"text": "already embedded"}, [0.1, 0.2]),
            ("u4", {"text": "   "}, None)]          # empty text: skipped
    coll = _Coll(objs)
    client = _Client(coll)

    info = inspect(client, "X")
    ok(info["objects"] == 4, f"objects {info['objects']}")
    ok(info["with_vector"] == 1, f"with_vector {info['with_vector']}")
    ok(info["missing_vector"] == 3, f"missing {info['missing_vector']}")
    ok(info["empty_text"] == 1, f"empty_text {info['empty_text']}")
    ok(info["dims"] == {2: 1}, f"dims {info['dims']}")

    calls = {"n": 0}

    def enc(texts):
        calls["n"] += 1
        # newline flattening must happen before encoding
        ok(all("\n" not in t for t in texts), "newlines reached the encoder")
        return [[float(len(t)), 1.0] for t in texts]

    res = repair(client, "X", enc, batch=10, force=False)
    ok(res["candidates"] == 2,
       f"should skip embedded + empty-text objects: {res}")
    ok(res["written"] == 2 and res["failed"] == 0, f"write result {res}")
    ok(set(coll.updated) == {"u1", "u2"},
       f"patched the wrong objects: {sorted(coll.updated)}")
    ok("u3" not in coll.updated, "an already-embedded object was overwritten")

    coll.updated.clear()
    res = repair(client, "X", enc, batch=10, force=True)
    ok(res["candidates"] == 3 and set(coll.updated) == {"u1", "u2", "u3"},
       f"--force should re-embed everything with text: {res}")

    coll.updated.clear()
    res = repair(client, "X", enc, batch=1, force=False, limit=1)
    ok(res["candidates"] == 1 and res["written"] == 1, f"--limit {res}")

    # newline handling in the real encoder path
    class _ST:
        def encode(self, texts, **kw):
            class A(list):
                def tolist(self):
                    return [[1.0] for _ in texts]
            ok(all("\n" not in t for t in texts),
               "real encoder got a newline")
            return A()
    import types as _t
    mod = _t.ModuleType("sentence_transformers")
    mod.SentenceTransformer = lambda *a, **k: _ST()
    sys.modules["sentence_transformers"] = mod
    os.environ["RAG_EMBEDDING_MODEL"] = "BAAI/bge-large-en-v1.5"
    enc2, name = make_encoder()
    enc2(["line one\nline two"])
    ok("bge-large" in name, f"encoder name {name}")

    v = verify(client, "X", lambda ts: [[0.0]], "Parliament")
    ok(v["dense_hits"] == 2 and "ch23_parliament.md" in v["top"],
       f"verify {v}")

    # moving legacy vectors into a named space must write dict-shaped
    # vectors, reuse the existing values, and skip vector-less objects
    coll2 = _Coll([("a1", {"text": "x"}, [0.1, 0.2]),
                   ("a2", {"text": "y"}, [0.3, 0.4]),
                   ("a3", {"text": "z"}, None)])
    res = rewrite_named_space(_Client(coll2), "X", "default")
    ok(res == {"moved": 2, "failed": 0, "skipped_no_vector": 1},
       f"rewrite result {res}")
    ok(coll2.updated == {"a1": {"default": [0.1, 0.2]},
                         "a2": {"default": [0.3, 0.4]}},
       f"vectors not written into the named space: {coll2.updated}")
    coll3 = _Coll([("b1", {"text": "x"}, [1.0]), ("b2", {"text": "y"}, [2.0])])
    ok(rewrite_named_space(_Client(coll3), "X", "default", limit=1)["moved"]
       == 1, "--limit ignored when moving vectors")

    # A PARTIALLY repaired collection must not report as fixed. Only the
    # first object is searchable here (as after a 5-object probe run); a
    # single-object probe called that success and skipped the other 1422.
    class _PartialClient(_Client):
        """Only `fixed` uuids answer a self-query."""

        def __init__(self, coll, fixed):
            _Client.__init__(self, coll)
            self.fixed = fixed
            outer = self

            class _Q:
                def near_vector(self, near_vector, limit=3, **kw):
                    uid = outer.by_vec.get(tuple(near_vector))

                    class O:
                        pass
                    if uid in outer.fixed:
                        o = O()
                        o.uuid = uid
                        return type("R", (), {"objects": [o]})()
                    return type("R", (), {"objects": []})()

                def near_object(self, near_object, limit=3, **kw):
                    return type("R", (), {"objects": []})()
            coll.query = _Q()
            self.by_vec = {}

    objs2 = [(f"c{i}", {"text": "t"}, [float(i)]) for i in range(6)]
    coll4 = _Coll(objs2)
    pc = _PartialClient(coll4, fixed={"c0"})
    pc.by_vec = {(float(i),): f"c{i}" for i in range(6)}
    d = diagnose_dense(pc, "X", probes=3)
    ok(d.get("self_query_probes", 0) >= 2,
       f"should probe more than one object: {d.get('self_query_probes')}")
    ok(d.get("self_query_found_itself") is False,
       f"a partly repaired collection reported as fully working: {d}")
    ok(d.get("self_query_found_count") == 1,
       f"found_count {d.get('self_query_found_count')}")

    print(f"repair_vectors self-test: {checks} checks, {len(fails)} failures")
    for f in fails:
        print(f"  FAIL: {f}")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(
        description="Diagnose and backfill missing Weaviate vectors.")
    ap.add_argument("--knowledge-id", default="",
                    help="knowledge base UUID (mapped to its class name)")
    ap.add_argument("--collections", default="",
                    help="explicit comma-separated class names")
    ap.add_argument("--apply", action="store_true",
                    help="actually write vectors (default: diagnose only)")
    ap.add_argument("--force", action="store_true",
                    help="re-embed objects that already have a vector")
    ap.add_argument("--limit", type=int, default=0,
                    help="repair at most N objects (smoke test)")
    ap.add_argument("--batch", type=int, default=64,
                    help="embedding batch size (default 64)")
    ap.add_argument("--verify", action="store_true",
                    help="after repair, run a dense probe query")
    ap.add_argument("--probe", default="powers of the Rajya Sabha",
                    help="text for the --verify dense query")
    ap.add_argument("--diagnose-dense", action="store_true",
                    help="explain why a collection returns no dense hits "
                         "(vector config, named spaces, self-query test)")
    ap.add_argument("--skip-if-working", action="store_true",
                    help="skip a collection whose probe objects are already "
                         "searchable (off by default: the move is idempotent, "
                         "and skipping can leave a partial repair unfinished)")
    ap.add_argument("--move-to-named-space", default="",
                    help="move each object's legacy vector into this named "
                         "vector space (usually 'default'). Fixes 'vector not "
                         "found for target: default' without re-embedding. "
                         "Needs --apply.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    env = load_env(ROOT)
    sys.path.insert(0, str(ROOT / "backend"))
    from futureverse_ai.retrieval.vector.factory import VECTOR_DB_CLIENT

    names = target_collections(VECTOR_DB_CLIENT, env, args)

    if args.diagnose_dense:
        for n in names:
            d = diagnose_dense(VECTOR_DB_CLIENT, n)
            print(f"\n{n}")
            for key in ("vectorizer", "vectorIndexType", "index_skip",
                        "distance", "named_vectors",
                        "vector_spaces_on_object", "sample_vector_dim",
                        "self_query_probes", "self_query_found_count",
                        "self_query_hits", "self_query_found_itself",
                        "near_object_hits", "wrapper_search_hits",
                        "schema_error", "iterate_error", "self_query_error",
                        "near_object_error", "wrapper_search_error"):
                if key in d:
                    print(f"    {key:26s} {d[key]}")
            if d.get("sample_vector_dim") and not d.get("self_query_hits"):
                print("    → objects carry vectors but a nearVector query "
                      "finds nothing: they are not in the searched space "
                      "(named vector, or the vector index is disabled).")
            elif d.get("self_query_found_itself"):
                print("    → dense search works on this collection.")
        for obj in (VECTOR_DB_CLIENT,
                    getattr(VECTOR_DB_CLIENT, "client", None)):
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                    break
                except Exception:
                    pass
        return

    if args.move_to_named_space:
        space = args.move_to_named_space
        for n in names:
            before = diagnose_dense(VECTOR_DB_CLIENT, n)
            print(f"\n{n}")
            print(f"    before: {before.get('self_query_found_count')}/"
                  f"{before.get('self_query_probes')} probe objects found "
                  f"themselves, named spaces {before.get('named_vectors')}")
            # No early skip. Writing a vector back into the space it already
            # occupies is a no-op, so the only thing a skip can do is leave a
            # partially repaired collection half broken — which is exactly
            # what happened when this checked a single object after a 5-object
            # probe run and declared all 1427 fixed.
            if args.skip_if_working and before.get("self_query_found_itself"):
                print("    every probe object already searchable and "
                      "--skip-if-working given — skipping.")
                continue
            if not args.apply:
                print(f"    would move {before.get('sample_vector_dim')}-dim "
                      f"vectors into space {space!r} for every object "
                      f"(add --apply)")
                continue
            res = rewrite_named_space(VECTOR_DB_CLIENT, n, space, args.limit)
            print(f"    moved {res['moved']}, failed {res['failed']}, "
                  f"no-vector {res['skipped_no_vector']}")
            after = diagnose_dense(VECTOR_DB_CLIENT, n)
            print(f"    after: {after.get('self_query_found_count')}/"
                  f"{after.get('self_query_probes')} probe objects found "
                  f"themselves, wrapper hits "
                  f"{after.get('wrapper_search_hits')}")
            if after.get("self_query_found_itself"):
                print("    → dense search now WORKS on this collection.")
            else:
                print("    → still not searchable; the collection likely "
                      "needs recreating and re-ingesting.")
        for obj in (VECTOR_DB_CLIENT,
                    getattr(VECTOR_DB_CLIENT, "client", None)):
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                    break
                except Exception:
                    pass
        return

    print(f"{SCRIPT_VERSION}: inspecting {len(names)} collection(s)\n")

    encode = enc_name = None
    broken = []
    for n in names:
        try:
            info = inspect(VECTOR_DB_CLIENT, n)
        except Exception as e:
            print(f"  {n[:48]:48s} ! {type(e).__name__}: {str(e)[:60]}")
            continue
        dims = ", ".join(f"{d}-dim x{c}" for d, c in info["dims"].items()) \
            or "none"
        state = "OK" if info["missing_vector"] == 0 else \
            ("NO VECTORS AT ALL" if info["with_vector"] == 0 else "PARTIAL")
        print(f"  {n[:48]:48s} {info['objects']:6d} objects | "
              f"vectors {info['with_vector']:6d} ({dims}) | {state}")
        if info["empty_text"]:
            print(f"      note: {info['empty_text']} object(s) have empty "
                  f"text and cannot be embedded")
        if info["missing_vector"]:
            broken.append((n, info))

    if not broken:
        print("\nEvery collection has vectors for every object — nothing to "
              "repair.")
        return

    total = sum(i["missing_vector"] for _n, i in broken)
    print(f"\n{len(broken)} collection(s), {total} object(s) missing vectors. "
          f"Dense retrieval returns NOTHING for those objects.")
    if not args.apply:
        print("Re-run with --apply to embed and patch them in place "
              "(properties are not touched). Add --verify to prove it with a "
              "dense query afterwards.")
        return

    encode, enc_name = make_encoder()
    print(f"\nembedder: {enc_name}   batch={args.batch}\n")
    for n, info in broken:
        print(f"  repairing {n} ({info['missing_vector']} object(s))")
        res = repair(VECTOR_DB_CLIENT, n, encode, args.batch, args.force,
                     args.limit)
        print(f"    written {res['written']}, failed {res['failed']}")
        after = inspect(VECTOR_DB_CLIENT, n)
        print(f"    now: {after['with_vector']}/{after['objects']} objects "
              f"have vectors")
        if args.verify:
            v = verify(VECTOR_DB_CLIENT, n, encode, args.probe)
            print(f"    dense probe {v['probe']!r} → {v['dense_hits']} hit(s)"
                  + (f": {', '.join(v['top'])}" if v["top"] else ""))
            if not v["dense_hits"]:
                print("    ! still no dense hits — check the collection's "
                      "vector index config")

    for obj in (VECTOR_DB_CLIENT, getattr(VECTOR_DB_CLIENT, "client", None)):
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
                break
            except Exception:
                pass


if __name__ == "__main__":
    main()
