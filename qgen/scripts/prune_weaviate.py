"""
prune_weaviate.py — delete every Weaviate collection except the ones you keep.

    # see exactly what would go, and what would remain (writes nothing)
    python qgen\\scripts\\prune_weaviate.py --keep 30fcb31b-5e7a-4de2-b134-e676f2e0678a --keep PyqKnowledge

    # do it
    python qgen\\scripts\\prune_weaviate.py --keep 30fcb31b-5e7a-4de2-b134-e676f2e0678a --keep PyqKnowledge --apply

--keep accepts a knowledge-base UUID or a class name; UUIDs are mapped
through the same sanitiser the Weaviate wrapper uses, so
"30fcb31b-5e7a-4de2-b134-e676f2e0678a" and
"C30fcb31b_5e7a_4de2_b134_e676f2e0678a" both work.

THIS DELETES DATA PERMANENTLY. Two safeguards: nothing happens without
--apply, and the script refuses to run with an empty keep-list.

WHAT DELETION DOES AND DOES NOT DO
──────────────────────────────────
Dropping a collection removes the vectors and the BM25 index for that
corpus. It does NOT remove the knowledge-base or file records from the
application database, so:
  • those knowledge bases still appear in the UI, now backed by nothing —
    a query against one returns no results rather than an error;
  • re-adding any file to one recreates its collection and re-ingests it,
    which is the normal way back if you delete something by mistake;
  • to remove them for good, delete the knowledge base in the UI as well —
    that is the only path that also cleans up the database records.
So prefer deleting a knowledge base in the UI when you want it gone
entirely; use this script when you want the index cleared out now, e.g. to
make retrieval draw on one corpus only.

Per-file collections (File_*) belong to a knowledge base and duplicate its
master, so keeping a master does not require keeping them — but if you keep
a KB whose master is empty, its content lives ONLY in those per-file
collections and deleting them destroys it. The script checks for that case
and refuses unless you pass --allow-orphaning.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

SCRIPT_VERSION = "prune-weaviate-1"


def find_repo_root(start: Path) -> Path:
    for cand in [start] + list(start.parents):
        if (cand / "backend" / "futureverse_ai").is_dir():
            return cand
    return start


ROOT = find_repo_root(Path(__file__).resolve().parent)


def load_env(root: Path) -> dict:
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
    col = re.sub(r"[^a-zA-Z0-9_]", "", str(name).replace("-", "_")).strip("_")
    if not col:
        return col
    return ("C" + col) if not col[0].isalpha() else (col[0].upper() + col[1:])


def _weaviate_get(path: str, body: dict = None) -> dict:
    import urllib.request
    host = os.environ.get("WEAVIATE_HTTP_HOST", "localhost")
    port = os.environ.get("WEAVIATE_HTTP_PORT", "8080")
    secure = (os.environ.get("WEAVIATE_HTTP_SECURE", "false").lower()
              in ("1", "true", "yes"))
    base = f"{'https' if secure else 'http'}://{host}:{port}"
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json"} if body else {})
    key = os.environ.get("WEAVIATE_API_KEY", "")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def collection_counts() -> dict:
    names = [c["class"] for c in (_weaviate_get("/v1/schema").get("classes")
                                 or []) if c.get("class")]
    if not names:
        return {}
    try:
        q = "{Aggregate{" + " ".join(f"{n}{{meta{{count}}}}" for n in names) \
            + "}}"
        agg = (_weaviate_get("/v1/graphql", {"query": q}).get("data")
               or {}).get("Aggregate") or {}
        return {n: ((agg.get(n) or [{}])[0].get("meta", {}).get("count") or 0)
                for n in names}
    except Exception:
        return {n: None for n in names}


def resolve_keep(keep_args: list, names: list) -> tuple:
    """(kept, unmatched) — map each --keep value onto a real class name."""
    kept, unmatched = set(), []
    lower = {n.lower(): n for n in names}
    for k in keep_args:
        k = k.strip()
        if not k:
            continue
        for cand in (k, sanitize_collection_name(k)):
            hit = lower.get(cand.lower())
            if hit:
                kept.add(hit)
                break
        else:
            unmatched.append(k)
    return kept, unmatched


def plan(counts: dict, keep_args: list) -> dict:
    names = sorted(counts)
    kept, unmatched = resolve_keep(keep_args, names)
    doomed = [n for n in names if n not in kept]
    # A kept master with no objects depends on its per-file collections;
    # deleting those would destroy the only copy of its content.
    orphaning = []
    if any(counts.get(n) == 0 for n in kept):
        per_file_doomed = [n for n in doomed
                           if re.match(r"^C?file[_-]", n, re.I)
                           and (counts.get(n) or 0) > 0]
        if per_file_doomed:
            orphaning = per_file_doomed
    return {"kept": sorted(kept), "doomed": doomed, "unmatched": unmatched,
            "orphaning": orphaning}


def main():
    ap = argparse.ArgumentParser(
        description="Delete all Weaviate collections except those kept.")
    ap.add_argument("--keep", action="append", default=[],
                    help="knowledge-base UUID or class name to keep "
                         "(repeatable)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default: show the plan only)")
    ap.add_argument("--allow-orphaning", action="store_true",
                    help="permit deleting per-file collections even when a "
                         "kept master is empty and depends on them")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if not args.keep:
        raise SystemExit(
            "refusing to run with an empty keep-list — pass --keep for every "
            "collection or knowledge-base id you want to survive, e.g.\n"
            "  --keep 30fcb31b-5e7a-4de2-b134-e676f2e0678a --keep PyqKnowledge")

    load_env(ROOT)
    counts = collection_counts()
    if not counts:
        raise SystemExit("no collections found (is Weaviate reachable?)")
    p = plan(counts, args.keep)

    if p["unmatched"]:
        raise SystemExit(
            "these --keep values match no collection: "
            + ", ".join(p["unmatched"])
            + "\nNothing was deleted. Check the names with:\n"
              "  python qgen\\scripts\\pyq_chunk_extractor.py --list-collections")

    print(f"\n{SCRIPT_VERSION}\n")
    print("KEEP:")
    for n in p["kept"]:
        print(f"  {n[:52]:52s} {counts.get(n)} objects")
    print("\nDELETE:")
    if not p["doomed"]:
        print("  (nothing)")
    total = 0
    for n in p["doomed"]:
        c = counts.get(n) or 0
        total += c
        print(f"  {n[:52]:52s} {c} objects")
    print(f"\n{len(p['doomed'])} collection(s), {total} object(s) would be "
          f"deleted permanently.")

    if p["orphaning"] and not args.allow_orphaning:
        raise SystemExit(
            "\nREFUSING: a collection you are keeping has 0 objects, so its "
            "content lives only in per-file collections that this would "
            "delete:\n  " + "\n  ".join(p["orphaning"])
            + "\nKeep those too, or pass --allow-orphaning if you really "
              "mean it.")

    if not args.apply:
        print("\nDry run — nothing deleted. Re-run with --apply to proceed.")
        return

    sys.path.insert(0, str(ROOT / "backend"))
    from futureverse_ai.retrieval.vector.factory import VECTOR_DB_CLIENT
    ok = failed = 0
    for n in p["doomed"]:
        try:
            VECTOR_DB_CLIENT.delete_collection(collection_name=n)
            print(f"  deleted {n}")
            ok += 1
        except Exception as e:
            print(f"  ! failed {n}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\ndeleted {ok}, failed {failed}")
    after = collection_counts()
    print("remaining collections:")
    for n in sorted(after):
        print(f"  {n[:52]:52s} {after[n]} objects")
    print("\nNote: the knowledge-base and file records still exist in the "
          "app database. Those knowledge bases will now return no results; "
          "delete them in the UI to remove them entirely, or re-add a file "
          "to rebuild a collection.")
    for obj in (VECTOR_DB_CLIENT, getattr(VECTOR_DB_CLIENT, "client", None)):
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
                break
            except Exception:
                pass


def self_test() -> int:
    checks = 0
    fails = []

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(msg)

    ok(sanitize_collection_name("30fcb31b-5e7a-4de2-b134-e676f2e0678a")
       == "C30fcb31b_5e7a_4de2_b134_e676f2e0678a", "uuid mapping")

    counts = {
        "AkuKnowledge": 2203,
        "C22e7c3df_4026_4ceb_93e5_1547a98eb2bf": 30,
        "C2dfdbb43_07dc_4deb_b8ed_4d774ba14e38": 1292,
        "C30fcb31b_5e7a_4de2_b134_e676f2e0678a": 1427,
        "File_5eb53337_fa9b_41d1_9a86_9509a203bca0": 15,
        "Knowledge_bases": 2,
        "PyqKnowledge": 395,
    }
    # keeping by UUID and by class name, mixed
    p = plan(counts, ["30fcb31b-5e7a-4de2-b134-e676f2e0678a", "PyqKnowledge"])
    ok(p["kept"] == ["C30fcb31b_5e7a_4de2_b134_e676f2e0678a", "PyqKnowledge"],
       f"kept {p['kept']}")
    ok(len(p["doomed"]) == 5 and "AkuKnowledge" in p["doomed"],
       f"doomed {p['doomed']}")
    ok(not p["unmatched"], f"unmatched {p['unmatched']}")
    ok(not p["orphaning"], f"orphaning {p['orphaning']}")

    # a typo must be caught rather than silently deleting everything
    p2 = plan(counts, ["PqyKnowledge"])
    ok(p2["unmatched"] == ["PqyKnowledge"], f"typo not caught: {p2}")

    # keeping an EMPTY master while deleting its per-file collections is
    # data loss and must be flagged
    counts2 = dict(counts, C30fcb31b_5e7a_4de2_b134_e676f2e0678a=0)
    p3 = plan(counts2, ["30fcb31b-5e7a-4de2-b134-e676f2e0678a"])
    ok(p3["orphaning"] == ["File_5eb53337_fa9b_41d1_9a86_9509a203bca0"],
       f"orphaning not detected: {p3['orphaning']}")

    # class name given directly, case-insensitively
    p4 = plan(counts, ["c30fcb31b_5e7a_4de2_b134_e676f2e0678a"])
    ok(p4["kept"] == ["C30fcb31b_5e7a_4de2_b134_e676f2e0678a"],
       f"case-insensitive keep failed: {p4['kept']}")

    print(f"prune_weaviate self-test: {checks} checks, {len(fails)} failures")
    for f in fails:
        print(f"  FAIL: {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    main()
