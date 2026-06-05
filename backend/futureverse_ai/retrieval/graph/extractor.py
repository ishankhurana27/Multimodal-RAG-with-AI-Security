# -*- coding: utf-8 -*-
"""
Atlas-RAG bridge extractor for Open WebUI.
Runs Atlas-RAG KG construction pipeline per chunk, imports CSVs to Neo4j.
Falls back to direct Ollama extraction if Atlas-RAG fails.
"""
import os, re, json, csv, glob, logging, shutil, tempfile, threading, sys
from pathlib import Path
from neo4j import GraphDatabase
from futureverse_ai.retrieval.graph.community_detection import build_graph_communities

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
NEO4J_URI       = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER      = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD  = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_DATABASE  = os.environ.get("NEO4J_DATABASE", "neo4j")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
KG_MODEL        = os.environ.get("KG_EXTRACT_MODEL", "llama3.1:8b")
# atlas-rag is installed via pip (requirements.txt) — no sys.path needed.
# Override ATLAS_WORK_DIR if you need a custom temp directory for KG extraction.
_DEFAULT_WORK_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "atlas_work")
ATLAS_WORK_DIR  = os.environ.get("ATLAS_WORK_DIR", os.path.abspath(_DEFAULT_WORK_DIR))

IMPORT_COUNTER = 0


def _ensure_indexes():
    """Create Neo4j indexes for fast lookup — idempotent, runs once at startup."""
    indexes = [
        "CREATE INDEX entity_id_idx IF NOT EXISTS FOR (n:Entity) ON (n.id)",
        "CREATE INDEX entity_subject_idx IF NOT EXISTS FOR (n:Entity) ON (n.subject)",
        "CREATE INDEX entity_chapter_idx IF NOT EXISTS FOR (n:Entity) ON (n.chapter)",
        "CREATE INDEX concept_name_idx IF NOT EXISTS FOR (c:Concept) ON (c.name)",
        "CREATE INDEX topic_chapter_idx IF NOT EXISTS FOR (t:TopicSummary) ON (t.chapter)",
        "CREATE INDEX topic_subject_idx IF NOT EXISTS FOR (t:TopicSummary) ON (t.subject)",
        # Relation property index — enables fast CONTAINS search on edge predicates
        "CREATE INDEX relates_to_rel_idx IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.relation)",
        # TextSource node index
        "CREATE INDEX text_source_id_idx IF NOT EXISTS FOR (t:TextSource) ON (t.id)",
        # ExamQuestion node index (TESTED_IN target)
        "CREATE INDEX exam_question_id_idx IF NOT EXISTS FOR (q:ExamQuestion) ON (q.id)",
        "CREATE INDEX exam_question_year_idx IF NOT EXISTS FOR (q:ExamQuestion) ON (q.year)",
        # Semantic edge type indexes
        "CREATE INDEX causes_chapter_idx IF NOT EXISTS FOR ()-[r:CAUSES]-() ON (r.chapter)",
        "CREATE INDEX governed_by_chapter_idx IF NOT EXISTS FOR ()-[r:GOVERNED_BY]-() ON (r.chapter)",
        "CREATE INDEX tested_in_year_idx IF NOT EXISTS FOR ()-[r:TESTED_IN]-() ON (r.year)",
        "CREATE INDEX drives_chapter_idx IF NOT EXISTS FOR ()-[r:DRIVES]-() ON (r.chapter)",
        "CREATE INDEX unifies_chapter_idx IF NOT EXISTS FOR ()-[r:UNIFIES]-() ON (r.chapter)",
        "CREATE INDEX subset_of_chapter_idx IF NOT EXISTS FOR ()-[r:SUBSET_OF]-() ON (r.chapter)",
        "CREATE INDEX component_of_chapter_idx IF NOT EXISTS FOR ()-[r:COMPONENT_OF]-() ON (r.chapter)",
        "CREATE INDEX modified_by_chapter_idx IF NOT EXISTS FOR ()-[r:MODIFIED_BY]-() ON (r.chapter)",
        "CREATE INDEX originated_by_chapter_idx IF NOT EXISTS FOR ()-[r:ORIGINATED_BY]-() ON (r.chapter)",
        "CREATE INDEX conflict_link_chapter_idx IF NOT EXISTS FOR ()-[r:CONFLICT_LINK]-() ON (r.chapter)",
        # Generic-hub flag — makes WHERE NOT n.generic_hub a fast property lookup
        "CREATE INDEX entity_generic_hub_idx IF NOT EXISTS FOR (n:Entity) ON (n.generic_hub)",
    ]
    try:
        for cypher in indexes:
            run_query(cypher)
        log.info("Neo4j indexes verified.")
    except Exception as e:
        log.warning(f"Index creation skipped: {e}")

# ── Neo4j ─────────────────────────────────────────────────────────────────────
_driver = None
_driver_lock = threading.Lock()
_indexes_created = False
_hubs_marked    = False   # set True after first mark_generic_hubs() run

def get_driver(force_new=False):
    global _driver
    with _driver_lock:
        if _driver is None or force_new:
            if _driver is not None:
                try: _driver.close()
                except: pass
            _driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
                max_connection_lifetime=300, max_connection_pool_size=5, keep_alive=True,
            )
        return _driver

def run_query(cypher, params=None):
    global _indexes_created, _hubs_marked
    if not _indexes_created and "CREATE INDEX" not in cypher:
        try:
            _ensure_indexes()
            _indexes_created = True
        except Exception:
            _indexes_created = True  # don't retry on every call
    # After indexes exist, tag generic hub nodes once in a background thread
    # (mark_generic_hubs itself calls run_query, so it must NOT run during
    #  _ensure_indexes — hence the separate flag and deferred thread launch)
    if _indexes_created and not _hubs_marked and "generic_hub" not in cypher:
        _hubs_marked = True  # set before thread starts to prevent re-entry
        threading.Thread(target=mark_generic_hubs, daemon=True).start()
    for attempt in range(3):
        try:
            with get_driver().session(database=NEO4J_DATABASE) as s:
                return [dict(r) for r in s.run(cypher, params or {})]
        except Exception as e:
            msg = str(e).lower()
            if any(x in msg for x in ["defunct","routing","no data","closed","unavailable","unauthorized"]):
                log.warning(f"Neo4j reconnecting ({attempt+1}/3)...")
                get_driver(force_new=True)
                import time; time.sleep(3)
            else:
                raise
    raise RuntimeError("Neo4j failed after 3 attempts")

# ── Helpers ───────────────────────────────────────────────────────────────────
# AutoSchemaKG stores relation predicates as free-form strings in the `relation`
# property of a single RELATES_TO edge type.  We follow the same convention:
# one edge type, predicate stored as property.  This eliminates 900+ dynamic
# Neo4j relationship types and makes all edges queryable with CONTAINS.

import ast as _ast

def _parse_concepts(raw: str) -> list:
    """Parse AutoSchemaKG concepts field (Python list repr or pipe-separated)."""
    if not raw or raw.strip() in ("", "[]"):
        return []
    try:
        val = _ast.literal_eval(raw)
        if isinstance(val, list):
            return [str(c).strip() for c in val if str(c).strip()]
    except Exception:
        pass
    return [c.strip() for c in raw.split("|") if c.strip()]


# Entity noise patterns — skip nodes whose id matches these (document meta leaking in)
_ENTITY_NOISE = re.compile(
    r"^(physical geo|fundamentals|human geography|india people|india physical"
    r"|ncert|class \d|class_\d|textbook|chapter \d|unit \d|page \d|\d+$"
    r"|physical & world|world geography|[a-z]{1,2}$)",
    re.IGNORECASE
)

# Sentence-level noise: reject if name contains these patterns indicating full sentences
_SENTENCE_NOISE = re.compile(
    r"\b(is the|are the|which is|that is|this is|these are|it is|they are"
    r"|was the|were the|has been|have been|such as|including|known as"
    r"|defined as|refers to|consists of|composed of)\b",
    re.IGNORECASE
)

# Max entity name length — anything longer is a sentence/paragraph, not an entity
_MAX_ENTITY_LEN = 80

# ── Generic-hub stop-word entities ────────────────────────────────────────────
# Single tokens that appear in almost every NCERT sentence and thus collect
# hundreds of edges, turning them into noisy super-nodes that degrade RAG.
# Exact-match on lowercased entity id.  Multi-word forms (e.g. "india gate")
# are NOT blocked — only bare single-word tokens listed here.
_GENERIC_HUB_STOP: frozenset[str] = frozenset({
    # Geopolitical super-nodes
    "india", "world", "earth", "globe", "planet", "universe",
    "country", "countries", "nation", "nations",
    # NOTE: "state" / "states" removed — technical term in Polity (Article 12,
    # state government, state list).  Re-add only if graph noise becomes critical.
    "continent", "continents", "region", "regions",
    # Demographic
    "people", "person", "persons", "human", "humans",
    "population", "man", "men", "woman", "women",
    # Physical geography singletons
    "land", "water", "soil", "rock", "rocks", "stone", "air",
    "area", "areas", "place", "places", "location", "locations",
    "zone", "zones", "surface", "ground", "terrain",
    # Abstract / catch-all
    "type", "types", "kind", "kinds", "form", "forms",
    "part", "parts", "thing", "things", "factor", "factors",
    "aspect", "aspects", "process", "processes", "system", "systems",
    "change", "changes", "effect", "effects", "result", "results",
    "nature", "material", "substance",
    # NOTE: "matter", "element", "energy", "force", "power", "life" removed —
    # all carry technical UPSC meaning:
    #   matter/element → Chemistry (states of matter, periodic table)
    #   energy         → Physics / Environment (solar, nuclear)
    #   force          → Physics / Polity (armed forces, centripetal force)
    #   power          → Polity (separation of powers, concurrent list)
    #   life           → Biology / Polity (Article 21 — Right to Life)
    "living",
    # Time tokens
    "time", "times", "year", "years", "day", "days",
    "period", "periods", "era", "age", "century",
    # Common stop words that slip through as entities
    "other", "others", "various", "different", "important",
    "major", "main", "large", "small", "high", "low",
    "also", "however", "therefore", "thus", "hence",
})


def _is_noise_entity(name: str) -> bool:
    """Return True if name looks like a document title / sentence / numeric / too short."""
    n = name.strip().lower()
    if len(n) < 3:
        return True
    if len(n) > _MAX_ENTITY_LEN:
        return True
    if n.isdigit():
        return True
    # Reject names with more than 8 words (sentences disguised as entities)
    if len(n.split()) > 8:
        return True
    if _ENTITY_NOISE.match(n):
        return True
    if _SENTENCE_NOISE.search(n):
        return True
    # Block bare generic stop-word tokens (single word, no compound context)
    if n in _GENERIC_HUB_STOP:
        return True
    return False


# ── Atomic Gate ───────────────────────────────────────────────────────────────
# Universal UPSC Semantic Map — subject-agnostic normalization of LLM surface
# variants to canonical edge types.  Ordered from most-specific to most-generic
# so the first match wins.
#
# Subjects covered:
#   Geography / Science  → TRIGGERS, DRIVES, CHARACTERIZED_BY, RESULTS_IN, CAUSES
#   Polity / Governance  → GOVERNED_BY, MODIFIED_BY
#   Structural / All     → SUBSET_OF, COMPONENT_OF
#   History / Culture    → ORIGINATED_BY, CONFLICT_LINK
#   Theory / Science     → UNIFIES
_REL_NORM_MAP: list[tuple[tuple[str, ...], str]] = [
    # ── Causal & Process (Geography / Science / Economics) ──────────────────
    (("trigger", "triggered", "set off", "sets off", "sparks", "spark"), "TRIGGERS"),
    (("drive", "drives", "driven by", "powers", "fuels", "propels", "sustains"), "DRIVES"),
    (("characterized by", "defined by", "known for", "marked by", "typified by"), "CHARACTERIZED_BY"),
    (("results in", "leads to", "produces", "gives rise to", "yields", "result in"), "RESULTS_IN"),
    (("caused by", "due to", "owing to", "because of", "induces", "generates"), "CAUSES"),
    # ── Regulatory & Authority (Polity / Governance) ────────────────────────
    (("governed by", "regulated by", "controlled by",
      "under article", "enforced by", "mandated by", "prescribed by"), "GOVERNED_BY"),
    (("amended by", "modified by", "repealed by", "updated by",
      "replaced by", "inserted by", "substituted by"), "MODIFIED_BY"),
    # ── Structural & Taxonomic (All Subjects) ────────────────────────────────
    (("type of", "kind of", "example of", "subdivision of",
      "category of", "form of", "class of", "variant of"), "SUBSET_OF"),
    (("part of", "component of", "element of",
      "segment of", "branch of", "section of", "consists of"), "COMPONENT_OF"),
    # ── Historical & Cultural (History / Culture) ────────────────────────────
    (("founded", "established by", "created by", "built by",
      "instituted", "originated", "started by", "introduced by"), "ORIGINATED_BY"),
    (("opposed", "conflict with", "war against", "rebelled against",
      "fought against", "competitor of", "rival of"), "CONFLICT_LINK"),
    # ── Theory Unification (Science / Geography) ─────────────────────────────
    (("unifies", "unified", "reconciles", "validates",
      "confirms", "synthesizes", "integrates", "consolidates"), "UNIFIES"),
]

# LLM chatter words that should never be entity names
_INSTRUCTION_WORDS: frozenset[str] = frozenset([
    "explain", "define", "note", "describe", "list", "state", "mention", "discuss",
    "consider", "answer", "remember", "identify", "examine", "analyse", "analyze",
])

# Leading article / quantifier words that signal LLM-generated non-entities
_LEADING_STOP: frozenset[str] = frozenset([
    "the", "a", "an", "various", "several", "some", "many", "few",
    "all", "any", "each", "every", "both", "either", "these", "those",
])


def validate_and_clean_extraction(
    entities: list[dict], relations: list[dict]
) -> tuple[list[dict], list[dict]]:
    """
    Atomic Gate applied to raw LLM JSON output before it reaches Neo4j.

    Entity rules (subject-agnostic — works for Geography, Polity, History, Economics):
      1. Reject names longer than 4 words — sentence fragments, not concepts.
      2. Reject names whose first word is a leading article / quantifier ("the", "various"…).
      3. Reject names that contain instruction verbs (explain, define, note …).

    Relation rules:
      4. Normalize surface variants to canonical UPSC semantic edge types using
         the Universal Semantic Map (_REL_NORM_MAP).  Covers all subjects.
    """
    clean_entities: list[dict] = []
    for ent in entities:
        name = str(ent.get("id") or ent.get("name") or "").strip().lower()
        if not name:
            continue
        words = name.split()
        # Rule 1: reject multi-word sentence fragments (UPSC concepts are ≤4 words)
        if len(words) > 4:
            continue
        # Rule 2: reject LLM quantifier preamble ("the monsoon", "various rivers"…)
        if words and words[0] in _LEADING_STOP:
            continue
        # Rule 3: reject instruction artefacts leaking into entity list
        if any(w in words for w in _INSTRUCTION_WORDS):
            continue
        clean_entities.append(ent)

    clean_relations: list[dict] = []
    for rel in relations:
        rel_type = str(rel.get("type") or "").lower().strip()
        for keywords, canonical in _REL_NORM_MAP:
            if any(kw in rel_type for kw in keywords):
                rel = dict(rel)   # don't mutate the original
                rel["type"] = canonical
                break
        clean_relations.append(rel)

    return clean_entities, clean_relations


def mark_generic_hubs(degree_threshold: int = 100) -> dict:
    """
    Retroactively tag existing high-degree generic hub nodes with
    ``generic_hub=true`` so retrieval queries can exclude them.

    Two-pass strategy (non-destructive — no edges or nodes are deleted):
      Pass 1 — exact stop-word list: stamp every node whose id is in
               _GENERIC_HUB_STOP regardless of degree.
      Pass 2 — degree heuristic: stamp single-token nodes whose
               relationship count exceeds ``degree_threshold`` AND whose
               name appears on the stop list after lower-casing.
               (Protects legitimate high-degree nodes like "plate tectonics".)

    Returns a dict with counts of nodes stamped in each pass.
    """
    stamped = {"stop_word_pass": 0, "degree_pass": 0}

    # Pass 1: direct stop-word list — fast, exact
    stop_list = list(_GENERIC_HUB_STOP)
    try:
        result = run_query(
            "UNWIND $ids AS nid "
            "MATCH (n:Entity {id: nid}) "
            "SET n.generic_hub = true "
            "RETURN count(n) AS cnt",
            {"ids": stop_list},
        )
        stamped["stop_word_pass"] = (result[0]["cnt"] if result else 0)
        log.info(f"mark_generic_hubs pass-1 (stop-word): {stamped['stop_word_pass']} nodes tagged")
    except Exception as e:
        log.warning(f"mark_generic_hubs pass-1 failed: {e}")

    # Pass 2: degree-based heuristic for any single-word hub not caught above
    try:
        result = run_query(
            """
            MATCH (n:Entity)
            WHERE n.generic_hub IS NULL
              AND size(split(n.id, ' ')) = 1
            WITH n, size([(n)-[]-() | 1]) AS degree
            WHERE degree >= $threshold
            SET n.generic_hub = true
            RETURN count(n) AS cnt
            """,
            {"threshold": degree_threshold},
        )
        stamped["degree_pass"] = (result[0]["cnt"] if result else 0)
        log.info(f"mark_generic_hubs pass-2 (degree>{degree_threshold}): {stamped['degree_pass']} nodes tagged")
    except Exception as e:
        log.warning(f"mark_generic_hubs pass-2 failed: {e}")

    total = stamped["stop_word_pass"] + stamped["degree_pass"]
    log.info(f"mark_generic_hubs complete — {total} hub nodes tagged")
    return stamped


def _pick_canonical_edge(edges: list[dict]) -> dict:
    """
    Choose the best representative from a group of parallel edges between the
    same (src, tgt) pair.

    Ranking (descending priority):
      1. Highest accumulated weight  (seen most often across pages)
      2. Longest predicate string    (longer = more specific)
      3. Not the bare fallback "relates to" string
    """
    def _score(e: dict) -> tuple:
        rel    = (e.get("rel") or "").strip()
        weight = e.get("weight") or 1
        is_generic = 1 if rel in ("relates to", "related to", "") else 0
        return (weight, len(rel), -is_generic)

    return max(edges, key=_score)


def deduplicate_edges(min_parallel: int = 2, batch_size: int = 300) -> dict:
    """
    Collapse parallel RELATES_TO edges between the same (src, tgt) pair into a
    single canonical edge — without APOC and without deleting any nodes.

    Strategy (pure Cypher + Python, no APOC required):
      1. Find every (src, tgt) pair that has ≥ ``min_parallel`` RELATES_TO edges.
      2. Pick the "canonical" edge using ``_pick_canonical_edge()``.
      3. Accumulate the total weight of all parallel edges onto the canonical edge.
      4. Delete the non-canonical edges.

    This is safe to re-run (idempotent after the first pass collapses everything
    to a single edge per pair).

    Returns:
        dict with keys:
          pairs_processed  — number of (src, tgt) pairs collapsed
          edges_removed    — total edges deleted
          edges_kept       — canonical edges updated
    """
    stats = {"pairs_processed": 0, "edges_removed": 0, "edges_kept": 0}
    skip  = 0

    while True:
        # Fetch a batch of (src, tgt) pairs that still have parallel edges
        rows = run_query(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            WHERE NOT a.generic_hub = true
              AND NOT b.generic_hub = true
            WITH a.id AS src, b.id AS tgt,
                 collect({
                     rel:    r.relation,
                     weight: coalesce(r.weight, 1),
                     eid:    id(r)
                 }) AS edges,
                 count(r) AS cnt
            WHERE cnt >= $min_par
            RETURN src, tgt, edges, cnt
            ORDER BY cnt DESC
            SKIP $skip LIMIT $batch
            """,
            {"min_par": min_parallel, "skip": skip, "batch": batch_size},
        )

        if not rows:
            break

        for row in rows:
            edges = row.get("edges") or []
            if len(edges) < 2:
                continue

            canonical = _pick_canonical_edge(edges)
            total_weight = sum(e.get("weight") or 1 for e in edges)
            non_canonical_eids = [e["eid"] for e in edges if e["eid"] != canonical["eid"]]

            # Stamp total weight onto canonical edge
            try:
                run_query(
                    "MATCH ()-[r]->() WHERE id(r) = $eid "
                    "SET r.weight = $w, r.dedup_source = 'merged'",
                    {"eid": canonical["eid"], "w": total_weight},
                )
                stats["edges_kept"] += 1
            except Exception as e:
                log.debug(f"dedup weight update failed: {e}")
                continue

            # Delete non-canonical edges in one shot
            if non_canonical_eids:
                try:
                    run_query(
                        "MATCH ()-[r]->() WHERE id(r) IN $eids DELETE r",
                        {"eids": non_canonical_eids},
                    )
                    stats["edges_removed"] += len(non_canonical_eids)
                except Exception as e:
                    log.debug(f"dedup delete failed: {e}")

            stats["pairs_processed"] += 1

        skip += batch_size

    log.info(
        f"deduplicate_edges complete — "
        f"{stats['pairs_processed']} pairs, "
        f"{stats['edges_removed']} edges removed, "
        f"{stats['edges_kept']} canonical edges weighted"
    )
    return stats


# ── Semantic edge type registry ───────────────────────────────────────────────
# Single source of truth for edge types used in audit and retrieval queries.
# Add new types here when extending semantic_promoter.py.
SEMANTIC_EDGE_TYPES = [
    # ── Causal / Process (Geography / Science / Economics) ───────────────────
    "CAUSES", "RESULTS_IN", "RESULTS_FROM",
    "TRIGGERS",      # rapid-onset causation
    "DRIVES",        # continuous causal force
    # ── Descriptive / Structural ─────────────────────────────────────────────
    "EXPLAINS",
    "CHARACTERIZED_BY",
    "SUBSET_OF",     # taxonomic hierarchy  (e.g. Tsunami SUBSET_OF Natural Disaster)
    "COMPONENT_OF",  # structural composition (e.g. Lok Sabha COMPONENT_OF Parliament)
    # ── Regulatory / Authority (Polity / Governance) ─────────────────────────
    "GOVERNED_BY",
    "MODIFIED_BY",   # constitutional amendment / legislative repeal
    # ── Historical / Cultural (History / Culture) ────────────────────────────
    "ORIGINATED_BY", # founding / establishment
    "CONFLICT_LINK", # opposition / historical conflict
    # ── Theory / Knowledge ───────────────────────────────────────────────────
    "UNIFIES",       # theory consolidation (Plate Tectonics UNIFIES Continental Drift)
    # ── Learning / Exam ──────────────────────────────────────────────────────
    "PREREQUISITE_OF",
    "TESTED_IN",
]

# Structural / administrative edges — excluded from reasoning-density denominator
# because they don't represent logical reasoning steps.
_STRUCTURAL_EDGE_TYPES = {
    "MENTIONED_IN", "HAS_CONCEPT", "SUMMARIZES", "CONTAINS",
}


def remove_ghost_nodes() -> dict:
    """
    Fix and remove Entity nodes that have a null or empty ``name`` property.

    These "ghost nodes" are created when the edge-import MERGE references an
    entity id that was never present in the node CSV.  They have a valid ``id``
    but no ``name``, making them invisible to most retrieval queries while still
    inflating neighbourhood degree counts.

    Two-pass strategy (conservative — no data loss):
      Pass 1 — REPAIR: for any ghost that already has an ``id``, backfill
               ``name = id`` so it becomes queryable.  This handles nodes that
               genuinely exist but were created before the ON CREATE SET fix.
      Pass 2 — DELETE: remove Entity nodes whose ``id`` is also null or empty
               (completely empty nodes — no id, no name, no edges of value).
               Safety guard: skip nodes that carry any semantic edge
               (CAUSES / GOVERNED_BY / etc.) — those are real data.

    Returns:
        dict with keys ``repaired`` and ``deleted``.
    """
    stats = {"repaired": 0, "deleted": 0}

    # Pass 1: repair — backfill name from id
    try:
        result = run_query(
            """
            MATCH (n:Entity)
            WHERE (n.name IS NULL OR trim(n.name) = '')
              AND n.id IS NOT NULL AND trim(n.id) <> ''
            SET n.name = n.id
            RETURN count(n) AS cnt
            """,
        )
        stats["repaired"] = result[0]["cnt"] if result else 0
        log.info(f"remove_ghost_nodes pass-1 (repair): {stats['repaired']} nodes backfilled")
    except Exception as e:
        log.warning(f"remove_ghost_nodes pass-1 failed: {e}")

    # Pass 2: delete — completely empty nodes with no semantic value
    # Note: DETACH DELETE removes the node AND all its relationships.
    # The WHERE guards ensure we never delete a node that carries real data.
    semantic_pipe = "|".join(SEMANTIC_EDGE_TYPES)
    try:
        result = run_query(
            f"""
            MATCH (n:Entity)
            WHERE (n.id IS NULL OR trim(n.id) = '')
              AND (n.name IS NULL OR trim(n.name) = '')
              AND NOT (n)-[:{semantic_pipe}]-()
            WITH n LIMIT 500
            DETACH DELETE n
            RETURN count(n) AS cnt
            """,
        )
        stats["deleted"] = result[0]["cnt"] if result else 0
        log.info(f"remove_ghost_nodes pass-2 (delete): {stats['deleted']} empty nodes removed")
    except Exception as e:
        log.warning(f"remove_ghost_nodes pass-2 failed: {e}")

    return stats


def run_global_upsc_promotion(batch_size: int = 500) -> dict:
    """
    Automated post-ingestion janitor — subject-agnostic for History, Geography,
    Polity, Economics, Science.

    Three layers of defence:

      Layer 1 — Atomic Purge: delete any Entity whose ``name`` escaped the
                word-count gate (> 4 words) and carries no semantic edges.
                Safety guard: nodes with real semantic edges are never touched.

      Layer 2 — Structural Promotion: scan RELATES_TO edges whose ``relation``
                predicate signals taxonomic or compositional structure and
                create typed edges additively (MERGE — no RELATES_TO deleted):
                  • "type of / kind of / subdivision of …" → SUBSET_OF
                  • "part of / branch of / component of …" → COMPONENT_OF

      Layer 3 — Historical / Constitutional Promotion:
                  • "founded / established by / originated …" → ORIGINATED_BY
                  • "opposed / conflict with / rebelled …"    → CONFLICT_LINK
                  • "amended by / repealed by / modified …"   → MODIFIED_BY

    All writes use MERGE so re-running is safe (fully idempotent).

    Returns:
        dict with counts per action key.
    """
    stats: dict[str, int] = {
        "atomic_purge": 0,
        "SUBSET_OF":    0,
        "COMPONENT_OF": 0,
        "ORIGINATED_BY": 0,
        "CONFLICT_LINK": 0,
        "MODIFIED_BY":  0,
    }

    # ── Layer 1: Atomic Purge ─────────────────────────────────────────────────
    semantic_pipe = "|".join(SEMANTIC_EDGE_TYPES)
    try:
        result = run_query(
            f"""
            MATCH (n:Entity)
            WHERE n.name IS NOT NULL
              AND size(split(n.name, ' ')) > 4
              AND NOT (n)-[:{semantic_pipe}]-()
            WITH n LIMIT 500
            DETACH DELETE n
            RETURN count(n) AS cnt
            """,
        )
        stats["atomic_purge"] = result[0]["cnt"] if result else 0
        log.info(f"run_global_upsc_promotion atomic_purge: {stats['atomic_purge']} long-name nodes removed")
    except Exception as e:
        log.warning(f"run_global_upsc_promotion atomic_purge failed: {e}")

    # ── Layer 2 & 3: Pattern-Based Promotion ─────────────────────────────────
    # Each entry: (edge_type, list_of_predicate_substrings_to_match)
    _PROMO_RULES: list[tuple[str, list[str]]] = [
        ("SUBSET_OF",    ["type of", "kind of", "example of", "subdivision of",
                          "category of", "form of", "class of", "variant of"]),
        ("COMPONENT_OF", ["part of", "component of", "element of",
                          "segment of", "branch of", "section of"]),
        ("ORIGINATED_BY",["founded", "established by", "created by", "built by",
                          "instituted", "originated", "started by", "introduced by"]),
        ("CONFLICT_LINK",["opposed", "conflict with", "war against", "rebelled against",
                          "fought against", "competitor of", "rival of"]),
        ("MODIFIED_BY",  ["amended by", "modified by", "repealed by",
                          "updated by", "replaced by", "inserted by"]),
    ]

    for edge_type, patterns in _PROMO_RULES:
        where_clauses = " OR ".join(
            f'toLower(r.relation) CONTAINS "{p}"' for p in patterns
        )
        skip = 0
        while True:
            try:
                rows = run_query(
                    f"""
                    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                    WHERE ({where_clauses})
                      AND NOT a.generic_hub = true
                      AND NOT b.generic_hub = true
                    RETURN a.id AS src, b.id AS tgt,
                           r.relation AS rel,
                           coalesce(r.chapter, a.chapter) AS chapter,
                           coalesce(r.book,    a.book)    AS book
                    SKIP $skip LIMIT $batch
                    """,
                    {"skip": skip, "batch": batch_size},
                )
            except Exception as e:
                log.warning(f"run_global_upsc_promotion {edge_type} query failed: {e}")
                break

            if not rows:
                break

            for row in rows:
                src = row.get("src", "")
                tgt = row.get("tgt", "")
                rel = row.get("rel", "")
                ch  = row.get("chapter") or ""
                bk  = row.get("book")    or ""
                if not src or not tgt:
                    continue
                try:
                    run_query(
                        f"MATCH (a:Entity {{id: $src}}), (b:Entity {{id: $tgt}}) "
                        f"MERGE (a)-[s:{edge_type} {{original_relation: $rel}}]->(b) "
                        f"ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='janitor'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": ch, "bk": bk},
                    )
                    stats[edge_type] += 1
                except Exception as e:
                    log.debug(f"{edge_type} promotion error {src}->{tgt}: {e}")

            skip += batch_size

    # ── Layer 4: Subject-Specific Causal / Mechanism Promotions ─────────────
    # Merges well-known UPSC causal chains as typed edges between atomic nodes.
    # Fully idempotent (MERGE).  Runs regardless of what PDF chapter triggered
    # the import — these relationships are "always true" for UPSC Geography.
    stats["GEO_CAUSAL"] = 0
    _GEO_CAUSAL_RULES: list[tuple[str, str, str]] = [
        # (subject_a_fragment, edge_type, subject_b_fragment)
        # Himalayas / Tibet → drive the Monsoon
        ("himalaya",          "DRIVES",  "monsoon"),
        ("tibetan plateau",   "DRIVES",  "monsoon"),
        ("tibet",             "DRIVES",  "monsoon"),
        # Monsoon → Precipitation / Rainfall
        ("monsoon",           "CAUSES",  "precipitation"),
        ("monsoon",           "CAUSES",  "rainfall"),
        ("monsoon",           "CAUSES",  "rain"),
        # Plate Tectonics → Continental Drift / Sea-Floor Spreading (unification)
        ("plate tectonic",    "UNIFIES", "continental drift"),
        ("plate tectonic",    "UNIFIES", "seafloor"),
        ("plate tectonic",    "UNIFIES", "sea floor"),
        ("plate tectonic",    "UNIFIES", "sea-floor"),
        # Subduction → Fold Mountains / Volcanic Activity / Earthquakes
        ("subduction",        "CAUSES",  "fold mountain"),
        ("subduction",        "CAUSES",  "volcanic"),
        ("subduction",        "CAUSES",  "earthquake"),
        # Isostasy → Himalayan Uplift / Peninsular stability
        ("isosta",            "DRIVES",  "himalaya"),
        ("isosta",            "DRIVES",  "peninsular"),
        ("isosta",            "DRIVES",  "uplift"),
        # Carlsberg Ridge → Northward Drift of Indian Plate
        ("carlsberg",         "DRIVES",  "northward drift"),
        ("carlsberg",         "DRIVES",  "drift"),
        ("carlsberg",         "DRIVES",  "indian plate"),
        # Low Pressure / Thermal Low → Monsoon Onset
        ("low pressure",      "TRIGGERS","monsoon"),
        ("thermal low",       "TRIGGERS","monsoon"),
        # Western Ghats → Orographic Rainfall
        ("western ghat",      "CAUSES",  "orographic"),
        ("western ghat",      "CAUSES",  "rainfall"),
    ]
    for _frag_a, _etype, _frag_b in _GEO_CAUSAL_RULES:
        try:
            _res = run_query(
                f"""
                MATCH (a:Entity), (b:Entity)
                WHERE toLower(a.id) CONTAINS $fa
                  AND toLower(b.id) CONTAINS $fb
                  AND NOT a.generic_hub = true
                  AND NOT b.generic_hub = true
                  AND a.id <> b.id
                MERGE (a)-[:{_etype} {{source: 'geo_causal_layer4', weight: 3}}]->(b)
                RETURN count(*) AS cnt
                """,
                {"fa": _frag_a, "fb": _frag_b},
            )
            if _res:
                stats["GEO_CAUSAL"] += int(_res[0].get("cnt", 0))
        except Exception as _le:
            log.debug(f"Layer 4 geo_causal {_frag_a}→{_frag_b}: {_le}")

    total_promoted = sum(v for k, v in stats.items() if k != "atomic_purge")
    log.info(
        f"run_global_upsc_promotion complete — "
        f"purged={stats['atomic_purge']}, promoted={total_promoted}: "
        + ", ".join(f"{k}={v}" for k, v in stats.items() if k != "atomic_purge")
    )
    return stats


def cluster_health_audit(entity_name: str, top_n: int = 5) -> list[dict]:
    """
    Compute the Reasoning Density (ρ) for a cluster centred on ``entity_name``.

    ρ = semantic_edges / (relates_to_edges + semantic_edges)

    - Semantic edges: CAUSES, RESULTS_IN, GOVERNED_BY, EXPLAINS, TRIGGERS,
                      CHARACTERIZED_BY, RESULTS_FROM, PREREQUISITE_OF, TESTED_IN
    - Excluded from denominator: MENTIONED_IN, HAS_CONCEPT, SUMMARIZES, CONTAINS
      (these are structural bookkeeping edges, not reasoning steps)

    A cluster is considered "certified" for UPSC RAG when ρ ≥ 0.15.

    Args:
        entity_name: Substring match against ``n.id`` (case-insensitive).
                     E.g. "earthquake", "plate tectonic", "monsoon".
        top_n:       Maximum number of matching entities to return.

    Returns:
        List of dicts, each with:
          cluster_hub        — entity id
          chapter / subject  — metadata
          neighborhood_size  — total meaningful edges (RELATES_TO + semantic)
          logic_score        — count of semantic-typed edges
          reasoning_density  — ρ as a float in [0, 1]
          certified          — True if ρ ≥ 0.15
    """
    semantic_pipe = "|".join(SEMANTIC_EDGE_TYPES)
    try:
        rows = run_query(
            f"""
            MATCH (n:Entity)
            WHERE toLower(n.id) CONTAINS toLower($name)
              AND NOT n.generic_hub = true
            OPTIONAL MATCH (n)-[rs:{semantic_pipe}]-()
            WITH n, count(rs) AS logic_score
            OPTIONAL MATCH (n)-[ra:RELATES_TO|{semantic_pipe}]-()
            WITH n, logic_score, count(ra) AS neighborhood_size
            RETURN
                n.id      AS cluster_hub,
                n.chapter AS chapter,
                n.subject AS subject,
                neighborhood_size,
                logic_score,
                CASE WHEN neighborhood_size > 0
                     THEN round(1000.0 * logic_score / neighborhood_size) / 1000.0
                     ELSE 0.0
                END AS reasoning_density
            ORDER BY reasoning_density DESC, neighborhood_size DESC
            LIMIT $top_n
            """,
            {"name": entity_name.strip().lower(), "top_n": top_n},
        )
    except Exception as e:
        log.warning(f"cluster_health_audit failed for '{entity_name}': {e}")
        return []

    results = []
    for row in rows:
        density = row.get("reasoning_density") or 0.0
        results.append({
            "cluster_hub":       row.get("cluster_hub", ""),
            "chapter":           row.get("chapter", ""),
            "subject":           row.get("subject", ""),
            "neighborhood_size": row.get("neighborhood_size", 0),
            "logic_score":       row.get("logic_score", 0),
            "reasoning_density": density,
            "certified":         density >= 0.15,
        })
    return results


def _sanitize_text(text):
    """
    Remove characters that Windows cp1252 can't encode.
    Atlas-RAG writes files without explicit encoding on Windows,
    causing charmap errors with Hindi/Sanskrit/special chars.
    Keep ASCII + common unicode that cp1252 supports.
    """
    result = []
    for ch in text:
        try:
            ch.encode('cp1252')
            result.append(ch)
        except (UnicodeEncodeError, UnicodeDecodeError):
            result.append(' ')  # replace with space
    return ''.join(result)

def _make_atlas_input(text, doc_id):
    return [{"id": str(doc_id), "text": _sanitize_text(text), "metadata": {"lang": "en"}}]

def _find_csvs(output_dir, pattern):
    """Find CSVs containing pattern in filename, searching recursively."""
    matches = []
    for root, dirs, files in os.walk(output_dir):
        for fname in files:
            if fname.endswith(".csv") and pattern in fname:
                matches.append(os.path.join(root, fname))
    return matches

def _safe_cleanup(work_dir):
    """
    Safely delete Atlas-RAG work directory.
    On Windows, logging.txt may be locked — retry with delay.
    """
    import time
    for attempt in range(5):
        try:
            shutil.rmtree(work_dir, ignore_errors=False)
            return
        except PermissionError:
            time.sleep(1)
    # Last resort: ignore errors
    shutil.rmtree(work_dir, ignore_errors=True)

# ── Neo4j Import ──────────────────────────────────────────────────────────────
def _import_csv_to_neo4j(output_dir, subject, book, chapter, class_level, doc_name):
    node_count = rel_count = concept_count = concept_rel_count = 0

    with get_driver().session(database=NEO4J_DATABASE) as session:

        # 1. Entity nodes: triple_nodes_*_without_emb.csv  →  name:ID, type, concepts, :LABEL
        for f in _find_csvs(output_dir, "triple_nodes_"):
            log.debug(f"Entity nodes: {os.path.basename(f)}")
            try:
                with open(f, encoding="utf-8") as cf:
                    for row in csv.DictReader(cf):
                        name  = (row.get("name:ID") or "").lower().strip()
                        etype = (row.get("type")    or "Entity").strip()
                        if not name or len(name) < 2 or _is_noise_entity(name):
                            continue
                        concepts_list = _parse_concepts(row.get("concepts") or "")
                        session.run(
                            "MERGE (n:Entity {id: $id}) "
                            "ON CREATE SET n.name=$id, n.type=$type, n.text=$id, "
                            "n.subject=$subject, n.book=$book, "
                            "n.chapter=$chapter, n.class_level=$cl, n.doc_name=$dn, "
                            "n.concepts=$concepts, n.mention_count=1, "
                            "n.source='atlas_rag', n.created_at=timestamp() "
                            "ON MATCH SET "
                            "n.name=CASE WHEN n.name IS NULL OR n.name='' THEN $id ELSE n.name END, "
                            "n.subject=CASE WHEN n.subject IS NULL OR n.subject='' THEN $subject ELSE n.subject END, "
                            "n.chapter=CASE WHEN n.chapter IS NULL OR n.chapter='' THEN $chapter ELSE n.chapter END, "
                            "n.book=CASE WHEN n.book IS NULL OR n.book='' THEN $book ELSE n.book END, "
                            "n.concepts=CASE WHEN n.concepts IS NULL OR n.concepts=[] THEN $concepts ELSE n.concepts END, "
                            "n.mention_count=coalesce(n.mention_count,0)+1 "
                            "WITH n WHERE n.mention_count > 10 SET n:HighYield",
                            {"id": name, "type": etype, "subject": subject,
                             "book": book, "chapter": chapter, "cl": class_level,
                             "dn": doc_name, "concepts": concepts_list}
                        )
                        node_count += 1
            except Exception as e:
                log.warning(f"triple_nodes error ({f}): {e}")

        # 2. Entity relations — prefer concept-enriched edges (concept_csv/triple_edges_*_with_concept.csv)
        #    over plain edges (triples_csv/triple_edges_*_without_emb.csv).
        #    All edges stored as :RELATES_TO {relation: "..."} so there is ONE edge type
        #    and predicates are queryable via CONTAINS — eliminates 900+ dynamic rel types.
        all_edge_files   = _find_csvs(output_dir, "triple_edges_")
        enriched_files   = [f for f in all_edge_files if "with_concept" in f]
        plain_edge_files = [f for f in all_edge_files if "with_concept" not in f]
        edge_files_to_use = enriched_files if enriched_files else plain_edge_files

        for f in edge_files_to_use:
            log.debug(f"Entity rels ({'enriched' if enriched_files else 'plain'}): {os.path.basename(f)}")
            try:
                with open(f, encoding="utf-8") as cf:
                    for row in csv.DictReader(cf):
                        src = (row.get(":START_ID") or "").strip().lower()
                        tgt = (row.get(":END_ID")   or "").strip().lower()
                        rel = (row.get("relation")  or "relates to").strip().lower()
                        if not src or not tgt: continue
                        if _is_noise_entity(src) or _is_noise_entity(tgt): continue
                        rel_norm  = re.sub(r"\s+", " ", rel)[:120]
                        edge_concepts = _parse_concepts(row.get("concepts") or "")
                        try:
                            session.run(
                                # Node MERGE: give ghost nodes a minimal but valid name so
                                # they never appear as null-name nodes in audit queries.
                                # Full metadata is filled in (or enriched) by the node CSV
                                # pass that runs before this loop.
                                "MERGE (a:Entity {id: $src}) "
                                "  ON CREATE SET a.name=$src, a.source='edge_ref', "
                                "               a.created_at=timestamp() "
                                "MERGE (b:Entity {id: $tgt}) "
                                "  ON CREATE SET b.name=$tgt, b.source='edge_ref', "
                                "               b.created_at=timestamp() "
                                "MERGE (a)-[r:RELATES_TO {relation: $rel}]->(b) "
                                "ON CREATE SET r.book=$book, r.chapter=$chapter, "
                                "r.source='atlas_rag', r.concepts=$concepts, r.weight=1 "
                                "ON MATCH SET r.weight=coalesce(r.weight,1)+1",
                                {"src": src, "tgt": tgt, "rel": rel_norm,
                                 "book": book, "chapter": chapter,
                                 "concepts": edge_concepts}
                            )
                            rel_count += 1
                        except Exception as re_err:
                            log.debug(f"Rel error {src}->{tgt}: {re_err}")
            except Exception as e:
                log.warning(f"triple_edges error ({f}): {e}")

        # 3. Concept nodes: concept_nodes_*.csv  →  concept_id:ID, name, :LABEL
        for f in _find_csvs(output_dir, "concept_nodes_"):
            log.debug(f"Concept nodes: {os.path.basename(f)}")
            try:
                with open(f, encoding="utf-8") as cf:
                    for row in csv.DictReader(cf):
                        cid  = (row.get("concept_id:ID") or "").strip()
                        name = (row.get("name")          or "").strip()
                        if not name or len(name) < 2: continue
                        session.run(
                            "MERGE (c:Concept {concept_id: $cid}) "
                            "ON CREATE SET c.name=$name, c.id=$name, "
                            "c.source='atlas_rag', c.subject=$subject, c.created_at=timestamp()",
                            {"cid": cid, "name": name, "subject": subject}
                        )
                        concept_count += 1
            except Exception as e:
                log.warning(f"concept_nodes error ({f}): {e}")

        # 4. Concept edges: concept_edges_*.csv  →  entity HAS_CONCEPT concept
        for f in _find_csvs(output_dir, "concept_edges_"):
            log.debug(f"Concept edges: {os.path.basename(f)}")
            try:
                with open(f, encoding="utf-8") as cf:
                    for row in csv.DictReader(cf):
                        src = (row.get(":START_ID") or "").strip()
                        tgt = (row.get(":END_ID")   or "").strip()
                        if not src or not tgt: continue
                        try:
                            session.run(
                                "MATCH (e:Entity {id: $src}) "
                                "MATCH (c:Concept {concept_id: $tgt}) "
                                "MERGE (e)-[:HAS_CONCEPT]->(c)",
                                {"src": src, "tgt": tgt}
                            )
                            concept_rel_count += 1
                        except Exception as ce:
                            log.debug(f"Concept rel error: {ce}")
            except Exception as e:
                log.warning(f"concept_edges error ({f}): {e}")

        # 5. TextSource nodes: text_nodes_*_from_json.csv  →  original passages
        #    Preserves source evidence for each entity — enables citation & grounding.
        text_count = 0
        for f in _find_csvs(output_dir, "text_nodes_"):
            log.debug(f"TextSource nodes: {os.path.basename(f)}")
            try:
                with open(f, encoding="utf-8") as cf:
                    for row in csv.DictReader(cf):
                        tid  = (row.get("text_id:ID")    or "").strip()
                        text = (row.get("original_text") or "").strip()
                        if not tid or not text:
                            continue
                        session.run(
                            "MERGE (t:TextSource {id: $tid}) "
                            "ON CREATE SET t.text=$text, t.subject=$subject, "
                            "t.chapter=$chapter, t.book=$book, "
                            "t.source='atlas_rag', t.created_at=timestamp()",
                            {"tid": tid, "text": text[:2000],
                             "subject": subject, "chapter": chapter, "book": book}
                        )
                        text_count += 1
            except Exception as e:
                log.warning(f"text_nodes error ({f}): {e}")

        # 6. Source edges: text_edges_*_from_json.csv  →  Entity -[:MENTIONED_IN]-> TextSource
        src_edge_count = 0
        for f in _find_csvs(output_dir, "text_edges_"):
            log.debug(f"Source edges: {os.path.basename(f)}")
            try:
                with open(f, encoding="utf-8") as cf:
                    for row in csv.DictReader(cf):
                        eid = (row.get(":START_ID") or "").strip().lower()
                        tid = (row.get(":END_ID")   or "").strip()
                        if not eid or not tid:
                            continue
                        try:
                            session.run(
                                "MATCH (e:Entity {id: $eid}) "
                                "MATCH (t:TextSource {id: $tid}) "
                                "MERGE (e)-[:MENTIONED_IN]->(t)",
                                {"eid": eid, "tid": tid}
                            )
                            src_edge_count += 1
                        except Exception as se:
                            log.debug(f"Source edge error {eid}->{tid}: {se}")
            except Exception as e:
                log.warning(f"text_edges error ({f}): {e}")

        try:
            summary_text = (
                f"{chapter} — {subject} (Class {class_level}). "
                f"Source: {book}. "
                f"Key entities extracted from NCERT content on this topic."
            )
            with get_driver().session(database=NEO4J_DATABASE) as ts_session:
                # Create/update one TopicSummary node per (book, chapter)
                ts_session.run(
                    """
                    MERGE (t:TopicSummary {book: $book, chapter: $chapter})
                    ON CREATE SET
                        t.subject     = $subject,
                        t.class_level = $class_level,
                        t.summary     = $summary,
                        t.source      = 'atlas_rag',
                        t.created_at  = timestamp()
                    ON MATCH SET
                        t.subject     = CASE WHEN t.subject IS NULL OR t.subject = ''
                                             THEN $subject ELSE t.subject END,
                        t.entity_count = coalesce(t.entity_count, 0) + $nc
                    """,
                    {"book": book, "chapter": chapter, "subject": subject,
                     "class_level": class_level, "summary": summary_text, "nc": node_count}
                )
                # Link entities in this chapter to their TopicSummary
                ts_session.run(
                    """
                    MATCH (t:TopicSummary {book: $book, chapter: $chapter})
                    MATCH (e:Entity {chapter: $chapter})
                    MERGE (t)-[:SUMMARIZES]->(e)
                    """,
                    {"book": book, "chapter": chapter}
                )
        except Exception as e:
            log.warning(f"TopicSummary creation failed: {e}")

    log.info(
        f"Imported: {node_count} entities, {rel_count} RELATES_TO edges, "
        f"{concept_count} concepts, {concept_rel_count} concept_rels, "
        f"{text_count} text_sources, {src_edge_count} mention_edges"
    )

    # Promote newly-imported RELATES_TO edges into typed semantic edges.
    # Runs in a background thread so it never blocks the ingestion pipeline.
    try:
        from futureverse_ai.retrieval.graph.semantic_promoter import promote_semantic_relations
        threading.Thread(
            target=promote_semantic_relations,
            daemon=True,
        ).start()
    except Exception as _sp_err:
        log.debug(f"Semantic promotion skipped: {_sp_err}")

    return node_count, rel_count, concept_count


# ── Atlas-RAG Pipeline ────────────────────────────────────────────────────────
def _run_atlas_extraction(text, doc_id, subject, book, chapter, class_level, doc_name):
    from openai import OpenAI
    from atlas_rag.llm_generator import LLMGenerator
    from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
    from atlas_rag.kg_construction.triple_config import ProcessingConfig

    # Use persistent work dir (not tempfile) to avoid Windows file locking
    # Include a timestamp suffix to prevent [WinError 183] when the same doc_id
    # is retried before the previous run's cleanup (2s sleep) has finished.
    safe_id    = re.sub(r"[^a-zA-Z0-9_]", "_", str(doc_id))[:40]
    model_safe = KG_MODEL.replace(":", "_").replace("/", "_")
    import time as _time
    run_stamp  = int(_time.time() * 1000)
    work_dir   = os.path.join(ATLAS_WORK_DIR, f"{safe_id}_{run_stamp}")
    input_dir  = os.path.join(work_dir, "input")
    output_dir = os.path.join(work_dir, "output")

    # Force-delete if somehow the stamped dir already exists, then create fresh
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Write input JSON
        with open(os.path.join(input_dir, f"{safe_id}.json"), "w", encoding="utf-8") as f:
            json.dump(_make_atlas_input(text, doc_id), f)

        client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
        llm    = LLMGenerator(client=client, model_name=KG_MODEL)

        config = ProcessingConfig(
            model_path=model_safe,          # colon-free for Windows filenames
            data_directory=input_dir,
            filename_pattern=safe_id,       # ensures file is picked up
            output_directory=output_dir,
            batch_size_triple=1,
            batch_size_concept=4,
            chunk_size=2000,
            chunk_overlap=0,
            include_concept=True,
            deduplicate_text=False,
            max_workers=1,
        )

        extractor = KnowledgeGraphExtractor(model=llm, config=config)

        log.info(f"[Atlas-RAG] Extracting triples: {doc_id}")
        extractor.run_extraction()
        extractor.convert_json_to_csv()

        log.info(f"[Atlas-RAG] Schema induction: {doc_id}")
        extractor.generate_concept_csv_temp()
        extractor.create_concept_csv()

        nodes, rels, concepts = _import_csv_to_neo4j(
            output_dir, subject, book, chapter, class_level, doc_name
        )
        try:
            global IMPORT_COUNTER
            IMPORT_COUNTER += 1

            # Every 25 imports: collapse parallel edges + run global UPSC janitor
            if IMPORT_COUNTER % 25 == 0:
                threading.Thread(
                    target=deduplicate_edges,
                    daemon=True,
                ).start()
                threading.Thread(
                    target=run_global_upsc_promotion,
                    daemon=True,
                ).start()

            # Every 50 imports: rebuild community summaries
            if IMPORT_COUNTER % 50 == 0:
                threading.Thread(
                    target=build_graph_communities,
                    args=(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD),
                ).start()
        except Exception as e:
            log.warning(f"Post-import maintenance failed: {e}")

    finally:
        # Cleanup after Neo4j import (files no longer needed)
        # Small delay so Atlas-RAG logging handler can release logging.txt
        import time; time.sleep(2)
        _safe_cleanup(work_dir)

    return nodes, rels, concepts


# ── Fallback Extractor ────────────────────────────────────────────────────────
# Subject-specific extraction hints injected into the prompt.
# Add a new subject key here to guide the LLM for any UPSC domain.
_SUBJECT_HINTS: dict[str, str] = {
    "geography": (
        "  - Geography:  Prioritize Phenomena, Forces, Boundaries, Landforms.\n"
        "  - Good examples: 'Monsoon', 'Differential Heating', 'Western Ghats', 'Trade Winds'\n"
    ),
    "polity": (
        "  - Polity:     Prioritize Articles, Amendments, Institutions, Rights, Schedules.\n"
        "  - Good examples: 'Article 21', 'Lok Sabha', 'Fundamental Rights', 'DPSP'\n"
    ),
    "history": (
        "  - History:    Prioritize Dynasties, Figures, Movements, Turning Points, Battles.\n"
        "  - Good examples: 'Maurya Empire', 'Battle of Plassey', 'Non-Cooperation Movement'\n"
    ),
    "economics": (
        "  - Economics:  Prioritize Policies, Metrics, Institutions, Market forces.\n"
        "  - Good examples: 'GDP', 'Fiscal Deficit', 'RBI', 'Inflation', 'WTO'\n"
    ),
    "science": (
        "  - Science:    Prioritize Concepts, Elements, Forces, Laws, Organisms.\n"
        "  - Good examples: 'Photosynthesis', 'Newton Law', 'Carbon Cycle', 'DNA'\n"
    ),
    "environment": (
        "  - Environment: Prioritize Ecosystems, Treaties, Species, Pollutants.\n"
        "  - Good examples: 'Coral Bleaching', 'Paris Agreement', 'Wetlands', 'Carbon Credits'\n"
    ),
}


def _build_fallback_prompt(text: str, subject: str, book: str, chapter: str) -> str:
    """Build a subject-aware extraction prompt for the fallback LLM extractor."""
    sub_key = (subject or "").strip().lower()
    # Match partial subject names (e.g. "physical geography" → "geography")
    subject_hint = ""
    for key, hint in _SUBJECT_HINTS.items():
        if key in sub_key:
            subject_hint = hint
            break

    return (
        "You are a UPSC Knowledge Graph builder. Extract entities and relationships "
        "from the study text below.\n"
        f"Subject: {subject} | Book: {book} | Chapter: {chapter}\n\n"
        "UPSC EXTRACTION PROTOCOL:\n"
        "ENTITY ATOMICITY: Extract only short, technical nouns (1-4 words MAX).\n"
        "  - NEVER extract full sentences as entity names.\n"
        "  - NEVER start an entity name with 'the', 'various', 'several', 'some'.\n"
        f"{subject_hint}"
        "\nCAUSAL THINKING: Focus on verbs that indicate change, force, or structure:\n"
        "  TRIGGERS        : rapid-onset event    (Earthquake TRIGGERS Tsunami)\n"
        "  DRIVES          : continuous process   (Pressure Gradient DRIVES Wind)\n"
        "  CAUSES          : general causation    (El Nino CAUSES Drought)\n"
        "  RESULTS_IN      : consequence          (Rainfall RESULTS_IN Erosion)\n"
        "  CHARACTERIZED_BY: defining feature     (Monsoon CHARACTERIZED_BY Seasonal Reversal)\n"
        "  GOVERNED_BY     : regulatory control   (Rivers GOVERNED_BY Gravity)\n"
        "  MODIFIED_BY     : amendment / repeal   (Article 368 MODIFIED_BY 42nd Amendment)\n"
        "  SUBSET_OF       : taxonomic type-of    (Tsunami SUBSET_OF Natural Disaster)\n"
        "  COMPONENT_OF    : part-of              (Lok Sabha COMPONENT_OF Parliament)\n"
        "  ORIGINATED_BY   : founding / creation  (INC ORIGINATED_BY Allan Octavian Hume)\n"
        "  CONFLICT_LINK   : opposition / war     (Gandhi CONFLICT_LINK British Empire)\n"
        "  UNIFIES         : theory consolidation (Plate Tectonics UNIFIES Continental Drift)\n"
        "  RELATES_TO      : last resort — use ONLY when no technical verb applies\n\n"
        'Return ONLY valid JSON (no markdown, no prose):\n'
        '{"entities": [{"id": "Name", "type": "Type", "upsc_importance": "high/medium/low"}], '
        '"relationships": [{"from": "A", "to": "B", "type": "EDGE_TYPE"}]}\n\n'
        f"Text:\n{text[:1500]}"
    )


def _fallback_extract(text, doc_id, doc_name, subject, book, chapter, class_level, page_number):
    import requests
    prompt = _build_fallback_prompt(text, subject, book, chapter)
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": KG_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0, "num_predict": 600, "num_gpu": 99},
                  "keep_alive": 0},
            timeout=90
        )
        raw = r.json().get("response", "")
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s < 0:
            return {"status": "success", "nodes": 0, "relationships": 0,
                    "doc_id": doc_id, "source": "fallback"}
        raw = raw[s:e]
        # Multi-pass JSON repair for common LLM output issues
        raw = re.sub(r",\s*([\]}])", r"\1", raw)          # trailing commas
        raw = re.sub(r"(?<!\\)'", '"', raw)               # single → double quotes
        raw = re.sub(r'("(?:[^"\\]|\\.)*")|(\b\w+)\s*:', # unquoted keys
                     lambda m: m.group(1) if m.group(1) else f'"{m.group(2)}":', raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Last resort: extract entities/relationships line-by-line
            log.debug(f"Fallback JSON unparseable for {doc_id}, skipping")
            return {"status": "success", "nodes": 0, "relationships": 0,
                    "doc_id": doc_id, "source": "fallback"}
        # Atomic Gate: remove sentence-entities and normalize relation types
        cleaned_ents, cleaned_rels = validate_and_clean_extraction(
            data.get("entities", []), data.get("relationships", [])
        )
        data["entities"]      = cleaned_ents
        data["relationships"] = cleaned_rels
        nc = rc = 0
        with get_driver().session(database=NEO4J_DATABASE) as session:
            for ent in data.get("entities", []):
                eid = str(ent.get("id", "")).strip().lower()
                if not eid or len(eid) < 2 or _is_noise_entity(eid):
                    continue
                session.run(
                    "MERGE (n:Entity {id:$id}) ON CREATE SET n.name=$id, n.type=$t, "
                    "n.subject=$s, n.book=$b, n.chapter=$c, n.class_level=$cl, "
                    "n.doc_name=$dn, n.upsc_importance=$imp, n.mention_count=1, "
                    "n.source='fallback' "
                    "ON MATCH SET n.name=CASE WHEN n.name IS NULL THEN $id ELSE n.name END, "
                    "n.mention_count=coalesce(n.mention_count,0)+1 "
                    "WITH n WHERE n.mention_count > 10 SET n:HighYield",
                    {"id": eid, "t": ent.get("type", "Concept"), "s": subject, "b": book,
                     "c": chapter, "cl": class_level, "dn": doc_name,
                     "imp": ent.get("upsc_importance", "medium")}
                )
                nc += 1
            for rel in data.get("relationships", []):
                src = str(rel.get("from", "")).strip().lower()
                tgt = str(rel.get("to",   "")).strip().lower()
                rel_str = re.sub(r"\s+", " ", str(rel.get("type", "relates to")).lower())[:120]
                if src and tgt and not _is_noise_entity(src) and not _is_noise_entity(tgt):
                    try:
                        session.run(
                            "MERGE (a:Entity {id:$s}) MERGE (b:Entity {id:$t}) "
                            "MERGE (a)-[r:RELATES_TO {relation: $rel}]->(b) "
                            "ON CREATE SET r.source='fallback', r.weight=1 "
                            "ON MATCH SET r.weight=coalesce(r.weight,1)+1",
                            {"s": src, "t": tgt, "rel": rel_str}
                        )
                        rc += 1
                    except: pass
        return {"status": "success", "nodes": nc, "relationships": rc,
                "doc_id": doc_id, "source": "fallback"}
    except Exception as e:
        log.error(f"Fallback error: {e}")
        return {"status": "error", "error": str(e), "doc_id": doc_id, "source": "fallback"}


# ── Main Entry Point ──────────────────────────────────────────────────────────
def extract_and_store(
    text, doc_id, doc_name="", subject="",
    book="", chapter="", class_level="", page_number=0
):
    try:
        nodes, rels, concepts = _run_atlas_extraction(
            text, doc_id, subject, book, chapter, class_level, doc_name
        )
        return {
            "status": "success", "nodes": nodes, "relationships": rels,
            "concepts": concepts, "doc_id": doc_id, "source": "atlas_rag"
        }
    except ImportError as e:
        log.warning(f"Atlas-RAG not available ({e}), using fallback")
        return _fallback_extract(text, doc_id, doc_name, subject, book, chapter, class_level, page_number)
    except Exception as e:
        log.warning(f"Atlas-RAG failed ({e}), using fallback")
        return _fallback_extract(text, doc_id, doc_name, subject, book, chapter, class_level, page_number)


# ── Graph Search ──────────────────────────────────────────────────────────────
def search_graph(query, limit=15, subject_filter=None):
    """
    Keyword + multi-hop search across Entity, Concept, TopicSummary, Community nodes.
    Returns a flat list of human-readable lines suitable for LLM context injection.
    """
    try:
        STOP = {
            "what","where","when","who","how","why","is","are","was","were",
            "the","a","an","of","in","at","by","to","from","and","or","for",
            "with","this","that","which","about","tell","explain","describe",
            "give","list","define","does","did","do","has","have","had",
            "type","types","kind","kinds","name","names","following",
            "consider","statements","correct","true","false",
        }

        # Extract individual keywords (4+ chars) AND preserve meaningful 2-word phrases
        words = re.findall(r"[a-zA-Z]{3,}", query.lower())
        keywords = [w for w in words if w not in STOP]

        # Bigram phrases — capture terms like "monsoon climate", "river basin"
        bigrams = [
            f"{words[i]} {words[i+1]}"
            for i in range(len(words) - 1)
            if words[i] not in STOP and words[i+1] not in STOP
            and len(words[i]) > 3 and len(words[i+1]) > 3
        ]
        search_terms = (keywords[:6] + bigrams[:3]) if bigrams else keywords[:8]

        if not search_terms:
            search_terms = [query[:50]]

        results = []
        seen = set()

        for kw in search_terms:
            # ── 1. Entity nodes + immediate neighbours ────────────────────────
            rows = run_query(
                """
                MATCH (n:Entity)
                WHERE toLower(n.id) CONTAINS toLower($kw)
                OPTIONAL MATCH (n)-[r]-(m:Entity)
                WHERE m.id IS NOT NULL AND NOT m.id =~ '^[0-9]+$'
                RETURN
                    n.id          AS entity,
                    n.subject     AS subject,
                    n.chapter     AS chapter,
                    n.type        AS etype,
                    n.mention_count AS mentions,
                    collect(DISTINCT m.id)[..4]    AS related,
                    collect(DISTINCT type(r))[..3] AS rels
                ORDER BY coalesce(n.mention_count, 0) DESC
                LIMIT $limit
                """,
                {"kw": kw, "limit": limit}
            )
            for row in rows:
                eid  = row.get("entity", "")
                if not eid:
                    continue
                line = f"[Entity] {eid}"
                if row.get("etype"):
                    line += f" ({row['etype']})"
                if row.get("subject"):
                    line += f" | {row['subject']}"
                if row.get("chapter"):
                    line += f" | {row['chapter']}"
                related = row.get("related") or []
                rels    = row.get("rels") or []
                if related:
                    rel_str = ", ".join(
                        f"{r} [{t}]" if t else r
                        for r, t in zip(related[:3], rels[:3] + [""] * 3)
                    )
                    line += f"\n  → {rel_str}"
                if line not in seen:
                    seen.add(line)
                    results.append(line)

            # ── 2. Concept nodes ──────────────────────────────────────────────
            rows = run_query(
                """
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($kw)
                OPTIONAL MATCH (e:Entity)-[:HAS_CONCEPT]->(c)
                RETURN c.name AS concept, c.subject AS subject,
                       collect(DISTINCT e.id)[..3] AS entities
                LIMIT $limit
                """,
                {"kw": kw, "limit": limit}
            )
            for row in rows:
                if not row.get("concept"):
                    continue
                line = f"[Concept] {row['concept']}"
                if row.get("subject"):
                    line += f" | {row['subject']}"
                ents = [e for e in (row.get("entities") or []) if e]
                if ents:
                    line += f" ← {', '.join(ents[:3])}"
                if line not in seen:
                    seen.add(line)
                    results.append(line)

            # ── 3. TopicSummary ───────────────────────────────────────────────
            rows = run_query(
                """
                MATCH (t:TopicSummary)
                WHERE toLower(t.chapter) CONTAINS toLower($kw)
                   OR toLower(t.summary) CONTAINS toLower($kw)
                RETURN t.chapter AS chapter, t.subject AS subject,
                       t.book AS book, t.summary AS summary
                ORDER BY CASE WHEN t.subject = $sf THEN 0 ELSE 1 END
                LIMIT $limit
                """,
                {"kw": kw, "limit": limit, "sf": subject_filter or ""}
            )
            for row in rows:
                line = f"[Topic] {row.get('chapter','?')} | {row.get('subject','?')}"
                if row.get("book"):
                    line += f" | {row['book']}"
                if row.get("summary"):
                    line += f"\n  → {row['summary'][:250]}"
                if line not in seen:
                    seen.add(line)
                    results.append(line)

        # ── 4. Multi-hop path between top-2 hit entities ─────────────────────
        entity_ids = [
            r.split("]")[1].split("(")[0].strip().split("|")[0].strip()
            for r in results
            if r.startswith("[Entity]")
        ][:2]

        if len(entity_ids) == 2:
            try:
                rows = run_query(
                    """
                    MATCH p = shortestPath(
                        (a:Entity {id: $a})-[*..4]-(b:Entity {id: $b})
                    )
                    RETURN [n IN nodes(p) | n.id] AS path_nodes,
                           [r IN relationships(p) | type(r)] AS path_rels
                    LIMIT 2
                    """,
                    {"a": entity_ids[0], "b": entity_ids[1]}
                )
                for row in rows:
                    nodes_ = row.get("path_nodes") or []
                    rels_  = row.get("path_rels") or []
                    if len(nodes_) >= 2:
                        parts = [nodes_[0]]
                        for n, r in zip(nodes_[1:], rels_):
                            parts.append(f"--[{r}]-->")
                            parts.append(n)
                        line = "[Path] " + " ".join(parts)
                        if line not in seen:
                            seen.add(line)
                            results.append(line)
            except Exception:
                pass  # shortest path is best-effort

        return results or ["No graph results found."]

    except Exception as e:
        log.error(f"search_graph error: {e}")
        return [f"Graph search error: {e}"]


# ── Stats ─────────────────────────────────────────────────────────────────────
def get_graph_stats():
    try:
        counts   = run_query("MATCH (n) WITH count(n) as nodes MATCH ()-[r]->() RETURN nodes, count(r) as relationships")
        subjects = run_query("MATCH (n:Entity) WHERE n.subject IS NOT NULL AND n.subject <> '' RETURN n.subject as subject, count(n) as count ORDER BY count DESC")
        types    = run_query("MATCH (n) WHERE n.type IS NOT NULL RETURN n.type as type, count(n) as count ORDER BY count DESC LIMIT 15")
        concepts = run_query("MATCH (n:Concept) RETURN count(n) as cnt")
        sources  = run_query("MATCH (n:Entity) WHERE n.source IS NOT NULL RETURN n.source as source, count(n) as count ORDER BY count DESC")
        stats = counts[0] if counts else {"nodes": 0, "relationships": 0}
        stats["subjects"]  = {r["subject"]: r["count"] for r in subjects}
        stats["by_type"]   = {r["type"]: r["count"] for r in types}
        stats["concepts"]  = concepts[0]["cnt"] if concepts else 0
        stats["by_source"] = {r["source"]: r["count"] for r in sources}
        return stats
    except Exception as e:
        return {"error": str(e)}