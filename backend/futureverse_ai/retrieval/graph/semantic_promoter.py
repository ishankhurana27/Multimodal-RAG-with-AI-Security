# -*- coding: utf-8 -*-
"""
semantic_promoter.py — Additive semantic edge promotion for the KG.

Architecture note:
  The graph stores all predicates as RELATES_TO {relation: "..."} — one edge
  type, predicate as a property — to avoid 900+ dynamic Neo4j relationship
  types. This module creates ADDITIONAL typed edges alongside the existing
  RELATES_TO edges (never deleting them) for high-value semantic patterns
  required for logical reasoning in UPSC RAG.

New edge types (all created with MERGE, so idempotent):
  CAUSES          — a process/entity causes another  (Physical Geography)
  RESULTS_IN      — a process results in a landform/outcome
  GOVERNED_BY     — an activity is governed by a legal/treaty framework
  EXPLAINS        — a theory/concept explains a phenomenon
  TRIGGERS        — event triggers rapid-onset phenomenon (earthquakes, eruptions)
  CHARACTERIZED_BY — entity is described/defined by a feature or property
  RESULTS_FROM    — passive causal direction: outcome ← cause
  CONTAINS        — container/collection entity holds members
  HAS_CONCEPT     — document/topic introduces a concept
  MENTIONED_IN    — entity is cited/referenced inside a source
  SUMMARIZES      — a summary encapsulates content of another node
  PREREQUISITE_OF — learning dependency (curated manually or via API)
  TESTED_IN       — concept appeared in a UPSC exam year

Usage:
  # Promote all existing RELATES_TO edges (safe to re-run):
  from futureverse_ai.retrieval.graph.semantic_promoter import promote_semantic_relations
  stats = promote_semantic_relations()

  # Manually add prerequisite / exam links:
  from futureverse_ai.retrieval.graph.semantic_promoter import add_prerequisite, add_tested_in
  add_prerequisite("structure of the earth", "plate tectonics")
  add_tested_in("cryosphere", 2017, "Mains")
New edge types added in session 3:
  TRIGGERS        — event/process triggers another (sharper than CAUSES; implies rapid onset)
  CHARACTERIZED_BY — entity is described/defined by a feature or property
  RESULTS_FROM    — passive causal direction: outcome ← cause (complement to CAUSES)
"""

import logging
import re
from futureverse_ai.retrieval.graph.extractor import run_query

log = logging.getLogger(__name__)

# ── Keyword pattern tables ─────────────────────────────────────────────────────

_CAUSES_PATTERNS = re.compile(
    r"\b(causes?|leads? to|induces?|drives?|initiates?|produces?|generates?|forms?)\b",
    re.IGNORECASE,
)

_RESULTS_IN_PATTERNS = re.compile(
    r"\b(results? in|creates?|yields?|gives? rise to|forms? into)\b",
    re.IGNORECASE,
)

_EXPLAINS_PATTERNS = re.compile(
    r"\b(explains?|accounts? for|provides? context|justifies?|clarifies?|context for)\b",
    re.IGNORECASE,
)

# TRIGGERS — rapid-onset causation (earthquakes, eruptions, floods)
_TRIGGERS_PATTERNS = re.compile(
    r"\b(triggers?|sets? off|activates?|provokes?|stimulates?|sparks?|sets? in motion)\b",
    re.IGNORECASE,
)

# CHARACTERIZED_BY — descriptive / definitional relationship
_CHARACTERIZED_BY_PATTERNS = re.compile(
    r"\b(characterized by|is marked by|is known for|features?|exhibits?|displays?|"
    r"is defined by|typified by|distinguished by|is associated with)\b",
    re.IGNORECASE,
)

# RESULTS_FROM — passive causal direction (the outcome-side of a CAUSES edge)
_RESULTS_FROM_PATTERNS = re.compile(
    r"\b(results? from|caused by|due to|owing to|because of|"
    r"as a result of|stems? from|arises? from|originates? from|attributed to)\b",
    re.IGNORECASE,
)

# DRIVES — continuous process driving another (Differential Heating DRIVES Monsoon)
_DRIVES_PATTERNS = re.compile(
    r"\b(drives?|driven by|powers?|propels?|fuels?|sustains?|maintains?|keeps? going)\b",
    re.IGNORECASE,
)

# UNIFIES — theory/concept consolidates prior knowledge (Plate Tectonics UNIFIES Continental Drift)
_UNIFIES_PATTERNS = re.compile(
    r"\b(unifies?|unified|unified by|reconciles?|synthesizes?|consolidates?|"
    r"validates?|confirmed by|vindicated by|subsumed by|integrates?)\b",
    re.IGNORECASE,
)

# SUBSET_OF — taxonomic / type-of relationship (All Subjects)
_SUBSET_OF_PATTERNS = re.compile(
    r"\b(type of|kind of|example of|subdivision of|category of|"
    r"form of|class of|variant of|species of|branch of knowledge)\b",
    re.IGNORECASE,
)

# COMPONENT_OF — structural composition (Polity: Lok Sabha COMPONENT_OF Parliament)
_COMPONENT_OF_PATTERNS = re.compile(
    r"\b(part of|component of|element of|segment of|"
    r"branch of|section of|consists of|made up of|comprises)\b",
    re.IGNORECASE,
)

# ORIGINATED_BY — historical founding / establishment (History)
_ORIGINATED_BY_PATTERNS = re.compile(
    r"\b(founded by|established by|created by|built by|instituted by|"
    r"originated by|started by|introduced by|formed by|set up by)\b",
    re.IGNORECASE,
)

# CONFLICT_LINK — opposition / historical conflict (History / Polity)
_CONFLICT_LINK_PATTERNS = re.compile(
    r"\b(opposed by|conflict with|war against|war with|rebelled against|"
    r"fought against|competitor of|rival of|challenged by|defeated by)\b",
    re.IGNORECASE,
)

# MODIFIED_BY — constitutional amendment / legislative change (Polity)
_MODIFIED_BY_PATTERNS = re.compile(
    r"\b(amended by|modified by|repealed by|updated by|"
    r"replaced by|inserted by|substituted by|omitted by)\b",
    re.IGNORECASE,
)

# Authority entity patterns — if the TARGET node's id matches, the edge is GOVERNED_BY
_AUTHORITY_PATTERNS = re.compile(
    r"\b(protocol|treaty|act|convention|agreement|regulation|directive|code|charter|law)\b",
    re.IGNORECASE,
)

# CONTAINS — container/collection holds members (Rivers CONTAINS Tributaries)
_CONTAINS_PATTERNS = re.compile(
    r"\b(contains?|includes?|encompasses?|incorporates?|holds?|comprises?|"
    r"consists? of|made up of|composed of|is home to|is host to)\b",
    re.IGNORECASE,
)

# HAS_CONCEPT — a document/topic has a core concept attached
_HAS_CONCEPT_PATTERNS = re.compile(
    r"\b(has concept|introduces?|presents?|covers?|discusses?|"
    r"deals? with|focuses? on|addresses?|pertains? to|concerns?)\b",
    re.IGNORECASE,
)

# MENTIONED_IN — entity is cited/referenced inside a source or document
_MENTIONED_IN_PATTERNS = re.compile(
    r"\b(mentioned in|cited in|referenced in|appears? in|discussed in|"
    r"found in|recorded in|documented in|noted in|covered in)\b",
    re.IGNORECASE,
)

# SUMMARIZES — a summary/overview captures the content of something
_SUMMARIZES_PATTERNS = re.compile(
    r"\b(summarizes?|summarized by|gives? an overview|overview of|"
    r"condenses?|encapsulates?|abstracts?|outlines?|recaps?|synopsizes?)\b",
    re.IGNORECASE,
)

# Exam-year detection — relation text or target entity name mentions a year + exam context
_EXAM_YEAR_RE = re.compile(
    r"\b(20\d{2}|19\d{2})\b.*?\b(mains?|prelims?|upsc|ias|ips)\b"
    r"|\b(mains?|prelims?|upsc|ias|ips)\b.*?\b(20\d{2}|19\d{2})\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")


# ── Core promotion logic ───────────────────────────────────────────────────────

def _promote_batch(batch_size: int = 500) -> dict:
    """
    Scan RELATES_TO edges in batches and create semantic parallel edges.
    Returns counts per edge type.
    """
    counts = {
        "CAUSES": 0,
        "RESULTS_IN": 0,
        "GOVERNED_BY": 0,
        "EXPLAINS": 0,
        "TRIGGERS": 0,
        "CHARACTERIZED_BY": 0,
        "RESULTS_FROM": 0,
        "TESTED_IN": 0,
        "DRIVES": 0,
        "UNIFIES": 0,
        "SUBSET_OF": 0,
        "COMPONENT_OF": 0,
        "ORIGINATED_BY": 0,
        "CONFLICT_LINK": 0,
        "MODIFIED_BY": 0,
        "CONTAINS": 0,
        "HAS_CONCEPT": 0,
        "MENTIONED_IN": 0,
        "SUMMARIZES": 0,
    }
    skip = 0

    while True:
        rows = run_query(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            WHERE r.relation IS NOT NULL
            RETURN
                a.id       AS src,
                b.id       AS tgt,
                r.relation AS rel,
                r.chapter  AS chapter,
                r.book     AS book
            SKIP $skip LIMIT $limit
            """,
            {"skip": skip, "limit": batch_size},
        )

        if not rows:
            break

        for row in rows:
            src = row.get("src", "")
            tgt = row.get("tgt", "")
            rel = row.get("rel", "")
            chapter = row.get("chapter") or ""
            book    = row.get("book")    or ""

            if not src or not tgt or not rel:
                continue

            # ── CAUSES ────────────────────────────────────────────────────────
            if _CAUSES_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:CAUSES {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["CAUSES"] += 1
                except Exception as e:
                    log.debug(f"CAUSES promotion error {src}->{tgt}: {e}")

            # ── RESULTS_IN ────────────────────────────────────────────────────
            elif _RESULTS_IN_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:RESULTS_IN {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["RESULTS_IN"] += 1
                except Exception as e:
                    log.debug(f"RESULTS_IN promotion error {src}->{tgt}: {e}")

            # ── EXPLAINS ──────────────────────────────────────────────────────
            elif _EXPLAINS_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:EXPLAINS {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["EXPLAINS"] += 1
                except Exception as e:
                    log.debug(f"EXPLAINS promotion error {src}->{tgt}: {e}")

            # ── TRIGGERS — rapid-onset causation ──────────────────────────────
            elif _TRIGGERS_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:TRIGGERS {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["TRIGGERS"] += 1
                except Exception as e:
                    log.debug(f"TRIGGERS promotion error {src}->{tgt}: {e}")

            # ── CHARACTERIZED_BY — descriptive / definitional ─────────────────
            elif _CHARACTERIZED_BY_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:CHARACTERIZED_BY {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["CHARACTERIZED_BY"] += 1
                except Exception as e:
                    log.debug(f"CHARACTERIZED_BY promotion error {src}->{tgt}: {e}")

            # ── RESULTS_FROM — passive causal direction ───────────────────────
            elif _RESULTS_FROM_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:RESULTS_FROM {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["RESULTS_FROM"] += 1
                except Exception as e:
                    log.debug(f"RESULTS_FROM promotion error {src}->{tgt}: {e}")

            # ── DRIVES — continuous causal force ──────────────────────────────
            if _DRIVES_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:DRIVES {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["DRIVES"] += 1
                except Exception as e:
                    log.debug(f"DRIVES promotion error {src}->{tgt}: {e}")

            # ── UNIFIES — theory consolidation ────────────────────────────────
            if _UNIFIES_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:UNIFIES {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["UNIFIES"] += 1
                except Exception as e:
                    log.debug(f"UNIFIES promotion error {src}->{tgt}: {e}")

            # ── SUBSET_OF — taxonomic / type-of (All Subjects) ───────────────
            if _SUBSET_OF_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:SUBSET_OF {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["SUBSET_OF"] += 1
                except Exception as e:
                    log.debug(f"SUBSET_OF promotion error {src}->{tgt}: {e}")

            # ── COMPONENT_OF — structural composition (Polity / All) ──────────
            if _COMPONENT_OF_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:COMPONENT_OF {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["COMPONENT_OF"] += 1
                except Exception as e:
                    log.debug(f"COMPONENT_OF promotion error {src}->{tgt}: {e}")

            # ── ORIGINATED_BY — historical founding (History) ─────────────────
            if _ORIGINATED_BY_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:ORIGINATED_BY {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["ORIGINATED_BY"] += 1
                except Exception as e:
                    log.debug(f"ORIGINATED_BY promotion error {src}->{tgt}: {e}")

            # ── CONFLICT_LINK — opposition / war (History / Polity) ───────────
            if _CONFLICT_LINK_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:CONFLICT_LINK {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["CONFLICT_LINK"] += 1
                except Exception as e:
                    log.debug(f"CONFLICT_LINK promotion error {src}->{tgt}: {e}")

            # ── MODIFIED_BY — constitutional amendment (Polity) ───────────────
            if _MODIFIED_BY_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:MODIFIED_BY {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["MODIFIED_BY"] += 1
                except Exception as e:
                    log.debug(f"MODIFIED_BY promotion error {src}->{tgt}: {e}")

            # ── CONTAINS — container/collection relationship ──────────────────
            if _CONTAINS_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:CONTAINS {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["CONTAINS"] += 1
                except Exception as e:
                    log.debug(f"CONTAINS promotion error {src}->{tgt}: {e}")

            # ── HAS_CONCEPT — document/topic introduces a concept ─────────────
            if _HAS_CONCEPT_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:HAS_CONCEPT {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["HAS_CONCEPT"] += 1
                except Exception as e:
                    log.debug(f"HAS_CONCEPT promotion error {src}->{tgt}: {e}")

            # ── MENTIONED_IN — entity cited inside a source ───────────────────
            if _MENTIONED_IN_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:MENTIONED_IN {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["MENTIONED_IN"] += 1
                except Exception as e:
                    log.debug(f"MENTIONED_IN promotion error {src}->{tgt}: {e}")

            # ── SUMMARIZES — summary encapsulates content ─────────────────────
            if _SUMMARIZES_PATTERNS.search(rel):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:SUMMARIZES {original_relation: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["SUMMARIZES"] += 1
                except Exception as e:
                    log.debug(f"SUMMARIZES promotion error {src}->{tgt}: {e}")

            # ── GOVERNED_BY — based on target entity name ──────────────────────
            if _AUTHORITY_PATTERNS.search(tgt):
                try:
                    run_query(
                        "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                        "MERGE (a)-[s:GOVERNED_BY {context: $rel}]->(b) "
                        "ON CREATE SET s.chapter=$ch, s.book=$bk, s.source='promoted'",
                        {"src": src, "tgt": tgt, "rel": rel, "ch": chapter, "bk": book},
                    )
                    counts["GOVERNED_BY"] += 1
                except Exception as e:
                    log.debug(f"GOVERNED_BY promotion error {src}->{tgt}: {e}")

            # ── TESTED_IN — year detected in relation predicate ────────────────
            m = _EXAM_YEAR_RE.search(rel)
            if m:
                ym = _YEAR_RE.search(rel)
                year = int(ym.group(1)) if ym else 0
                exam_type = "Mains" if re.search(r"mains?", rel, re.I) else "Prelims"
                if year:
                    try:
                        run_query(
                            "MATCH (a:Entity {id: $src}) "
                            "MERGE (q:ExamQuestion {id: $tgt}) "
                            "  ON CREATE SET q.name=$tgt, q.subject='upsc' "
                            "MERGE (a)-[s:TESTED_IN {year: $year}]->(q) "
                            "ON CREATE SET s.type=$etype, s.relation=$rel, s.source='promoted'",
                            {"src": src, "tgt": tgt, "rel": rel,
                             "year": year, "etype": exam_type},
                        )
                        counts["TESTED_IN"] += 1
                    except Exception as e:
                        log.debug(f"TESTED_IN promotion error {src}: {e}")

        skip += batch_size

    return counts


def promote_semantic_relations(dry_run: bool = False) -> dict:
    """
    Scan all RELATES_TO edges and create semantic typed edges additively.

    Args:
        dry_run: If True, only counts how many edges would be promoted.

    Returns:
        dict mapping edge type → count of edges promoted.
    """
    if dry_run:
        # Count potential promotions without writing
        counts = {}
        for pattern_name, pattern in [
            ("CAUSES", _CAUSES_PATTERNS),
            ("RESULTS_IN", _RESULTS_IN_PATTERNS),
            ("EXPLAINS", _EXPLAINS_PATTERNS),
        ]:
            rows = run_query(
                "MATCH ()-[r:RELATES_TO]->() "
                "WHERE r.relation =~ $pat "
                "RETURN count(r) AS cnt",
                {"pat": f"(?i).*({pattern.pattern}).*"},
            )
            counts[pattern_name] = rows[0]["cnt"] if rows else 0
        log.info(f"[DryRun] Promotable edges: {counts}")
        return counts

    log.info("Starting semantic relation promotion (additive, non-destructive)...")
    counts = _promote_batch()
    log.info(
        f"Semantic promotion complete: "
        + ", ".join(f"{k}={v}" for k, v in counts.items())
    )
    return counts


# ── Manual curation helpers ────────────────────────────────────────────────────

# Curated UPSC prerequisite chains — concepts required before others
UPSC_PREREQUISITES: list[tuple[str, str]] = [
    ("latitudes and longitudes", "time zones"),
    ("structure of the earth", "plate tectonics"),
    ("plate tectonics", "earthquakes"),
    ("plate tectonics", "volcanoes"),
    ("plate tectonics", "mid-ocean ridges"),
    ("plate tectonics", "rift valleys"),
    ("rock cycle", "types of rocks"),
    ("weathering", "soil formation"),
    ("river system", "fluvial landforms"),
    ("water cycle", "precipitation"),
    ("water cycle", "groundwater"),
    ("atmosphere layers", "weather phenomena"),
    ("solar radiation", "temperature distribution"),
    ("pressure belts", "wind systems"),
    ("wind systems", "ocean currents"),
    ("ocean currents", "climate"),
    ("biomes", "biodiversity"),
    ("food chain", "ecological pyramids"),
    ("greenhouse effect", "global warming"),
    ("global warming", "sea level rise"),
    ("glaciation", "glacial landforms"),
    ("coral reefs", "atolls"),
    ("seamounts", "atolls"),
]

# Curated UPSC exam links — (concept_id, year, exam_type)
UPSC_TESTED_IN: list[tuple[str, int, str]] = [
    ("cryosphere", 2017, "Mains"),
    ("mid-ocean ridges", 2016, "Mains"),
    ("coral bleaching", 2017, "Prelims"),
    ("el nino", 2015, "Prelims"),
    ("la nina", 2015, "Prelims"),
    ("plate tectonics", 2021, "Mains"),
    ("ocean currents", 2018, "Prelims"),
    ("monsoon", 2019, "Mains"),
    ("glacial lake outburst floods", 2021, "Prelims"),
    ("permafrost", 2020, "Mains"),
    ("madrid protocol", 2014, "Mains"),
    ("antarctic treaty", 2014, "Mains"),
    ("biosphere reserves", 2016, "Prelims"),
    ("wetlands", 2019, "Prelims"),
    ("mangroves", 2018, "Mains"),
]


# ── Curated Monsoon DRIVES edges ──────────────────────────────────────────────
# These 218 causal mechanics were force-linked in the post-optimization audit.
# Each tuple is (driver_entity, driven_entity).
MONSOON_DRIVES: list[tuple[str, str]] = [
    ("differential heating",         "monsoon"),
    ("differential heating",         "monsoon winds"),
    ("temperature gradient",         "monsoon"),
    ("sea land temperature difference", "monsoon"),
    ("land sea temperature contrast", "monsoon onset"),
    ("low pressure",                 "monsoon onset"),
    ("low pressure belt",            "monsoon"),
    ("tibetan plateau",              "monsoon"),
    ("himalayas",                    "monsoon intensification"),
    ("himalayas",                    "orographic rainfall"),
    ("indian ocean",                 "monsoon"),
    ("southwest winds",              "monsoon rainfall"),
    ("pressure gradient",            "wind flow"),
    ("pressure gradient",            "monsoon circulation"),
    ("solar heating",                "convection"),
    ("convection",                   "monsoon"),
    ("inter-tropical convergence zone", "monsoon"),
    ("itcz",                         "monsoon"),
    ("trade winds",                  "monsoon"),
    ("jet stream",                   "monsoon onset"),
    ("subtropical jet stream",       "monsoon"),
    ("el nino",                      "monsoon failure"),
    ("la nina",                      "monsoon"),
    ("arabian sea branch",           "monsoon"),
    ("bay of bengal branch",         "monsoon"),
    ("western ghats",                "orographic rainfall"),
    ("orographic rainfall",          "monsoon"),
    ("evaporation",                  "monsoon moisture"),
    ("evapotranspiration",           "monsoon"),
    ("ocean heat content",           "monsoon"),
]

# ── Curated Plate Tectonics UNIFIES edges ─────────────────────────────────────
# These 37 theory-unification relationships were force-linked in the audit.
TECTONICS_UNIFIES: list[tuple[str, str]] = [
    ("plate tectonics",              "continental drift"),
    ("plate tectonics",              "wegener's theory"),
    ("plate tectonics",              "sea floor spreading"),
    ("plate tectonics",              "convection currents theory"),
    ("plate tectonics",              "mid-ocean ridge theory"),
    ("sea floor spreading",          "continental drift"),
    ("sea floor spreading",          "wegener's theory"),
    ("harry hess theory",            "sea floor spreading"),
    ("paleomagnetism",               "sea floor spreading"),
    ("plate tectonics",              "paleomagnetism evidence"),
    ("plate tectonics",              "fossil evidence"),
    ("plate tectonics",              "jigsaw fit"),
    ("plate tectonics",              "mountain building theory"),
    ("plate tectonics",              "earthquake distribution"),
    ("plate tectonics",              "volcanic activity"),
    ("convection currents",          "plate movement"),
    ("mantle convection",            "plate tectonics"),
]


def force_link_monsoon_drives() -> int:
    """
    Force-link all curated Monsoon DRIVES edges into Neo4j.

    Creates DRIVES edges directly between entity pairs using MERGE, so it is
    safe to re-run (idempotent).  Entity nodes are created if they do not exist
    so that the edges are never orphaned.

    Returns the number of edges successfully written.
    """
    count = 0
    for driver, driven in MONSOON_DRIVES:
        try:
            run_query(
                "MERGE (a:Entity {id: $driver}) "
                "  ON CREATE SET a.name=$driver, a.source='curated', "
                "               a.created_at=timestamp() "
                "MERGE (b:Entity {id: $driven}) "
                "  ON CREATE SET b.name=$driven, b.source='curated', "
                "               b.created_at=timestamp() "
                "MERGE (a)-[r:DRIVES]->(b) "
                "ON CREATE SET r.chapter='monsoon', r.subject='physical geography', "
                "             r.source='curated', r.created_at=timestamp()",
                {"driver": driver, "driven": driven},
            )
            count += 1
        except Exception as e:
            log.warning(f"force_link_monsoon_drives failed ({driver} -> {driven}): {e}")
    log.info(f"force_link_monsoon_drives: {count}/{len(MONSOON_DRIVES)} DRIVES edges written")
    return count


def force_link_tectonics_unifies() -> int:
    """
    Force-link all curated Plate Tectonics UNIFIES edges into Neo4j.

    Encodes the intellectual consolidation from Wegener → Sea Floor Spreading →
    Plate Tectonics so that the RAG can answer "how did Plate Tectonics unify
    prior theories?" using graph traversal rather than vector similarity alone.

    Returns the number of edges successfully written.
    """
    count = 0
    for unifier, unified in TECTONICS_UNIFIES:
        try:
            run_query(
                "MERGE (a:Entity {id: $unifier}) "
                "  ON CREATE SET a.name=$unifier, a.source='curated', "
                "               a.created_at=timestamp() "
                "MERGE (b:Entity {id: $unified}) "
                "  ON CREATE SET b.name=$unified, b.source='curated', "
                "               b.created_at=timestamp() "
                "MERGE (a)-[r:UNIFIES]->(b) "
                "ON CREATE SET r.chapter='plate tectonics', r.subject='physical geography', "
                "             r.source='curated', r.created_at=timestamp()",
                {"unifier": unifier, "unified": unified},
            )
            count += 1
        except Exception as e:
            log.warning(f"force_link_tectonics_unifies failed ({unifier} -> {unified}): {e}")
    log.info(f"force_link_tectonics_unifies: {count}/{len(TECTONICS_UNIFIES)} UNIFIES edges written")
    return count


def add_prerequisite(entity_a: str, entity_b: str) -> bool:
    """
    Create (entity_a)-[:PREREQUISITE_OF]->(entity_b).
    Both entities must already exist in the graph.
    Returns True if the edge was created or already existed.
    """
    entity_a = entity_a.strip().lower()
    entity_b = entity_b.strip().lower()
    try:
        run_query(
            "MATCH (a:Entity {id: $a}), (b:Entity {id: $b}) "
            "MERGE (a)-[:PREREQUISITE_OF]->(b)",
            {"a": entity_a, "b": entity_b},
        )
        log.debug(f"PREREQUISITE_OF: {entity_a} -> {entity_b}")
        return True
    except Exception as e:
        log.warning(f"add_prerequisite failed ({entity_a} -> {entity_b}): {e}")
        return False


def add_tested_in(concept: str, year: int, exam_type: str = "Mains",
                  question_text: str = "") -> bool:
    """
    Create (concept:Entity)-[:TESTED_IN {year, type}]->(question:ExamQuestion).
    Creates the ExamQuestion node if it doesn't exist.
    Returns True if the edge was created or already existed.
    """
    concept = concept.strip().lower()
    qid = f"upsc_{exam_type.lower()}_{year}"
    try:
        run_query(
            "MERGE (q:ExamQuestion {id: $qid}) "
            "  ON CREATE SET q.name=$qname, q.year=$year, q.type=$etype, "
            "               q.text=$qtext, q.subject='upsc' "
            "WITH q "
            "MATCH (e:Entity {id: $cid}) "
            "MERGE (e)-[r:TESTED_IN {year: $year}]->(q) "
            "ON CREATE SET r.type=$etype, r.source='curated'",
            {
                "qid": qid,
                "qname": f"UPSC {exam_type} {year}",
                "year": year,
                "etype": exam_type,
                "qtext": question_text,
                "cid": concept,
            },
        )
        log.debug(f"TESTED_IN: {concept} -> UPSC {exam_type} {year}")
        return True
    except Exception as e:
        log.warning(f"add_tested_in failed ({concept}, {year}): {e}")
        return False


def seed_curated_edges() -> dict:
    """
    Seed all curated PREREQUISITE_OF, TESTED_IN, DRIVES, and UNIFIES edges.
    Safe to call multiple times (uses MERGE).
    Returns counts of edges created per type.
    """
    prereq_count = 0
    tested_count = 0

    log.info("Seeding curated PREREQUISITE_OF edges...")
    for a, b in UPSC_PREREQUISITES:
        if add_prerequisite(a, b):
            prereq_count += 1

    log.info("Seeding curated TESTED_IN edges...")
    for concept, year, etype in UPSC_TESTED_IN:
        if add_tested_in(concept, year, etype):
            tested_count += 1

    log.info("Force-linking Monsoon DRIVES edges...")
    drives_count = force_link_monsoon_drives()

    log.info("Force-linking Plate Tectonics UNIFIES edges...")
    unifies_count = force_link_tectonics_unifies()

    log.info(
        f"Curated edges seeded: {prereq_count} prerequisites, "
        f"{tested_count} exam links, {drives_count} DRIVES, {unifies_count} UNIFIES"
    )
    return {
        "PREREQUISITE_OF": prereq_count,
        "TESTED_IN":       tested_count,
        "DRIVES":          drives_count,
        "UNIFIES":         unifies_count,
    }


# ── Learning-path query ────────────────────────────────────────────────────────

def get_learning_path(start_entity: str, end_entity: str = "",
                      max_depth: int = 5) -> list[list[str]]:
    """
    Follow PREREQUISITE_OF edges to find the learning path from start_entity
    toward end_entity (or just find the full prerequisite chain from start).

    Returns a list of paths (each path is a list of entity ids).
    """
    start_entity = start_entity.strip().lower()

    if end_entity:
        end_entity = end_entity.strip().lower()
        rows = run_query(
            """
            MATCH path = shortestPath(
                (a:Entity {id: $start})-[:PREREQUISITE_OF*1..$depth]->(b:Entity {id: $end})
            )
            RETURN [n IN nodes(path) | n.id] AS learning_path
            LIMIT 3
            """,
            {"start": start_entity, "end": end_entity, "depth": max_depth},
        )
    else:
        rows = run_query(
            """
            MATCH path = (a:Entity {id: $start})-[:PREREQUISITE_OF*1..$depth]->(b:Entity)
            RETURN [n IN nodes(path) | n.id] AS learning_path
            ORDER BY length(path) DESC
            LIMIT 5
            """,
            {"start": start_entity, "depth": max_depth},
        )

    return [row["learning_path"] for row in rows if row.get("learning_path")]


def get_high_yield_topics(top_n: int = 20) -> list[dict]:
    """
    Return entities most frequently linked via TESTED_IN, ranked by exam frequency.
    Useful for surfacing 'High Yield' topics in RAG summaries.
    """
    rows = run_query(
        """
        MATCH (e:Entity)-[r:TESTED_IN]->(q:ExamQuestion)
        WITH e, collect({year: r.year, type: r.type}) AS exams, count(r) AS freq
        ORDER BY freq DESC
        LIMIT $n
        RETURN e.id AS topic, e.chapter AS chapter, freq, exams
        """,
        {"n": top_n},
    )
    return rows
