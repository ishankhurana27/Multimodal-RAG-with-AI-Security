# -*- coding: utf-8 -*-
"""
graph_rag.py — Updated version.

Entity nodes now have: id, book, chapter, source, subject,
created_at, class_level, type, text, doc_name, mention_count

Uses all available fields for richer context.
"""
import logging
import os
import threading

log = logging.getLogger(__name__)


def _escape_lucene(s: str) -> str:
    """Escape Lucene special characters before passing to db.index.fulltext.queryNodes."""
    s = s.replace("\\", "\\\\")  # backslash must be first
    for ch in ['+', '-', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '/']:
        s = s.replace(ch, '\\' + ch)
    s = s.replace("&&", "\\&&")
    s = s.replace("||", "\\||")
    return s


# ── Schema existence cache ────────────────────────────────────────────────────
# Avoids repeated db.relationshipTypes() calls and suppresses Neo4j warnings
# for relationship types (PREREQUISITE_OF, TESTED_IN …) that haven't been seeded yet.
_schema_rel_cache: dict[str, bool] = {}
_schema_rel_lock  = threading.Lock()


def _rel_type_exists(rel_type: str) -> bool:
    """Return True if the given relationship type exists in Neo4j schema (cached)."""
    with _schema_rel_lock:
        if rel_type in _schema_rel_cache:
            return _schema_rel_cache[rel_type]
    try:
        from futureverse_ai.retrieval.graph.extractor import run_query
        rows = run_query(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "WHERE relationshipType = $rt RETURN count(*) AS cnt",
            {"rt": rel_type},
        )
        exists = bool(rows and rows[0].get("cnt", 0) > 0)
    except Exception:
        exists = False
    with _schema_rel_lock:
        _schema_rel_cache[rel_type] = exists
    return exists

# ── Semantic edge priority (lower number = surfaced first in context) ──────────
# Rationale for UPSC:
#   EXPLAINS    — theory/concept explains a phenomenon: highest analytical value
#   CAUSES      — direct long-term causation
#   TRIGGERS    — rapid-onset event causation (earthquake→tsunami)
#   RESULTS_FROM — passive causal direction (complement to CAUSES)
#   RESULTS_IN  — outcome of a process
#   CHARACTERIZED_BY — descriptive/definitional
#   GOVERNED_BY — regulatory context (important but less analytical)
_SEMANTIC_PRIORITY: dict[str, int] = {
    # Tier 1 — highest analytical value: explanation & theory
    "EXPLAINS":         1,
    "UNIFIES":          1,   # theory consolidation (Plate Tectonics UNIFIES Continental Drift)
    # Tier 2 — direct causation & process
    "CAUSES":           2,
    "DRIVES":           2,   # continuous causal force
    "TRIGGERS":         2,   # rapid-onset causation
    "ORIGINATED_BY":    2,   # historical founding — high UPSC value
    # Tier 3 — outcomes & passive causation
    "RESULTS_FROM":     3,
    "RESULTS_IN":       3,
    "CONFLICT_LINK":    3,   # historical opposition
    # Tier 4 — descriptive & structural
    "CHARACTERIZED_BY": 4,
    "SUBSET_OF":        4,   # taxonomic hierarchy
    "COMPONENT_OF":     4,   # structural composition
    # Tier 5 — regulatory / administrative
    "GOVERNED_BY":      5,
    "MODIFIED_BY":      5,   # constitutional amendment
}
_AUTO_REFINE_DENSITY_THRESHOLD = 0.15   # trigger promotion below this ρ
_AUTO_REFINE_COOLDOWN: set[str] = set() # entity ids refined this process lifetime

# ── UPSC Term Aliases ─────────────────────────────────────────────────────────
# Maps complex / compound UPSC terms to the atomic node IDs stored in Neo4j.
# When the query expansion LLM generates phrases like "isostatic adjustment",
# the alias table resolves them to the actual node "isostasy" so the graph
# CONTAINS search finds a match.
_UPSC_ALIASES: dict[str, list[str]] = {
    # ── Physical Geography / Tectonics ────────────────────────────────────────
    "isostatic":       ["isostasy", "isostatic"],
    "isostasy":        ["isostasy"],
    "peninsular":      ["peninsular block", "peninsular india", "peninsular"],
    "carlsberg":       ["carlsberg ridge"],
    "northward":       ["northward drift", "drift"],
    "tibetan":         ["tibetan plateau", "tibet"],
    "himalayan":       ["himalayas"],
    "himalaya":        ["himalayas"],
    "subduction":      ["subduction", "subduction zone"],
    "seafloor":        ["sea floor spreading"],
    "divergent":       ["divergent boundary"],
    "convergent":      ["convergent boundary"],
    "transform":       ["transform fault"],
    "rifting":         ["rift valley", "rift"],
    "tectonic":        ["plate tectonics", "tectonic"],
    "seismic":         ["earthquake", "seismic activity"],
    "volcanic":        ["volcano", "volcanic activity"],
    # ── Monsoon ───────────────────────────────────────────────────────────────
    "southwest":       ["southwest monsoon", "monsoon"],
    "northeast":       ["northeast monsoon", "monsoon"],
    "intensification": ["monsoon", "intensification"],
    "intertropical":   ["itcz", "convergence zone"],
    "hadley":          ["hadley cell", "hadley"],
    "westerlies":      ["westerlies", "jet stream"],
    "subtropical":     ["subtropical ridge"],
    "onset":           ["monsoon onset", "monsoon"],
    "withdrawal":      ["monsoon withdrawal", "monsoon"],
    "thermal":         ["thermal low", "low pressure"],
    # ── Climate / Atmosphere ──────────────────────────────────────────────────
    "cyclonic":        ["cyclone", "tropical cyclone"],
    "anticyclonic":    ["anticyclone"],
    "orographic":      ["orographic rainfall", "relief rainfall"],
    "convectional":    ["convectional rainfall"],
    "maritime":        ["maritime climate"],
    "continental":     ["continental climate"],
    "adiabatic":       ["adiabatic lapse rate", "lapse rate"],
    # ── History ───────────────────────────────────────────────────────────────
    "imperial":        ["imperialism", "colonialism"],
    "nationalism":     ["nationalism", "freedom movement"],
    "revolutionary":   ["revolution", "revolt"],
    "archaeological":  ["archaeology", "excavation"],
    # ── Polity / Constitution ─────────────────────────────────────────────────
    "constitutional":  ["constitution"],
    "amendment":       ["constitutional amendment"],
    "directive":       ["directive principles", "dpsp"],
    "fundamental":     ["fundamental rights"],
    "parliamentary":   ["parliament"],
    "legislative":     ["legislature", "parliament"],
    "judicial":        ["judiciary", "supreme court"],
    "executive":       ["executive", "president"],
    "federal":         ["federalism", "federal structure"],
    # ── Economics ─────────────────────────────────────────────────────────────
    "fiscal":          ["fiscal policy"],
    "monetary":        ["monetary policy", "rbi"],
    "inflationary":    ["inflation"],
    "deflationary":    ["deflation"],
    "gdp":             ["gross domestic product", "gdp"],
}


def _decompose_upsc_queries(queries: list[str]) -> list[str]:
    """
    Decompose LLM-generated UPSC query strings into atomic keywords that
    match the short node IDs stored in Neo4j.

    Steps:
      1. Split each query into tokens, strip punctuation.
      2. Apply _UPSC_ALIASES to map complex stems → atomic node names.
      3. Preserve useful bigrams (e.g. "tibetan plateau").
      4. Return at most 15 unique terms (cap to avoid excessive Neo4j round-trips).

    Example:
      "Indian Monsoon isostatic adjustment Peninsular Block"
      → ["indian", "monsoon", "isostasy", "isostatic", "adjustment",
         "peninsular block", "peninsular india", "peninsular",
         "indian monsoon", "isostasy adjustment", ...]
    """
    _STOP = {
        "what","where","when","who","how","why","is","are","was","were",
        "the","a","an","of","in","at","by","to","from","and","or","for",
        "with","that","this","which","about","tell","explain","describe",
        "give","list","define","does","did","do","has","have","had",
        "type","types","kind","kinds","following","consider","correct",
        "option","options","answer","true","false","statements","given",
        "impact","effect","result","role","write","note","briefly",
        "critically","analyze","examine","discuss","explain","following",
        "statement","statements","context","concept","factor","factors",
    }

    terms: list[str] = []
    seen:  set[str]  = set()

    def _add(t: str) -> None:
        t = t.strip()
        if t and len(t) > 1 and t not in seen:
            seen.add(t)
            terms.append(t)

    for q in queries[:5]:
        raw_words = [w.strip("?,.:;'\"()[]!-") for w in q.split()]
        words = [w.lower() for w in raw_words if w]

        for w in words:
            if w in _STOP or len(w) < 2:
                continue
            # Apply alias expansion first
            aliases = _UPSC_ALIASES.get(w)
            if aliases:
                for alias in aliases:
                    _add(alias)
            else:
                _add(w)

        # Useful bigrams from non-stop words
        clean = [w for w in words if w not in _STOP and len(w) > 2]
        for i in range(len(clean) - 1):
            _add(f"{clean[i]} {clean[i+1]}")

    return terms[:15]


def get_graph_source(
    queries: list[str],
    limit: int = 10,
    metadata_context: dict = None,
    **kwargs
) -> dict | None:

    neo4j_uri = os.environ.get("NEO4J_URI", "")
    if not neo4j_uri:
        return None

    # === RAG KG CACHE — lookup ===
    # Default OFF (ENABLE_RAG_KG_CACHE). Fail-open: any cache error falls
    # through to the normal Neo4j pipeline. None results are never cached.
    _kg_parts = None
    try:
        from futureverse_ai.retrieval import rag_cache as _rag_cache
        if _rag_cache.KG_CACHE_ENABLED:
            _kg_mc = metadata_context or {}
            _kg_parts = [
                list(queries),
                limit,
                _kg_mc.get("subject", ""),
                _kg_mc.get("book", ""),
                _kg_mc.get("chapter", ""),
            ]
            _kg_hit = _rag_cache.cache_get("kg", _kg_parts)
            if _kg_hit is not None:
                log.debug("[RAG KG cache] HIT — %d queries", len(queries))
                return _kg_hit
            log.debug("[RAG KG cache] MISS")
    except Exception as _kg_e:
        log.debug("[RAG KG cache] lookup failed (fail-open): %s", _kg_e)
        _kg_parts = None
    # === END RAG KG CACHE lookup ===

    try:
        from futureverse_ai.retrieval.graph.extractor import run_query

        chapter = (metadata_context or {}).get("chapter", "")
        book    = (metadata_context or {}).get("book",    "")
        subject = (metadata_context or {}).get("subject", "")

        # ── Extract keywords ───────────────────────────────────────────────────
        STOP = {
            "what","where","when","who","how","why","is","are","was","were",
            "the","a","an","of","in","at","by","to","from","and","or","for",
            "with","that","this","which","about","tell","explain","describe",
            "give","list","define","does","did","do","has","have","had",
            "type","types","kind","kinds","following","consider","correct",
            "option","options","answer","true","false","statements","given",
        }

        keywords = []
        bigrams  = []
        for q in queries[:4]:
            words = [w.strip("?,.:;'\"()[]").lower() for w in q.split()]
            words = [w for w in words if w]
            for i, w in enumerate(words):
                if len(w) > 3 and w not in STOP and w not in keywords:
                    keywords.append(w)
                # Bigram: "equatorial climate", "river basin", "west coast" etc.
                if i < len(words) - 1:
                    w2 = words[i + 1].strip("?,.:;'\"()[]")
                    if (len(w) > 3 and len(w2) > 3
                            and w not in STOP and w2 not in STOP):
                        bg = f"{w} {w2}"
                        if bg not in bigrams:
                            bigrams.append(bg)

        # Bigrams first (more specific), then individual keywords, cap at 10
        bigram_tokens = {tok for bg in bigrams for tok in bg.split()}
        search_terms = bigrams[:4] + [
            k for k in keywords if k not in bigram_tokens and k not in bigrams
        ]
        search_terms = search_terms[:10]

        # ── UPSC alias expansion ───────────────────────────────────────────────
        # Augment with alias-decomposed atomic terms so that complex LLM-generated
        # queries like "isostatic adjustment Peninsular Block" still find the
        # "isostasy" and "peninsular block" nodes in Neo4j.
        _alias_terms = _decompose_upsc_queries(queries)
        for _at in _alias_terms:
            if _at not in search_terms:
                search_terms.append(_at)
                if len(search_terms) >= 18:
                    break

        if not search_terms:
            return None

        keywords = search_terms  # used in queries below

        concept_results    = []
        relation_results   = []
        concept_tag_results = []
        topic_results      = []
        entity_results     = []
        community_results  = []
        neighbourhood      = []
        semantic_results   = []   # list of (line, priority_int) — sorted before output
        prereq_results     = []   # PREREQUISITE_OF learning paths
        exam_results       = []   # TESTED_IN high-yield links
        seen = set()

        for kw in keywords:

            # ── 1. Entity nodes — now with full metadata ───────────────────────
            rows = run_query(
                """
                CALL db.index.fulltext.queryNodes("entity_text", $kw_ft)
                YIELD node AS n, score
                WHERE n.id IS NOT NULL
                  AND NOT n.generic_hub = true
                OPTIONAL MATCH (n)-[r]-(m:Entity)
                WHERE m.id IS NOT NULL
                  AND NOT m.id =~ '^\\\\d+$'
                  AND NOT m.generic_hub = true
                RETURN
                    n.id          AS entity_id,
                    n.text        AS entity_text,
                    n.subject     AS subject,
                    n.chapter     AS chapter,
                    n.book        AS book,
                    n.class_level AS class_level,
                    n.type        AS entity_type,
                    n.mention_count AS mentions,
                    score,
                    collect(DISTINCT m.id)[..5] AS related_ids,
                    collect(DISTINCT coalesce(r.relation, toLower(replace(type(r),'_',' '))))[..3] AS rel_types
                ORDER BY
                    score DESC,
                    CASE WHEN n.subject = $subject THEN 0 ELSE 1 END,
                    CASE WHEN n.chapter = $chapter THEN 0 ELSE 1 END,
                    coalesce(n.mention_count, 0) DESC
                LIMIT $limit
                """,
                {"kw_ft": _escape_lucene(kw), "limit": limit,
                 "subject": subject, "chapter": chapter}
            )
            for row in rows:
                eid   = row.get("entity_id", "")
                etext = row.get("entity_text", "")
                esub  = row.get("subject", "")
                echap = row.get("chapter", "")
                ebook = row.get("book", "")
                ecls  = row.get("class_level", "")
                etype = row.get("entity_type", "")
                ment  = row.get("mentions") or 0
                rel_ids  = [r for r in (row.get("related_ids") or []) if r]
                rel_types = row.get("rel_types") or []

                if not eid:
                    continue

                # Build readable line
                line = f"[Entity] {eid}"
                if etext and etext.lower() != eid.lower():
                    line += f": {etext[:150]}"
                if etype:
                    line += f" | Type: {etype}"
                if echap:
                    line += f" | Chapter: {echap}"
                if esub:
                    line += f" | Subject: {esub}"
                if ecls:
                    line += f" | Class: {ecls}"
                if ebook:
                    line += f" | Book: {ebook}"
                if ment > 1:
                    line += f" | Mentions: {ment}"
                if rel_ids and rel_types:
                    rels = ", ".join(
                        f"{rid} ({rt})"
                        for rid, rt in zip(rel_ids[:3], rel_types[:3])
                    )
                    line += f"\n  Related: {rels}"
                elif rel_ids:
                    line += f"\n  Related: {', '.join(rel_ids[:3])}"

                if line not in seen:
                    seen.add(line)
                    entity_results.append((line, ment, eid))

            # ── 1b. Relation-predicate search — find edges whose predicate matches ──
            # New data stores relation as r.relation property on RELATES_TO edges.
            rows = run_query(
                """
                MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                WHERE toLower(r.relation) CONTAINS toLower($kw)
                  AND NOT a.generic_hub = true
                  AND NOT b.generic_hub = true
                RETURN
                    a.id                   AS from_entity,
                    r.relation             AS relation,
                    b.id                   AS to_entity,
                    r.chapter              AS chapter,
                    r.concepts             AS concepts,
                    coalesce(r.weight, 1)  AS weight
                ORDER BY
                    CASE WHEN r.chapter = $chapter THEN 0 ELSE 1 END,
                    coalesce(r.weight, 1) DESC
                LIMIT $limit
                """,
                {"kw": kw, "limit": limit, "chapter": chapter}
            )
            for row in rows:
                fe     = row.get("from_entity", "")
                te     = row.get("to_entity",   "")
                rel    = row.get("relation",    "")
                weight = row.get("weight", 1) or 1
                if not fe or not te:
                    continue
                line = f"[Relation] {fe} --[{rel}]--> {te}"
                chap = row.get("chapter", "")
                if chap:
                    line += f" | Chapter: {chap}"
                if weight > 1:
                    line += f" | Evidence: {weight}x"
                edge_concepts = row.get("concepts") or []
                if isinstance(edge_concepts, list) and edge_concepts:
                    line += f" | Concepts: {', '.join(edge_concepts[:3])}"
                if line not in seen:
                    seen.add(line)
                    relation_results.append(line)

            # ── 1c. Concept-tag search — entities whose abstract semantic type matches ──
            # Searches the `concepts` list property set during AutoSchemaKG concept induction.
            rows = run_query(
                """
                MATCH (n:Entity)
                WHERE n.concepts IS NOT NULL
                  AND any(c IN n.concepts WHERE toLower(c) CONTAINS toLower($kw))
                RETURN
                    n.id       AS entity_id,
                    n.type     AS entity_type,
                    n.chapter  AS chapter,
                    n.subject  AS subject,
                    n.concepts AS concepts
                ORDER BY
                    CASE WHEN n.subject = $subject THEN 0 ELSE 1 END,
                    coalesce(n.mention_count, 0) DESC
                LIMIT $limit
                """,
                {"kw": kw, "limit": limit, "subject": subject}
            )
            for row in rows:
                eid = row.get("entity_id", "")
                if not eid:
                    continue
                line = f"[ConceptTag] {eid}"
                if row.get("entity_type"):
                    line += f" | Type: {row['entity_type']}"
                if row.get("chapter"):
                    line += f" | Chapter: {row['chapter']}"
                tags = row.get("concepts") or []
                if isinstance(tags, list) and tags:
                    line += f" | Tags: {', '.join(tags[:4])}"
                if line not in seen:
                    seen.add(line)
                    concept_tag_results.append(line)

            # ── 1d. Semantic typed edges — priority-ordered by reasoning value ──
            # DB-level ORDER BY: chapter match first, then edge-type priority.
            # Python-level sort applied across all keywords before context assembly.
            try:
                rows = run_query(
                    """
                    MATCH (a:Entity)-[r:CAUSES|RESULTS_IN|GOVERNED_BY|EXPLAINS|TRIGGERS|CHARACTERIZED_BY|RESULTS_FROM|DRIVES|UNIFIES|SUBSET_OF|COMPONENT_OF|ORIGINATED_BY|CONFLICT_LINK|MODIFIED_BY]->(b:Entity)
                    WHERE (toLower(a.id) CONTAINS toLower($kw)
                       OR toLower(b.id) CONTAINS toLower($kw))
                      AND NOT a.generic_hub = true
                      AND NOT b.generic_hub = true
                    RETURN
                        a.id                  AS from_entity,
                        type(r)               AS rel_type,
                        coalesce(r.original_relation, r.context, '') AS original_rel,
                        b.id                  AS to_entity,
                        coalesce(r.chapter, a.chapter) AS chapter
                    ORDER BY
                        CASE WHEN coalesce(r.chapter, a.chapter) = $chapter THEN 0 ELSE 1 END,
                        CASE type(r)
                            WHEN 'EXPLAINS'         THEN 1
                            WHEN 'UNIFIES'          THEN 1
                            WHEN 'CAUSES'           THEN 2
                            WHEN 'DRIVES'           THEN 2
                            WHEN 'TRIGGERS'         THEN 2
                            WHEN 'ORIGINATED_BY'    THEN 2
                            WHEN 'RESULTS_FROM'     THEN 3
                            WHEN 'RESULTS_IN'       THEN 3
                            WHEN 'CONFLICT_LINK'    THEN 3
                            WHEN 'CHARACTERIZED_BY' THEN 4
                            WHEN 'SUBSET_OF'        THEN 4
                            WHEN 'COMPONENT_OF'     THEN 4
                            WHEN 'GOVERNED_BY'      THEN 5
                            WHEN 'MODIFIED_BY'      THEN 5
                            ELSE 6
                        END
                    LIMIT $limit
                    """,
                    {"kw": kw, "limit": limit, "chapter": chapter},
                )
                for row in rows:
                    fe    = row.get("from_entity", "")
                    te    = row.get("to_entity",   "")
                    rtype = row.get("rel_type",    "")
                    orig  = row.get("original_rel", "")
                    chap  = row.get("chapter", "")
                    if not fe or not te:
                        continue
                    label = {
                        "CAUSES":           "causes",
                        "RESULTS_IN":       "results in",
                        "GOVERNED_BY":      "governed by",
                        "EXPLAINS":         "explains",
                        "TRIGGERS":         "triggers",
                        "CHARACTERIZED_BY": "characterized by",
                        "RESULTS_FROM":     "results from",
                        "DRIVES":           "drives",
                        "UNIFIES":          "unifies",
                        "SUBSET_OF":        "is a type of",
                        "COMPONENT_OF":     "is part of",
                        "ORIGINATED_BY":    "originated by",
                        "CONFLICT_LINK":    "in conflict with",
                        "MODIFIED_BY":      "modified by",
                    }.get(rtype, rtype.lower())
                    line = f"[{rtype}] {fe} --[{label}]--> {te}"
                    if orig:
                        line += f" (predicate: \"{orig[:60]}\")"
                    if chap:
                        line += f" | Chapter: {chap}"
                    if line not in seen:
                        seen.add(line)
                        priority = _SEMANTIC_PRIORITY.get(rtype, 6)
                        semantic_results.append((line, priority))
            except Exception:
                pass  # semantic edges are best-effort; graph may not have them yet

            # ── 1e. TESTED_IN — high-yield exam links ─────────────────────────
            # Skipped until seed_curated_edges() has been run.
            if _rel_type_exists("TESTED_IN"):
                try:
                    rows = run_query(
                        """
                        MATCH (e:Entity)-[r:TESTED_IN]->(q:ExamQuestion)
                        WHERE toLower(e.id) CONTAINS toLower($kw)
                        RETURN
                            e.id    AS topic,
                            r.year  AS year,
                            r.type  AS exam_type,
                            e.chapter AS chapter
                        ORDER BY r.year DESC
                        LIMIT 5
                        """,
                        {"kw": kw},
                    )
                    for row in rows:
                        topic = row.get("topic", "")
                        year  = row.get("year",  "")
                        etype = row.get("exam_type", "")
                        chap  = row.get("chapter", "")
                        if not topic:
                            continue
                        line = f"[HighYield] {topic} — asked in UPSC {etype} {year}"
                        if chap:
                            line += f" | Chapter: {chap}"
                        if line not in seen:
                            seen.add(line)
                            exam_results.append(line)
                except Exception:
                    pass

            # ── 1f. PREREQUISITE_OF — learning path ───────────────────────────
            # Skipped until seed_curated_edges() has been run — avoids the
            # Neo4j schema warning for non-existent relationship types.
            if _rel_type_exists("PREREQUISITE_OF"):
                try:
                    rows = run_query(
                        """
                        MATCH (a:Entity)-[:PREREQUISITE_OF]->(b:Entity)
                        WHERE toLower(a.id) CONTAINS toLower($kw)
                           OR toLower(b.id) CONTAINS toLower($kw)
                        RETURN a.id AS prereq, b.id AS dependent
                        LIMIT 5
                        """,
                        {"kw": kw},
                    )
                    for row in rows:
                        pre = row.get("prereq",    "")
                        dep = row.get("dependent", "")
                        if not pre or not dep:
                            continue
                        line = f"[Prerequisite] Study '{pre}' before '{dep}'"
                        if line not in seen:
                            seen.add(line)
                            prereq_results.append(line)
                except Exception:
                    pass

            # ── 2. Concept nodes ───────────────────────────────────────────────
            rows = run_query(
                """
                CALL db.index.fulltext.queryNodes("concept_text", $kw_ft)
                YIELD node AS c, score
                OPTIONAL MATCH (e:Entity)-[:HAS_CONCEPT]->(c)
                RETURN
                    c.name    AS concept,
                    c.subject AS subject,
                    c.source  AS source,
                    e.id      AS entity,
                    e.chapter AS chapter,
                    e.book    AS book
                LIMIT $limit
                """,
                {"kw_ft": _escape_lucene(kw), "limit": limit}
            )
            for row in rows:
                if not row.get("concept"):
                    continue
                line = f"[Concept] {row['concept']}"
                if row.get("entity"):
                    line += f" ← {row['entity']}"
                if row.get("chapter"):
                    line += f" | Chapter: {row['chapter']}"
                if row.get("subject"):
                    line += f" | {row['subject']}"
                if line not in seen:
                    seen.add(line)
                    concept_results.append(line)

            # ── 3. TopicSummary nodes ──────────────────────────────────────────
            rows = run_query(
                """
                CALL db.index.fulltext.queryNodes("topic_text", $kw_ft)
                YIELD node AS t, score
                RETURN
                    t.chapter     AS chapter,
                    t.subject     AS subject,
                    t.book        AS book,
                    t.class_level AS class_level,
                    t.summary     AS summary
                ORDER BY
                    score DESC,
                    CASE WHEN t.subject = $subject THEN 0 ELSE 1 END,
                    CASE WHEN t.chapter = $chapter THEN 0 ELSE 1 END
                LIMIT $limit
                """,
                {"kw_ft": _escape_lucene(kw), "limit": limit,
                 "subject": subject, "chapter": chapter}
            )
            for row in rows:
                line = f"[Topic] {row.get('chapter','?')} | {row.get('subject','?')}"
                if row.get("book"):
                    line += f" | {row['book']}"
                if row.get("class_level"):
                    line += f" | Class {row['class_level']}"
                if row.get("summary"):
                    line += f"\n  {row['summary'][:300]}"
                if line not in seen:
                    seen.add(line)
                    topic_results.append(line)

            # ── 4. Community summaries ─────────────────────────────────────────
            rows = run_query(
                """
                MATCH (c:Community)-[:CONTAINS]->(e:Entity)
                WHERE toLower(e.id)   CONTAINS toLower($kw)
                   OR toLower(e.text) CONTAINS toLower($kw)
                RETURN DISTINCT c.summary AS summary
                LIMIT 3
                """,
                {"kw": kw}
            )
            for row in rows:
                if row.get("summary"):
                    line = f"[Community] {row['summary']}"
                    if line not in seen:
                        seen.add(line)
                        community_results.append(line)

        # ── 4a. 2-Hop membership traversal ───────────────────────────────────
        # For "which country is part of X" questions, follow CONTAINS/COMPONENT_OF/
        # SUBSET_OF edges outward from the matched entity to surface its members.
        # Runs once per top entity found (not per keyword) to avoid N² queries.
        membership_results: list[str] = []
        _top_entities_for_membership = [x[2] for x in entity_results[:3]]
        for _me in _top_entities_for_membership:
            try:
                rows = run_query(
                    """
                    MATCH (n:Entity {id: $eid})-
                          [:CONTAINS|COMPONENT_OF|SUBSET_OF|HAS_CONCEPT]->(m:Entity)
                    WHERE NOT m.generic_hub = true
                    RETURN m.id AS member, m.type AS mtype, m.chapter AS chapter
                    LIMIT 15
                    """,
                    {"eid": _me},
                )
                if rows:
                    members = [r["member"] for r in rows if r.get("member")]
                    if members:
                        line = f"[Members] {_me} contains/includes: {', '.join(members[:12])}"
                        if line not in seen:
                            seen.add(line)
                            membership_results.append(line)
            except Exception:
                pass

        # ── 4b. Priority-sort semantic results across all keywords ────────────
        # semantic_results is a list of (line, priority_int); sort ascending so
        # EXPLAINS (1) appears before GOVERNED_BY (5) regardless of query order.
        semantic_results.sort(key=lambda x: x[1])

        # ── 4c. Auto-Refine Trigger ───────────────────────────────────────────
        # When entities were found but no semantic edges exist, the cluster may
        # not yet have been promoted.  Check density for the top matched entity
        # and fire a background promotion pass if ρ < threshold.
        # _AUTO_REFINE_COOLDOWN prevents re-firing for the same entity within
        # the same process lifetime (avoid spamming promotions on busy queries).
        if entity_results and not semantic_results:
            top_eid = entity_results[0][2] if entity_results else ""
            if top_eid and top_eid not in _AUTO_REFINE_COOLDOWN:
                def _auto_refine(eid: str) -> None:
                    try:
                        from futureverse_ai.retrieval.graph.extractor import cluster_health_audit
                        from futureverse_ai.retrieval.graph.semantic_promoter import (
                            promote_semantic_relations,
                        )
                        audit = cluster_health_audit(eid, top_n=1)
                        if audit and audit[0]["reasoning_density"] < _AUTO_REFINE_DENSITY_THRESHOLD:
                            log.info(
                                f"Auto-refine: promoting semantics for low-density "
                                f"cluster '{eid}' (ρ={audit[0]['reasoning_density']:.3f})"
                            )
                            promote_semantic_relations()
                        else:
                            log.debug(
                                f"Auto-refine: cluster '{eid}' density ok or not found, "
                                f"skipping promotion"
                            )
                    except Exception as err:
                        log.debug(f"Auto-refine failed for '{eid}': {err}")

                _AUTO_REFINE_COOLDOWN.add(top_eid)
                threading.Thread(
                    target=_auto_refine,
                    args=(top_eid,),
                    daemon=True,
                ).start()

        # ── 5. Direct fact / shortest-path between top entities ───────────────
        entity_results.sort(key=lambda x: x[1], reverse=True)
        top_entity_ids = [x[2] for x in entity_results[:4]]

        path_results = []
        for i in range(len(top_entity_ids)):
            for j in range(i + 1, min(i + 3, len(top_entity_ids))):
                try:
                    rows = run_query(
                        """
                        MATCH p = shortestPath(
                            (a:Entity {id: $a})-[*..4]-(b:Entity {id: $b})
                        )
                        RETURN [n IN nodes(p) | n.id] AS path_nodes,
                               [r IN relationships(p) | coalesce(r.relation, toLower(replace(type(r),'_',' ')))] AS path_rels
                        LIMIT 1
                        """,
                        {"a": top_entity_ids[i], "b": top_entity_ids[j]}
                    )
                    for row in rows:
                        pnodes = row.get("path_nodes") or []
                        prels  = row.get("path_rels") or []
                        if len(pnodes) >= 2:
                            parts = [pnodes[0]]
                            for n, r in zip(pnodes[1:], prels):
                                parts.append(f"--[{r}]-->")
                                parts.append(n)
                            line = "[Path] " + " ".join(parts)
                            if line not in seen:
                                seen.add(line)
                                path_results.append(line)
                except Exception:
                    pass  # path query is best-effort

        # ── 6. Neighbourhood expansion ────────────────────────────────────────

        for eid in top_entity_ids[:3]:
            rows = run_query(
                """
                MATCH (n:Entity {id: $id})-[r]-(m:Entity)
                WHERE m.id IS NOT NULL
                  AND NOT m.id =~ '^\\\\d+$'
                  AND NOT m.generic_hub = true
                OPTIONAL MATCH (m)-[:CAUSES|RESULTS_IN|GOVERNED_BY|EXPLAINS|TRIGGERS|CHARACTERIZED_BY|RESULTS_FROM|DRIVES|UNIFIES|SUBSET_OF|COMPONENT_OF|ORIGINATED_BY|CONFLICT_LINK|MODIFIED_BY]->()
                WITH r, m, count(*) AS semantic_degree
                RETURN
                    coalesce(r.relation, toLower(replace(type(r),'_',' '))) AS rel,
                    m.id       AS neighbour,
                    m.text     AS neighbour_text,
                    m.chapter  AS chapter,
                    semantic_degree
                ORDER BY semantic_degree DESC, coalesce(m.mention_count, 0) DESC
                LIMIT 6
                """,
                {"id": eid}
            )
            for row in rows:
                nb   = row.get("neighbour", "")
                nbt  = row.get("neighbour_text", "")
                rel  = row.get("rel", "")
                chap = row.get("chapter", "")
                if not nb:
                    continue
                line = f"[Related] {nb}"
                if nbt and nbt.lower() != nb.lower():
                    line += f": {nbt[:100]}"
                line += f" <--[{rel}]--> {eid}"
                if chap:
                    line += f" | Chapter: {chap}"
                if line not in seen:
                    seen.add(line)
                    neighbourhood.append(line)

        # ── 7. Nothing found ──────────────────────────────────────────────────
        if not any([entity_results, concept_results, topic_results,
                    relation_results, concept_tag_results]):
            log.warning(f"Knowledge graph: no results for keywords {keywords}")
            return None

        # ── 8. Build context ──────────────────────────────────────────────────
        sections = []

        if membership_results:
            sections.append("--- Membership / Spatial Containment ---")
            sections.extend(membership_results[:6])

        if exam_results:
            sections.append("\n--- High-Yield UPSC Topics ---")
            sections.extend(exam_results[:5])

        if semantic_results:
            sections.append("\n--- Causal & Governance Reasoning ---")
            # semantic_results = [(line, priority)] — extract lines after sorting
            sections.extend(line for line, _ in semantic_results[:8])

        if prereq_results:
            sections.append("\n--- Learning Path ---")
            sections.extend(prereq_results[:5])

        if entity_results:
            sections.append("\n--- Entities ---" if sections else "--- Entities ---")
            sections.extend(line for line, _, _ in entity_results[:12])

        if relation_results:
            sections.append("\n--- Matching Relations ---")
            sections.extend(relation_results[:10])

        if path_results:
            sections.append("\n--- Relationship Paths ---")
            sections.extend(path_results[:6])

        if neighbourhood:
            sections.append("\n--- Related Entities ---")
            sections.extend(neighbourhood[:8])

        if concept_tag_results:
            sections.append("\n--- Concept Tags ---")
            sections.extend(concept_tag_results[:8])

        if concept_results:
            sections.append("\n--- Concepts ---")
            sections.extend(concept_results[:10])

        if topic_results:
            sections.append("\n--- Topic Summaries ---")
            sections.extend(topic_results[:6])

        if community_results:
            sections.append("\n--- Communities ---")
            sections.extend(community_results[:4])

        header = "=== Knowledge Graph Context ===\n"
        if subject or book or chapter:
            ctx = " > ".join(x for x in [subject, book, chapter] if x)
            header += f"Scope: {ctx}\n"
        header += "\n"

        graph_text = header + "\n".join(sections)

        log.info(
            f"Graph context: {len(entity_results)} entities, "
            f"{len(relation_results)} relations, "
            f"{len(concept_tag_results)} concept_tags, "
            f"{len(concept_results)} concepts, "
            f"{len(topic_results)} topics, "
            f"{len(neighbourhood)} related, "
            f"{len(path_results)} paths, "
            f"{len(semantic_results)} semantic, "
            f"{len(exam_results)} high_yield, "
            f"{len(prereq_results)} prereqs"
        )

        # Flat list of entity IDs found — used by the graph-guided re-ranker in
        # retrieval/utils.py to boost vector chunks that mention these entities.
        found_entity_names = [x[2] for x in entity_results if x[2]]

        _graph_source = {
            "documents":    [[graph_text]],
            "entity_names": found_entity_names,   # ← re-ranker anchor list
            "metadatas": [[{
                "file_id":          "knowledge_graph",
                "name":             "Knowledge Graph (Neo4j)",
                "source":           "neo4j",
                "type":             "graph",
                "subject":          subject,
                "book":             book,
                "chapter":          chapter,
                "entity_count":     len(entity_results),
                "relation_count":   len(relation_results),
                "concept_count":    len(concept_results),
                "topic_count":      len(topic_results),
                "path_count":       len(path_results),
                "semantic_count":   len(semantic_results),
                "exam_count":       len(exam_results),
                "prereq_count":     len(prereq_results),
            }]],
        }

        # === RAG KG CACHE — store (only non-None results reach this point) ===
        if _kg_parts is not None:
            try:
                from futureverse_ai.retrieval import rag_cache as _rag_cache
                _rag_cache.cache_set(
                    "kg", _kg_parts, _graph_source, _rag_cache.KG_CACHE_TTL
                )
                log.debug("[RAG KG cache] stored graph context")
            except Exception as _kg_e:
                log.debug("[RAG KG cache] store failed (fail-open): %s", _kg_e)
        # === END RAG KG CACHE store ===

        return _graph_source

    except Exception as e:
        log.warning(f"Knowledge graph search failed: {e}")
        return None