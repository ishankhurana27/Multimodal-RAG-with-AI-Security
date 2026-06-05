import os
import logging
from fastapi import APIRouter, Depends
from futureverse_ai.models.users import UserModel
from futureverse_ai.utils.auth import get_current_user

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/data")
async def get_graph_data(user: UserModel = Depends(get_current_user)):
    try:
        from futureverse_ai.retrieval.graph.extractor import run_query

        # ── Nodes: use n.id as the graph node ID (meaningful name) ───────────
        nodes_result = run_query("""
            MATCH (n)
            WHERE n.id IS NOT NULL
              AND NOT toString(n.id) =~ '^\\\\d+$'
              AND size(toString(n.id)) > 2
            WITH toString(n.id) AS did, n
            WITH DISTINCT did, n
            RETURN
                did AS id,
                did AS label,
                coalesce(n.type, labels(n)[0], 'Entity') AS type,
                coalesce(n.doc_name, n.source, n.book, '') AS doc_name,
                coalesce(n.subject, '')    AS subject,
                coalesce(n.chapter, '')    AS chapter,
                coalesce(n.class_level, '') AS class_level
            LIMIT 400
        """)

        # ── Edges: use n.id as source/target so they match node IDs ──────────
        rels_result = run_query("""
            MATCH (a)-[r]->(b)
            WHERE a.id IS NOT NULL AND b.id IS NOT NULL
              AND NOT toString(a.id) =~ '^\\\\d+$'
              AND NOT toString(b.id) =~ '^\\\\d+$'
              AND size(toString(a.id)) > 2
              AND size(toString(b.id)) > 2
            WITH DISTINCT toString(a.id) AS src, toString(b.id) AS tgt, type(r) AS rtype
            RETURN src AS source, tgt AS target, rtype AS type
            LIMIT 1000
        """)

        # ── Stats ─────────────────────────────────────────────────────────────
        stats_result = run_query("""
            MATCH (n)
            WHERE n.id IS NOT NULL AND NOT toString(n.id) =~ '^\\\\d+$'
            WITH count(n) AS nodes
            MATCH ()-[r]->()
            RETURN nodes, count(r) AS relationships
        """)

        nodes = [
            {
                "id":          str(n["id"]),
                "label":       str(n.get("label") or "unknown"),
                "type":        str(n.get("type") or "Entity"),
                "doc_name":    str(n.get("doc_name") or ""),
                "subject":     str(n.get("subject") or ""),
                "chapter":     str(n.get("chapter") or ""),
                "class_level": str(n.get("class_level") or ""),
            }
            for n in nodes_result
        ]

        edges = [
            {
                "source": str(r["source"]),
                "target": str(r["target"]),
                "type":   str(r.get("type") or ""),
            }
            for r in rels_result
        ]

        stats = stats_result[0] if stats_result else {"nodes": 0, "relationships": 0}
        return {"nodes": nodes, "edges": edges, "stats": stats}

    except Exception as e:
        log.error(f"Graph data error: {e}")
        return {"nodes": [], "edges": [], "stats": {"nodes": 0, "relationships": 0}, "error": str(e)}


@router.get("/search")
async def search_graph_entities(query: str, user: UserModel = Depends(get_current_user)):
    try:
        from futureverse_ai.retrieval.graph.extractor import run_query

        # Extract meaningful keywords (skip short/common words)
        stopwords = {"the","a","an","of","in","at","by","to","and","or","for",
                     "what","how","why","is","are","was","were","which","that"}
        keywords = [
            w.strip().lower()
            for w in query.split()
            if len(w.strip()) > 2 and w.strip().lower() not in stopwords
        ][:5]

        if not keywords:
            keywords = [query.strip()]

        nodes   = {}
        edges   = []
        seen_edges = set()

        for keyword in keywords:
            results = run_query("""
                MATCH (n)
                WHERE n.id IS NOT NULL
                  AND NOT toString(n.id) =~ '^\\\\d+$'
                  AND toLower(toString(n.id)) CONTAINS toLower($keyword)
                WITH toString(n.id) AS nid,
                     coalesce(n.type, labels(n)[0], 'Entity') AS ntype,
                     coalesce(n.subject, '') AS nsubject,
                     coalesce(n.chapter, '') AS nchapter
                WITH DISTINCT nid, ntype, nsubject, nchapter
                OPTIONAL MATCH (m)-[r]-(rel)
                WHERE toString(m.id) = nid
                  AND rel.id IS NOT NULL
                  AND NOT toString(rel.id) =~ '^\\\\d+$'
                  AND size(toString(rel.id)) > 2
                RETURN
                    nid           AS id,
                    nid           AS label,
                    ntype         AS type,
                    nsubject      AS subject,
                    nchapter      AS chapter,
                    toString(rel.id)    AS related_id,
                    toString(rel.id)    AS related_label,
                    coalesce(rel.type, labels(rel)[0], 'Entity') AS related_type,
                    type(r)       AS relation
                LIMIT 60
            """, {"keyword": keyword})

            for row in results:
                nid = str(row["id"])
                if nid not in nodes:
                    nodes[nid] = {
                        "id":          nid,
                        "label":       str(row.get("label") or nid),
                        "type":        str(row.get("type") or "Entity"),
                        "subject":     str(row.get("subject") or ""),
                        "chapter":     str(row.get("chapter") or ""),
                        "highlighted": True,
                    }

                if row.get("related_id") and row.get("related_label"):
                    rid = str(row["related_id"])
                    if rid not in nodes:
                        nodes[rid] = {
                            "id":          rid,
                            "label":       str(row.get("related_label") or rid),
                            "type":        str(row.get("related_type") or "Entity"),
                            "subject":     "",
                            "chapter":     "",
                            "highlighted": False,
                        }
                    edge_key = f"{nid}|{rid}|{row.get('relation','')}"
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append({
                            "source": nid,
                            "target": rid,
                            "type":   str(row.get("relation") or ""),
                        })

        return {"nodes": list(nodes.values()), "edges": edges}

    except Exception as e:
        log.error(f"Graph search error: {e}")
        return {"nodes": [], "edges": [], "error": str(e)}


@router.get("/entities")
async def extract_entities_from_text(text: str, user: UserModel = Depends(get_current_user)):
    try:
        from futureverse_ai.retrieval.graph.extractor import run_query
        import re

        # Extract candidate words
        words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        geo   = re.findall(r'\b(?:River|Mountain|Plain|Plateau|Delta|Lake|Sea|Ocean|Bay|Gulf)\b',
                           text, re.IGNORECASE)
        candidates = list(dict.fromkeys(words + geo))[:10]

        entities  = []
        seen      = set()

        for word in candidates:
            results = run_query("""
                MATCH (n)
                WHERE n.id IS NOT NULL
                  AND toLower(toString(n.id)) CONTAINS toLower($word)
                  AND NOT toString(n.id) =~ '^\\\\d+$'
                RETURN n.id AS name, n.type AS type, n.subject AS subject
                LIMIT 3
            """, {"word": word.lower()})

            for row in results:
                name = row.get("name", "")
                if name and name not in seen:
                    seen.add(name)
                    entities.append({
                        "name":          name,
                        "type":          row.get("type", "Entity"),
                        "subject":       row.get("subject", ""),
                        "graph_url":     f"/graph?search={name}",
                    })

        return {"entities": entities[:8]}

    except Exception as e:
        return {"entities": [], "error": str(e)}