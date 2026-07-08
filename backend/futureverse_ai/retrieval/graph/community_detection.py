"""
community_detection.py — Leiden community detection over the Entity graph.

Reuses one driver via context manager (no leak), batches all writes with UNWIND
(was one round-trip per node), and never lets a failure crash the background
thread that calls it. (Phase 1, M-16)
"""
import logging

from neo4j import GraphDatabase

log = logging.getLogger(__name__)


def build_graph_communities(uri, user, password):
    try:
        import igraph as ig
        import leidenalg
    except Exception as e:
        log.warning("community detection skipped \u2014 igraph/leidenalg unavailable: %s", e)
        return

    try:
        # Context manager guarantees the driver is closed even on error.
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            edges = []
            with driver.session() as session:
                results = session.run(
                    """
                    MATCH (a:Entity)-[r]->(b:Entity)
                    WHERE a.id IS NOT NULL AND b.id IS NOT NULL
                    RETURN a.id AS source, b.id AS target
                    """
                )
                for r in results:
                    edges.append((r["source"], r["target"]))

            if not edges:
                log.info("community detection: no edges, nothing to do")
                return

            nodes = list({x for e in edges for x in e})
            mapping = {n: i for i, n in enumerate(nodes)}
            ig_edges = [(mapping[a], mapping[b]) for a, b in edges]

            g = ig.Graph(n=len(nodes), edges=ig_edges)
            partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)

            communities = {}
            for cid, cluster in enumerate(partition):
                for node_idx in cluster:
                    communities.setdefault(cid, []).append(nodes[node_idx])

            summary_rows = [
                {"cid": cid, "summary": ", ".join(members[:10])}
                for cid, members in communities.items()
            ]
            link_rows = [
                {"cid": cid, "eid": eid}
                for cid, members in communities.items()
                for eid in members
            ]

            # Batched writes: one statement for all summaries, one for all CONTAINS
            # edges \u2014 was N+1 round-trips per node.
            with driver.session() as session:
                session.run(
                    "UNWIND $rows AS row "
                    "MERGE (c:Community {id: row.cid}) SET c.summary = row.summary",
                    {"rows": summary_rows},
                )
                session.run(
                    "UNWIND $rows AS row "
                    "MATCH (e:Entity {id: row.eid}) "
                    "MATCH (c:Community {id: row.cid}) "
                    "MERGE (c)-[:CONTAINS]->(e)",
                    {"rows": link_rows},
                )
            log.info(
                "community detection: %d communities over %d nodes, %d memberships",
                len(communities), len(nodes), len(link_rows),
            )
    except Exception as e:
        log.exception("community detection failed: %s", e)
