from neo4j import GraphDatabase
import igraph as ig
import leidenalg


def build_graph_communities(uri,user,password):

    driver = GraphDatabase.driver(uri,auth=(user,password))

    edges = []

    with driver.session() as session:

        results=session.run("""
        MATCH (a:Entity)-[r]->(b:Entity)
        RETURN a.id as source,b.id as target
        """)

        for r in results:
            edges.append((r["source"],r["target"]))

    if not edges:
        return

    nodes=list(set([x for e in edges for x in e]))

    mapping={n:i for i,n in enumerate(nodes)}

    ig_edges=[(mapping[a],mapping[b]) for a,b in edges]

    g=ig.Graph(ig_edges)

    partition=leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition
    )

    communities={}

    for cid,cluster in enumerate(partition):

        for node in cluster:

            name=nodes[node]

            communities.setdefault(cid,[]).append(name)

    with driver.session() as session:

        for cid,nodes in communities.items():

            summary=", ".join(nodes[:10])

            session.run(
            """
            MERGE (c:Community {id:$cid})
            SET c.summary=$summary
            """,
            {"cid":cid,"summary":summary}
            )

            for n in nodes:

                session.run(
                """
                MATCH (e:Entity {id:$e})
                MATCH (c:Community {id:$cid})
                MERGE (c)-[:CONTAINS]->(e)
                """,
                {"e":n,"cid":cid}
                )