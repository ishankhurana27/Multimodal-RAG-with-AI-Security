"""
Integration test stubs (TEMPLATE — place at backend/tests/integration/).

These use pytest-docker (already a dependency) to bring up ephemeral Weaviate
and Neo4j, then exercise the real ingest→index→query and KG round-trips.
Marked skip until you wire docker-compose.test.yml + service URLs.

Run (nightly):  pytest tests/integration -v
"""
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 (and provide test services) to run",
)


@pytest.fixture(scope="session")
def weaviate_url():
    # With pytest-docker: yield the mapped host:port after a readiness poll of
    # /v1/.well-known/ready. Placeholder for the static case:
    return os.environ.get("TEST_WEAVIATE_URL", "http://localhost:8080")


def test_vector_roundtrip(weaviate_url):
    """ingest a tiny doc → embed → index → query returns it."""
    # 1. build 2-3 Documents, run run_ingestion_pipeline against a temp collection
    # 2. query_doc_with_hybrid_search(query) -> assert the seeded chunk is top-1
    # 3. assert post-insert verified_stored == expected (catches silent Weaviate drops)
    pytest.skip("wire pipeline_agents.run_ingestion_pipeline against test Weaviate")


def test_kg_roundtrip():
    """extract_and_store a paragraph → get_graph_source surfaces its entities."""
    # 1. extract_and_store(text, doc_id, subject=..., chapter=...)
    # 2. get_graph_source([query]) -> assert an [Entity] line for a known term
    # 3. assert no parameterized var-length Cypher errors (Phase 1 H-8 regression)
    pytest.skip("wire extractor.extract_and_store against test Neo4j")


def test_embedding_dim_matches_collection(weaviate_url):
    """Guard for Phase 1 H-6: query-vector dim must equal the collection's dim."""
    pytest.skip("assert len(query_vector) == stored vector dim for the active model")
