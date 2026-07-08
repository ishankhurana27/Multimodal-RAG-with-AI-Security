"""
Shared pytest fixtures for the FutureverseAI backend test suite (TEMPLATE).

Place at backend/tests/conftest.py. These stubs let unit tests run with NO live
Weaviate / Neo4j / Ollama. Integration tests (tests/integration) spin up real
services via pytest-docker instead.

Run:  cd backend && pytest tests/unit -v
"""
from types import SimpleNamespace
import pytest


@pytest.fixture
def app_config():
    """Minimal stand-in for request.app.state.config used across retrieval code."""
    return SimpleNamespace(
        CHUNK_SIZE=1000,
        CHUNK_OVERLAP=100,
        CHUNKING_STRATEGY="hierarchical",
        TEXT_SPLITTER="",
        ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER=True,
        CHUNK_MIN_SIZE_TARGET=0,
        ENABLE_CONTEXTUAL_CHUNKING=False,
        ENABLE_KEYWORD_EXTRACTION=True,
        KEYWORD_EXTRACTION_METHOD="regex",
        RAG_EMBEDDING_ENGINE="ollama",
        RAG_EMBEDDING_MODEL="nomic-embed-text",
        RAG_EMBEDDING_BATCH_SIZE=8,
        ENABLE_ASYNC_EMBEDDING=True,
    )


@pytest.fixture
def fake_request(app_config):
    """request-like object exposing app.state.config / ef / main_loop."""
    state = SimpleNamespace(config=app_config, ef=None, main_loop=None)
    return SimpleNamespace(app=SimpleNamespace(state=state))


class _StubVectorClient:
    """In-memory stand-in for VECTOR_DB_CLIENT (search/get/has_collection/insert)."""
    def __init__(self):
        self.collections = {}
    def has_collection(self, collection_name):
        return collection_name in self.collections
    def insert(self, collection_name, items):
        self.collections.setdefault(collection_name, []).extend(items)
    def get(self, collection_name):
        items = self.collections.get(collection_name, [])
        return SimpleNamespace(ids=[[i["id"] for i in items]])
    def query(self, collection_name, filter=None, limit=1):
        return SimpleNamespace(ids=[[]])


@pytest.fixture
def stub_vector_client():
    return _StubVectorClient()


@pytest.fixture
def stub_neo4j(monkeypatch):
    """Patch extractor.run_query so graph code runs without a live Neo4j.

    Returns a list you can append canned row-dicts to; run_query pops FIFO.
    """
    canned = []
    def _fake_run_query(cypher, params=None):
        return canned.pop(0) if canned else []
    monkeypatch.setattr(
        "futureverse_ai.retrieval.graph.extractor.run_query",
        _fake_run_query, raising=False,
    )
    return canned
