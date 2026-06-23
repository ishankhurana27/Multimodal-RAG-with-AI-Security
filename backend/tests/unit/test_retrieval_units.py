"""
Unit tests for pure RAG logic (TEMPLATE — place at backend/tests/unit/).

No live services. These target the exact functions most likely to regress and
that the Phase 1 audit flagged. Adjust imports if module paths differ.
"""
import pytest

from futureverse_ai.retrieval.graph import extractor, graph_rag
from futureverse_ai.retrieval import utils


# ── extractor: entity gate ──────────────────────────────────────────────────
class TestEntityGate:
    def test_rejects_sentence_fragments(self):
        assert extractor._is_noise_entity("the monsoon is the seasonal reversal") is True

    def test_rejects_generic_hub_singletons(self):
        assert extractor._is_noise_entity("india") is True
        assert extractor._is_noise_entity("world") is True

    def test_keeps_real_entities(self):
        assert extractor._is_noise_entity("plate tectonics") is False
        assert extractor._is_noise_entity("western ghats") is False

    def test_validate_and_clean_drops_leading_quantifiers(self):
        ents = [{"id": "various rivers"}, {"id": "monsoon"}, {"id": "the himalayas"}]
        clean, _ = extractor.validate_and_clean_extraction(ents, [])
        ids = {e["id"] for e in clean}
        assert "monsoon" in ids
        assert "various rivers" not in ids and "the himalayas" not in ids

    def test_relation_normalization_maps_to_canonical(self):
        _, rels = extractor.validate_and_clean_extraction(
            [], [{"from": "a", "to": "b", "type": "leads to"}]
        )
        assert rels[0]["type"] == "RESULTS_IN"


# ── graph_rag: query helpers ────────────────────────────────────────────────
class TestGraphRagHelpers:
    def test_escape_lucene_escapes_specials(self):
        out = graph_rag._escape_lucene('article+21 (rights)')
        assert "\\+" in out and "\\(" in out and "\\)" in out

    def test_decompose_expands_aliases(self):
        terms = graph_rag._decompose_upsc_queries(["isostatic adjustment"])
        assert any("isostasy" in t for t in terms)

    def test_decompose_caps_term_count(self):
        terms = graph_rag._decompose_upsc_queries(["a b c d e f g h i j k l m n o p"])
        assert len(terms) <= 15


# ── utils: metadata filter + context header ─────────────────────────────────
class TestUtilsHelpers:
    def test_metadata_filter_detects_subject(self):
        f = utils._build_metadata_filter("geography of india", [{"subject": "Geography"}])
        assert f.get("subject") == "Geography"

    def test_metadata_filter_empty_when_field_absent(self):
        # No sample doc has subject/class → no filter (prevents zero-result filters)
        assert utils._build_metadata_filter("class 10 science", [{"text": "x"}]) == {}

    def test_context_header_builds_source_line(self):
        header = utils._build_context_header(
            {"Header 1": "Climate", "Header 2": "Monsoon",
             "position_in_section": 0, "child_chunk_ids": "a,b,c"}
        )
        assert "Climate > Monsoon" in header and "paragraph 1 of 3" in header


# ── embedding dispatch (would have caught Phase 1 C-1: dead CUDA branch) ─────
class TestEmbeddingDispatch:
    @pytest.mark.parametrize("engine", ["", "ollama", "openai", "azure_openai"])
    def test_known_engines_return_callable(self, engine):
        fn = utils.get_embedding_function(
            engine, "model", embedding_function=(lambda *a, **k: []),
            url="http://x", key="", embedding_batch_size=4,
        )
        assert callable(fn)

    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError):
            utils.get_embedding_function(
                "not_an_engine", "m", None, "u", "", 4,
            )

    def test_sentence_transformers_engine_is_reachable(self):
        # Regression guard for C-1: 'sentence_transformers' must resolve to a
        # working branch, not be shadowed by the "" branch with a None model.
        fn = utils.get_embedding_function(
            "sentence_transformers", "m", embedding_function=None,
            url="", key="", embedding_batch_size=4,
        )
        assert callable(fn)
