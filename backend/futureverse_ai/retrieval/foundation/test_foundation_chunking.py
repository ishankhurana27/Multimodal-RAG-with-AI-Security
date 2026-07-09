"""
Rule-engine tests for the Foundation chunker (no LLM required).

Covers the deterministic boundary rules from the manual: sentence splitting
(Ch. 2.3 / 4.3), paragraph chunking + 2–6 sentence sizing (Ch. 2.2), long-
paragraph splitting (Ch. 2.5), short-paragraph merge (Ch. 2.6), the connector
rule (Ch. 2.5 Rule 3), heading absorption (Ch. 2.4 Case 3), table-row chunking
(Ch. 3.4), and content-type classification (Ch. 3 / Appendix C).

Run:  pytest futureverse_ai/retrieval/foundation/test_foundation_chunking.py
"""
from futureverse_ai.retrieval.foundation.chunker import (
    chunk_body_text, chunk_table, MIN_SENTENCES, MAX_SENTENCES,
)
from futureverse_ai.retrieval.foundation.sentences import (
    split_sentences, dehyphenate, starts_with_connector,
)
from futureverse_ai.retrieval.foundation.content_type import Element, classify
from futureverse_ai.retrieval.foundation.schema import ContentType

_PARA = (
    "The Finance Commission is constituted under Article 280. "
    "It is constituted every five years. "
    "Its function is to recommend tax distribution. "
    "It recommends grants-in-aid to States. "
    "Its recommendations go to the President."
)


def test_dehyphenation():
    assert "constitution" in dehyphenate("consti-\ntution")


def test_sentence_split_protects_initials():
    s = split_sentences("It is constituted under Article 280. K. C. Neogy chaired it.")
    assert len(s) == 2


def test_single_paragraph_single_chunk():
    c = chunk_body_text(_PARA)
    assert len(c) == 1
    assert c[0].sentence_count == 5


def test_long_paragraph_splits_into_valid_pieces():
    long_para = " ".join(f"Sentence number {i} states a distinct fact here." for i in range(1, 9))
    c = chunk_body_text(long_para)
    assert len(c) > 1
    assert all(MIN_SENTENCES <= x.sentence_count <= MAX_SENTENCES for x in c)
    assert sum(x.sentence_count for x in c) == 8     # no sentence lost


def test_short_connector_paragraph_merges():
    text = _PARA + "\n\nThese recommendations are addressed to both governments."
    c = chunk_body_text(text)
    assert len(c) == 1
    assert "These recommendations" in c[0].verbatim_text


def test_no_chunk_starts_with_connector():
    long_para = " ".join(f"Sentence number {i} states a distinct fact here." for i in range(1, 9))
    text = long_para + "\n\n" + _PARA + "\n\nThese recommendations are addressed to all."
    c = chunk_body_text(text)
    assert all(not starts_with_connector(x.sentences[0]) for x in c)


def test_heading_absorption():
    htext = ("## 8.1 Composition of the Commission\n"
             "The Commission has a chairman and four members. "
             "They are appointed by the President.")
    c = chunk_body_text(htext)
    assert c[0].sub_section_title.endswith("Commission")
    assert "8.1 Composition" not in c[0].verbatim_text


def test_table_row_chunking():
    table = ("| Dimension | Finance Commission | Planning Commission |\n"
             "| Status | Statutory body | Extra-constitutional |\n"
             "| Binding | Advisory | Advisory |")
    c = chunk_table(table)
    assert len(c) == 2
    assert all("[Header:" in x.verbatim_text for x in c)
    assert all(x.content_type == ContentType.TABLE_ROW for x in c)


def test_strips_running_header_contamination():
    # PDF running-header artifact mashed onto the first sentence (Failure 7).
    from futureverse_ai.retrieval.foundation.pipeline import build_drafts
    text = ("## Chapter 1: Panchayati Raj\n\n"
            "Chapter 5 # Panchayati Raj What happens after people elect their "
            "representatives? How are decisions made? Let us look at rural areas.")
    drafts = build_drafts(text)
    assert drafts[0].verbatim_text.startswith("What happens after")
    assert not any("#" in d.verbatim_text for d in drafts)


def test_splits_inline_subheading():
    # Sub-heading inlined after a sentence becomes its own sub-section chunk.
    from futureverse_ai.retrieval.foundation.pipeline import build_drafts
    text = ("## Chapter 1: Panchayati Raj\n\n"
            "Leaders have planned for the village. They meet often. They decide together. "
            "## Gram Sabha The Gram Sabha meeting begins with the Sarpanch. "
            "All members attend. They discuss problems.")
    drafts = build_drafts(text)
    assert any(d.sub_section_title == "Gram Sabha" for d in drafts)
    assert any(d.verbatim_text.startswith("The Gram Sabha meeting begins") for d in drafts)
    assert not any("#" in d.verbatim_text for d in drafts)


def test_content_type_classification():
    assert classify(Element("Did you know? The first FC was in 1951.")) == ContentType.SIDEBAR_ENRICHMENT
    assert classify(Element("Do you know? Recall what you learnt about taxes.")) == ContentType.SIDEBAR_RECALL
    assert classify(Element("Activity: discuss the role of the FC.")) == ContentType.EXERCISE
    assert classify(Element("142")) == ContentType.HEADER_FOOTER
    assert classify(Element("The Finance Commission recommends tax devolution.")) == ContentType.BODY_TEXT
    assert classify(Element("What is the primary function of the Finance Commission?")) == ContentType.EXERCISE
