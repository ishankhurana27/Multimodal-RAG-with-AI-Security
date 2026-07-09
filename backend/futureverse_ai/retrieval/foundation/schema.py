"""
Foundation NCERT chunk schema.

Faithful implementation of Appendix A (ncert_chunks) and Appendix D
(proposition_sentence_map JSON schema) of the
"Infinity Foundation — NCERT Corpus Chunking Strategy Manual · V1.0".

Every field name and rule below maps 1:1 to the manual so a chunk produced by
this pipeline can be persisted to the ncert_chunks table without translation.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ── Enums (Appendix A: content_type, cognitive_frame) ───────────────────────────

class ContentType(str, Enum):
    """content_type ENUM — determines serving-pool assignment (Manual Ch. 3)."""
    BODY_TEXT          = "body_text"            # primary serving pool
    SIDEBAR_RECALL     = "sidebar_recall"       # supplementary pool (First Encounter)
    SIDEBAR_ENRICHMENT = "sidebar_enrichment"   # EXCLUDED from serving
    TABLE_ROW          = "table_row"            # one chunk per row + header
    DIAGRAM_CAPTION    = "diagram_caption"      # only if caption is propositional
    MARGIN_NOTE        = "margin_note"          # metadata only, never verbatim_text
    # Non-chunked element classes (Appendix C) — never become ncert_chunks rows,
    # but the classifier returns them so the pipeline can route/discard correctly.
    EXERCISE           = "exercise"             # routed to exercise review, not chunked
    HEADER_FOOTER      = "header_footer"        # stripped/discarded
    FOOTNOTE           = "footnote"             # metadata only


# Element classes that are never emitted as a servable chunk (Ch. 3, Appendix C).
NON_CHUNKED_TYPES = frozenset({
    ContentType.EXERCISE,
    ContentType.HEADER_FOOTER,
    ContentType.FOOTNOTE,
})

# Element classes excluded from the serving pool but still stored (enrichment).
EXCLUDED_FROM_SERVING = frozenset({
    ContentType.SIDEBAR_ENRICHMENT,
})


class CognitiveFrame(str, Enum):
    """cognitive_frame ENUM — secondary retrieval filter (Manual Ch. 5.5)."""
    FOUNDATIONAL = "foundational"
    INTEGRATIVE  = "integrative"
    ANALYTICAL   = "analytical"


# ── Attribute vocabulary (Q-matrix input to Pass 2) ─────────────────────────────

@dataclass
class AttributeVocabularyItem:
    """
    One committed Q-matrix attribute (an AKU node) that Pass 2 may tag against.

    The manual (Appendix A) stores attribute_tags as "AKU node IDs", so this is
    the bridge between the corpus pipeline and the AKU / Agent-Orchestration side.
    Pass 2 is Q-matrix-constrained: it can only assign IDs that exist here.
    """
    attribute_id:   str            # UUID of the AKU node / Q-matrix attribute
    label:          str            # human-readable label (used in prompts + map)
    cluster_id:     str            # topic_cluster_id this attribute belongs to
    is_integration: bool = False   # True for integration attributes


# ── proposition_sentence_map entry (Appendix D) ─────────────────────────────────

@dataclass
class PropositionMapEntry:
    """One entry of proposition_sentence_map keyed by attribute_id (Appendix D)."""
    sentence_indices:              list[int]
    attribute_id:                  str
    attribute_label:               str
    is_integration:                bool
    is_derived:                    bool
    trim_if_confirmed:             bool
    confirmation_evidence_quality: str = "primary"   # "primary" | "secondary"
    derivation_sources:            Optional[list[int]] = None  # null if not derived

    def to_dict(self) -> dict:
        return {
            "sentence_indices":              self.sentence_indices,
            "attribute_id":                  self.attribute_id,
            "attribute_label":               self.attribute_label,
            "is_integration":                self.is_integration,
            "is_derived":                    self.is_derived,
            "derivation_sources":            self.derivation_sources,
            "trim_if_confirmed":             self.trim_if_confirmed,
            "confirmation_evidence_quality": self.confirmation_evidence_quality,
        }


def _now_iso() -> str:
    """ISO 8601 UTC timestamp (Appendix A: created_at format)."""
    return datetime.now(timezone.utc).isoformat()


# ── The chunk record (Appendix A: ncert_chunks) ─────────────────────────────────

@dataclass
class NcertChunk:
    """
    One ncert_chunks row. Field set and constraints are from Appendix A.

    Mandatory-at-creation fields are required positionally; validation-gate and
    deprecation fields default to their "not yet" values. A chunk is only
    servable once academic_lead_validated is True (set by a human, Ch. 8).
    """
    # ── Mandatory (M) ──────────────────────────────────────────────────────────
    source_reference:    str                  # 'NCERT Class N, Subject, Chapter N: Title'
    ncert_class:         int                  # 6–12
    subject_id:          str                  # FK → gbl_subject.subject_id
    chapter_number:      int
    chapter_title:       str
    paragraph_sequence:  int                  # position of paragraph within chapter
    content_type:        ContentType
    verbatim_text:       str                  # EXACT NCERT text — never edited
    sentence_count:      int                  # 2–6 for body_text
    cognitive_frame:     CognitiveFrame = CognitiveFrame.FOUNDATIONAL
    topic_cluster_ids:   list[str] = field(default_factory=list)
    attribute_tags:      list[str] = field(default_factory=list)   # AKU node IDs, min 1
    embedding_model_version: str = ""         # set by the embedding stage

    # ── Optional (O) — populated by the two-pass tagging stage ───────────────────
    explicit_propositions:          list[str] = field(default_factory=list)   # Pass 1
    integration_attributes_derived: list[str] = field(default_factory=list)   # Pass 2
    proposition_sentence_map:       dict = field(default_factory=dict)        # Appendix D
    chunk_context_note:             Optional[str] = None   # margin/footnote/author notes

    # ── Embedding (Appendix A) ───────────────────────────────────────────────────
    # embedding_vector is populated by the bge-m3 embedding stage at index time;
    # held here as Optional so a freshly-chunked record is complete per Appendix A.
    embedding_vector: Optional[list[float]] = None   # pgvector vector(N)

    # ── System / identity ───────────────────────────────────────────────────────
    chunk_id:   str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_now_iso)

    # ── Validation gate (Ch. 8) ──────────────────────────────────────────────────
    academic_lead_validated: bool = False
    academic_lead_id:        Optional[str] = None
    validated_at:            Optional[str] = None

    # ── Never-Delete deprecation (Ch. 9) ─────────────────────────────────────────
    deprecated_at:      Optional[str] = None
    deprecated_reason:  Optional[str] = None
    successor_chunk_id: Optional[str] = None

    def is_multi_attribute(self) -> bool:
        """Manual Ch. 6 / Failure 4: a chunk addressing >1 attribute."""
        return len(self.attribute_tags) > 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["content_type"]    = self.content_type.value
        d["cognitive_frame"] = self.cognitive_frame.value
        return d
