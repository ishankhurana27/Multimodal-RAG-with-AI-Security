"""
Foundation chunking pipeline orchestrator.

Turns one NCERT chapter (raw extracted text + chapter metadata + the committed
Q-matrix vocabulary) into validated-ready ncert_chunks records, following the
manual end to end:

  1. Segment into elements and classify each (Ch. 3, Appendix C).
  2. Discard / route non-chunked classes (exercises, headers, footnotes, margins).
  3. Paragraph-level chunk the servable body / sidebar-recall / table elements (Ch. 2).
  4. Two-pass LLM tagging + cognitive_frame + proposition_sentence_map (Ch. 5).
  5. Assemble NcertChunk records (Appendix A); academic_lead_validated stays False
     until a human approves them (Ch. 8).

The pipeline never persists or embeds — those are existing/external stages. It
returns the records plus a PipelineReport so callers can see what was produced,
excluded, or warned about.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from futureverse_ai.retrieval.foundation.chunker import (
    ChunkDraft,
    chunk_body_text,
    chunk_table,
    _heading_of,
    _title_variants,
    _normalize_inline_headings,
    MIN_SENTENCES,
    MAX_SENTENCES,
)
from futureverse_ai.retrieval.foundation.sentences import starts_with_connector
from futureverse_ai.retrieval.foundation.content_type import (
    Element,
    classify,
    serving_note,
)
from futureverse_ai.retrieval.foundation.schema import (
    AttributeVocabularyItem,
    ContentType,
    CognitiveFrame,
    NcertChunk,
    NON_CHUNKED_TYPES,
)
from futureverse_ai.retrieval.foundation.sentences import split_sentences
from futureverse_ai.retrieval.foundation.tagging import tag_chunk

log = logging.getLogger(__name__)

# Servable content types that go through paragraph chunking + tagging.
_TAGGABLE = {
    ContentType.BODY_TEXT,
    ContentType.SIDEBAR_RECALL,
    ContentType.TABLE_ROW,
    ContentType.DIAGRAM_CAPTION,
}


@dataclass
class ChapterInput:
    raw_text:          str
    ncert_class:       int
    subject_id:        str
    subject_name:      str
    chapter_number:    int
    chapter_title:     str
    topic_cluster_ids: list[str] = field(default_factory=list)
    vocabulary:        list[AttributeVocabularyItem] = field(default_factory=list)


@dataclass
class PipelineReport:
    produced:        int = 0
    excluded:        int = 0          # enrichment / not-chunked
    by_type:         dict = field(default_factory=dict)
    warnings:        list[str] = field(default_factory=list)

    def bump(self, ct: ContentType) -> None:
        self.by_type[ct.value] = self.by_type.get(ct.value, 0) + 1


def _source_reference(c: ChapterInput) -> str:
    # Appendix A: 'NCERT Class [N], [Subject], Chapter [N]: [Title]'
    return (
        f"NCERT Class {c.ncert_class}, {c.subject_name}, "
        f"Chapter {c.chapter_number}: {c.chapter_title}"
    )


def _segment(raw_text: str) -> list[str]:
    """Split a chapter into coarse elements on blank lines (pre-classification)."""
    import re
    return [b.strip() for b in re.split(r'\n\s*\n+', raw_text.strip()) if b.strip()]


def _same_body_section(a: ChunkDraft, b: ChunkDraft) -> bool:
    return (a.content_type == ContentType.BODY_TEXT
            and b.content_type == ContentType.BODY_TEXT
            and a.sub_section_title == b.sub_section_title)


def _absorb(target: ChunkDraft, extra: ChunkDraft) -> None:
    target.sentences = target.sentences + extra.sentences
    target.verbatim_text = " ".join(target.sentences)
    if extra.context_note and not target.context_note:
        target.context_note = extra.context_note


def _merge_short_paragraphs(drafts: list[ChunkDraft]) -> list[ChunkDraft]:
    """
    Merge one-sentence / connector-led body paragraphs into a neighbour (Ch. 2.6).

    NCERT body chunks must be 2–6 sentences (Ch. 2.2); a standalone one-sentence
    paragraph is decontextualised. Because each paragraph arrives as its own block,
    this cross-block pass does the merging the manual requires. Merges stay within
    the same sub-section and never combine across a heading boundary.
    """
    # Pass 1 — greedy backward merge while there is room (≤ MAX_SENTENCES).
    out: list[ChunkDraft] = []
    for d in drafts:
        mergeable = (
            d.content_type == ContentType.BODY_TEXT
            and d.sentences
            and (d.sentence_count < MIN_SENTENCES or starts_with_connector(d.sentences[0]))
        )
        if (out and mergeable and _same_body_section(out[-1], d)
                and out[-1].sentence_count + d.sentence_count <= MAX_SENTENCES):
            _absorb(out[-1], d)
        else:
            out.append(d)

    # Pass 2 — any leftover sub-minimum body orphan folds into the previous
    # same-section chunk even if it nudges slightly past MAX (better than a
    # one-sentence chunk).
    result: list[ChunkDraft] = []
    for d in out:
        if (d.content_type == ContentType.BODY_TEXT
                and d.sentence_count < MIN_SENTENCES
                and result and _same_body_section(result[-1], d)):
            _absorb(result[-1], d)
        else:
            result.append(d)
    return result


def build_drafts(raw_text: str, report: "PipelineReport | None" = None) -> list[ChunkDraft]:
    """
    Segment → classify → route → chunk a block of text into ChunkDrafts.

    This is the structural (no-LLM) core of the strategy: content-type aware
    paragraph chunking (Ch. 2–3). Non-chunked classes (exercises, headers,
    footnotes, margins) are dropped or folded into context notes. Reused by both
    run_chapter() and the ingestion-pipeline ChunkingAgent.
    """
    drafts: list[ChunkDraft] = []
    pending_context: list[str] = []   # margin/footnote text awaiting a body chunk
    current_subsection = ""            # running heading carried across blocks (Ch. 2.4)
    seq = 0

    # Split extractor-inlined sub-headings into clean blocks before segmenting.
    raw_text = _normalize_inline_headings(raw_text)

    for block in _segment(raw_text):
        ct = classify(Element(text=block))
        if report is not None:
            report.bump(ct)

        # Not-chunked classes (Appendix C): discard / route, never a record.
        if ct in NON_CHUNKED_TYPES:
            if ct == ContentType.FOOTNOTE:
                pending_context.append(block)   # → chunk_context_note (Ch. 3.7)
            if report is not None:
                report.excluded += 1
            log.info("skip [%s] → %s", ct.value, serving_note(ct))
            continue
        if ct == ContentType.MARGIN_NOTE:
            pending_context.append(block)
            if report is not None:
                report.excluded += 1
            continue

        # Servable / storable classes → produce drafts.
        if ct == ContentType.TABLE_ROW:
            new = chunk_table(block, start_sequence=seq)
        elif ct in (ContentType.BODY_TEXT, ContentType.SIDEBAR_RECALL):
            # A heading on its own line is not a chunk — remember it as the
            # running sub-section and attach it to following body chunks (Ch. 2.4).
            lines = [ln for ln in block.splitlines() if ln.strip()]
            if len(lines) == 1 and _heading_of(lines[0]):
                current_subsection = _heading_of(lines[0])
                continue
            new = chunk_body_text(
                block, start_sequence=seq,
                known_titles=_title_variants(current_subsection),
            )
            for d in new:
                if d.sub_section_title:
                    current_subsection = d.sub_section_title   # heading inside block
                elif current_subsection:
                    d.sub_section_title = current_subsection    # inherit running heading
                if ct == ContentType.SIDEBAR_RECALL:
                    d.content_type = ContentType.SIDEBAR_RECALL
        elif ct == ContentType.DIAGRAM_CAPTION:
            sents = split_sentences(block)
            new = [ChunkDraft(block, sents, seq, content_type=ContentType.DIAGRAM_CAPTION)]
        else:  # SIDEBAR_ENRICHMENT — stored, excluded from serving, not tagged.
            sents = split_sentences(block)
            new = [ChunkDraft(block, sents, seq, content_type=ContentType.SIDEBAR_ENRICHMENT)]

        # Attach any pending footnote/margin text to the first new body chunk.
        if pending_context and new:
            new[0].context_note = "\n".join(pending_context)
            pending_context.clear()

        drafts.extend(new)
        seq += max(1, len(new))

    return _merge_short_paragraphs(drafts)


def run_chapter(chapter: ChapterInput, *, model: str = "") -> tuple[list[NcertChunk], PipelineReport]:
    """Process one chapter into ncert_chunks records + a report."""
    report = PipelineReport()
    src    = _source_reference(chapter)

    drafts = build_drafts(chapter.raw_text, report)

    # Build records (+ tag the taggable ones).
    chunks: list[NcertChunk] = []
    for d in drafts:
        chunk = NcertChunk(
            source_reference=src,
            ncert_class=chapter.ncert_class,
            subject_id=chapter.subject_id,
            chapter_number=chapter.chapter_number,
            chapter_title=chapter.chapter_title,
            paragraph_sequence=d.paragraph_sequence,
            content_type=d.content_type,
            verbatim_text=d.verbatim_text,
            sentence_count=d.sentence_count,
            topic_cluster_ids=list(chapter.topic_cluster_ids),
            chunk_context_note=d.context_note or None,
        )

        # body_text sentence-count guard (Ch. 2.2).
        if d.content_type == ContentType.BODY_TEXT and not (2 <= d.sentence_count <= 6):
            report.warnings.append(
                f"seq {d.paragraph_sequence}: body chunk has {d.sentence_count} "
                f"sentences (outside 2–6)"
            )

        if d.content_type in _TAGGABLE:
            tr = tag_chunk(d, chapter.vocabulary, model=model)
            chunk.explicit_propositions          = tr.explicit_propositions
            chunk.attribute_tags                 = tr.attribute_tags
            chunk.integration_attributes_derived = tr.integration_attributes_derived
            chunk.cognitive_frame                = tr.cognitive_frame
            chunk.proposition_sentence_map       = tr.proposition_sentence_map
            for w in tr.warnings:
                report.warnings.append(f"seq {d.paragraph_sequence}: {w}")
        else:
            # Enrichment: never an attribute anchor (Ch. 3.3).
            chunk.cognitive_frame = CognitiveFrame.FOUNDATIONAL

        chunks.append(chunk)

    report.produced = len(chunks)
    return chunks, report
