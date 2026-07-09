"""
Foundation NCERT corpus chunking pipeline.

Implements the "Infinity Foundation — NCERT Corpus Chunking Strategy Manual V1.0":
paragraph-level chunking (Ch. 2), content-type aware processing (Ch. 3),
two-pass LLM attribute tagging + cognitive frame + proposition_sentence_map
(Ch. 5), producing ncert_chunks records (Appendix A).

Entry point: run_chapter(ChapterInput) -> (list[NcertChunk], PipelineReport).
"""
from futureverse_ai.retrieval.foundation.schema import (
    AttributeVocabularyItem,
    CognitiveFrame,
    ContentType,
    NcertChunk,
)
from futureverse_ai.retrieval.foundation.pipeline import (
    ChapterInput,
    PipelineReport,
    run_chapter,
)

__all__ = [
    "AttributeVocabularyItem",
    "CognitiveFrame",
    "ContentType",
    "NcertChunk",
    "ChapterInput",
    "PipelineReport",
    "run_chapter",
]
