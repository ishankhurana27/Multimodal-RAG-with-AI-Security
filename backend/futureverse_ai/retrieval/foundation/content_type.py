"""
Content-type classification — Manual Ch. 3 + Appendix C decision tree.

"Content type classification must occur before paragraph-level chunking begins"
(Ch. 3.1). Each extracted element is sorted into one of the manual's classes so
the chunker can apply the right rule and the right serving-pool destination.

Classification is rule-based (no LLM): it uses the box/table/diagram/margin
layout hints when the extractor provides them, and falls back to textual signals
(opening phrases, question markers, header/footer patterns) otherwise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from futureverse_ai.retrieval.foundation.schema import ContentType


@dataclass
class Element:
    """
    One extracted PDF element handed to the classifier.

    The hints mirror what a layout-aware extractor (Ch. 4) can supply. When they
    are unavailable (plain markdown/text), they stay False and classification
    relies on the textual signals below.
    """
    text:          str
    is_boxed:      bool = False   # sits in a sidebar box
    is_table:      bool = False   # grid/table element
    is_diagram:    bool = False   # diagram/flowchart/image
    is_margin:     bool = False   # margin note alongside body
    is_footnote:   bool = False   # bottom-of-page footnote/endnote
    caption:       str  = ""      # caption text for a diagram element
    meta:          dict = field(default_factory=dict)


# Opening-phrase signals (Appendix C). Order matters: enrichment vs recall hinge
# on "Did you know?" vs "Do you know?", so those are matched explicitly.
_RECALL_OPENERS     = ("let us recall", "let's recall", "lets recall", "do you know")
_ENRICHMENT_OPENERS = ("did you know",)
_EXERCISE_OPENERS   = (
    "let's discuss", "lets discuss", "let us discuss",
    "let's do", "lets do", "activity", "project",
    "exercises", "exercise", "questions", "answer the following",
)

_FOOTNOTE_PREFIX_RE = re.compile(r'^\s*(\*|†|‡|\[\d+\]|\d+\.)\s')   # *, †, ‡, [1], 1.
_PAGE_NUM_RE        = re.compile(r'^\s*(page\s*)?\d{1,4}\s*$', re.IGNORECASE)
_HEADER_TOKENS      = ("ncert", "reprint", "rationalised")


def _normalised_opener(text: str) -> str:
    return text.strip().lower()[:40]


def classify(element: Element) -> ContentType:
    """Return the ContentType for one extracted element (Appendix C)."""
    text   = (element.text or "").strip()
    opener = _normalised_opener(text)

    # 1. Layout hints take priority — they are unambiguous when present.
    if element.is_footnote:
        return ContentType.FOOTNOTE
    if element.is_margin:
        return ContentType.MARGIN_NOTE
    if element.is_table or _looks_like_table(text):
        return ContentType.TABLE_ROW
    if element.is_diagram:
        return ContentType.DIAGRAM_CAPTION

    # 2. Header / footer contamination (Ch. 4 Challenge 2, Failure 7).
    if _is_header_footer(text):
        return ContentType.HEADER_FOOTER

    # 3. Boxed / opening-phrase sidebars and exercises.
    if any(opener.startswith(p) for p in _ENRICHMENT_OPENERS):
        return ContentType.SIDEBAR_ENRICHMENT          # "Did you know?" — excluded
    if any(opener.startswith(p) for p in _RECALL_OPENERS):
        return ContentType.SIDEBAR_RECALL              # "Do you know?" / recall
    if any(opener.startswith(p) for p in _EXERCISE_OPENERS):
        return ContentType.EXERCISE                    # discuss/activity/exercises

    # 4. Footnote text by prefix marker even without a layout hint.
    if _FOOTNOTE_PREFIX_RE.match(text) and len(text) < 400:
        return ContentType.FOOTNOTE

    # 5. Pure question blocks (end-of-chapter Qs) are exercises, never body text
    #    (Ch. 3.6: "verbatim_text of a chunk must never contain a question").
    if _is_question_block(text):
        return ContentType.EXERCISE

    # 6. Default: narrative prose → primary serving pool.
    return ContentType.BODY_TEXT


def _is_header_footer(text: str) -> bool:
    if _PAGE_NUM_RE.match(text):
        return True
    low = text.lower()
    if len(text) <= 60 and any(tok in low for tok in _HEADER_TOKENS):
        return True
    return False


def _looks_like_table(text: str) -> bool:
    """Markdown/pipe table detection when no layout hint is given."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    piped = sum(1 for ln in lines if ln.count("|") >= 2)
    return piped >= max(2, len(lines) // 2)


def _is_question_block(text: str) -> bool:
    # A markdown heading may legitimately end in '?' (e.g. "## Why Do We Need a
    # Constitution?") — it is a section label, never an exercise question.
    if text.lstrip().startswith("#"):
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    q_lines = sum(1 for ln in lines if ln.endswith("?"))
    # A block that is mostly questions, or a single short interrogative line.
    if q_lines and q_lines >= max(1, len(lines) // 2):
        return True
    return False


def serving_note(content_type: ContentType) -> str:
    """Human-readable serving destination for logging (Appendix C right column)."""
    return {
        ContentType.BODY_TEXT:          "primary serving pool",
        ContentType.SIDEBAR_RECALL:     "supplementary pool (First Encounter only)",
        ContentType.SIDEBAR_ENRICHMENT: "EXCLUDED from serving (enrichment)",
        ContentType.TABLE_ROW:          "primary pool if Q-matrix relevant",
        ContentType.DIAGRAM_CAPTION:    "primary pool if caption is propositional",
        ContentType.MARGIN_NOTE:        "chunk_context_note metadata only",
        ContentType.EXERCISE:           "exercise review queue — not chunked",
        ContentType.HEADER_FOOTER:      "discarded",
        ContentType.FOOTNOTE:           "chunk_context_note metadata only",
    }.get(content_type, "unknown")
