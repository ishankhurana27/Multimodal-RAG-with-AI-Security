"""
Paragraph-level chunking — Manual Chapter 2 (the primary strategy).

The chunking unit is the NCERT paragraph (Ch. 2.1), not a fixed token count.
Rules implemented here, all from Chapter 2:

  • One paragraph → one chunk by default.
  • Body-text chunks hold 2–6 sentences (Ch. 2.2).
  • Never split a sentence or a proposition (Ch. 2.3) — we only ever cut on
    whole-sentence boundaries.
  • Short paragraphs (≤2 sentences) that cannot stand alone are merged into the
    preceding chunk (Ch. 2.6).
  • Long paragraphs (>6 sentences) are split at a topic shift; if none is
    detectable, after the definitional claim; never immediately before a
    connector sentence (Ch. 2.5).
  • A heading appearing as the first line of a paragraph is absorbed into
    metadata, never into verbatim_text (Ch. 2.4 Case 3).

Comparative/summary tables are chunked one row per chunk with the header row
attached to every chunk (Ch. 3.4).

Semantic topic-shift detection uses sentence embeddings when available; it
degrades gracefully to the definitional-claim rule when the model is absent, so
the chunker runs fully offline.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from futureverse_ai.retrieval.foundation.schema import ContentType
from futureverse_ai.retrieval.foundation.sentences import (
    split_sentences,
    starts_with_connector,
)

log = logging.getLogger(__name__)

MIN_SENTENCES = 2
MAX_SENTENCES = 6


@dataclass
class ChunkDraft:
    """A boundary-correct text chunk, before LLM tagging / metadata enrichment."""
    verbatim_text:      str
    sentences:          list[str]
    paragraph_sequence: int
    content_type:       ContentType = ContentType.BODY_TEXT
    sub_section_title:  str = ""               # absorbed heading (Ch. 2.4 Case 3)
    context_note:       str = ""               # footnote/margin text (Ch. 3.7)
    meta:               dict = field(default_factory=dict)

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)


# ── Paragraph + heading helpers ─────────────────────────────────────────────────

_HEADING_MD_RE = re.compile(r'^\s{0,3}#{1,6}\s+(.*\S)\s*$')


def split_paragraphs(text: str) -> list[str]:
    """Split a block of body text into paragraphs on blank lines."""
    parts = re.split(r'\n\s*\n+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _heading_of(line: str) -> str | None:
    """Return the heading label if `line` is a heading, else None (Ch. 2.4 Case 3)."""
    m = _HEADING_MD_RE.match(line)
    if m:
        return m.group(1).strip()
    # Heuristic for un-marked headings: a short line with no terminal punctuation
    # that is title-case or section-numbered (e.g. "6.1 Nutrition").
    stripped = line.strip()
    if (
        2 <= len(stripped) <= 80
        and not stripped.endswith((".", "!", "?", ":", ";", ","))
        and (re.match(r'^\d+(\.\d+)*\s+\S', stripped) or stripped.istitle())
    ):
        return stripped
    return None


def _absorb_heading(paragraph: str) -> tuple[str, str]:
    """
    Strip a leading heading line from a paragraph.

    Returns (sub_section_title, body_without_heading). The heading is kept as
    metadata and excluded from verbatim_text (Ch. 2.4 Case 3 / Failure 7).
    """
    lines = paragraph.splitlines()
    if not lines:
        return "", paragraph
    head = _heading_of(lines[0])
    if head and len(lines) > 1:
        return head, "\n".join(lines[1:]).strip()
    if head and len(lines) == 1:
        # The whole "paragraph" is just a heading — no body text to chunk.
        return head, ""
    return "", paragraph


# ── Header / running-header contamination cleanup (Ch. 4 Ch. 2, Failure 7) ───────

_RUNNING_HEADER = re.compile(
    r'^(?:Chapter|Unit|Lesson|Section|Module|Page)\s+\d+\b[\s:.\-—]*', re.IGNORECASE
)
_INLINE_HEADING = re.compile(r'^#{1,6}\s*')


def _title_variants(heading: str) -> set[str]:
    """Title strings a contaminating running header might echo (e.g. 'Panchayati Raj')."""
    out: set[str] = set()
    h = (heading or "").strip()
    if not h:
        return out
    out.add(h)
    # Drop a leading 'Chapter N:' / 'Unit N -' prefix to get the bare title.
    m = re.match(r'^(?:Chapter|Unit|Lesson|Section)\s+\d+\s*[:.\-—]?\s*(.+)$', h, re.IGNORECASE)
    if m:
        out.add(m.group(1).strip())
    if ":" in h:
        out.add(h.split(":", 1)[1].strip())
    return {t for t in out if t}


def _strip_header_contamination(text: str, known_titles: "set[str] | frozenset" = frozenset()) -> str:
    """
    Strip a leading running-header / stray-heading artifact from body text.

    NCERT PDF extraction often prepends a page running-header and a stray '#'
    chapter title to the first paragraph of a page, e.g.
        'Chapter 5 # Panchayati Raj What happens after...'
    where the real sentence is 'What happens after...'. We only strip when the
    contamination is unambiguous: a stray '#' marker is present, or a running
    header is immediately followed by a known chapter-title echo. Otherwise the
    text is returned untouched (never eat real sentence words).
    """
    s = text.lstrip()

    m = _RUNNING_HEADER.match(s)
    header_stripped = bool(m)
    if m:
        s = s[m.end():]

    m = _INLINE_HEADING.match(s)
    hash_stripped = bool(m)
    if m:
        s = s[m.end():]

    title_stripped = False
    for t in sorted(known_titles, key=len, reverse=True):
        if t and s.lower().startswith(t.lower()):
            s = s[len(t):].lstrip(" :.-—")
            title_stripped = True
            break

    if hash_stripped or (header_stripped and title_stripped):
        return s.lstrip()
    return text


# A heading marker that the extractor mashed in right after a sentence end ON THE
# SAME LINE, e.g. '...for the village. ## Gram Sabha The Gram Sabha meeting...'.
# Uses [ \t]+ (not \s+) so it never touches a proper blank-line-separated heading
# like '...India.\n\n## Introduction' — those must keep their heading status.
_INLINE_HEADING_BOUNDARY = re.compile(r'(?<=[.!?])[ \t]+(#{1,6})\s+([^\n]+)')


def _split_inline_title(words: list[str]) -> tuple[str, str]:
    """
    Split 'Gram Sabha The Gram Sabha meeting...' → ('Gram Sabha', 'The Gram Sabha meeting...').

    NCERT writes a sub-heading then immediately re-states it ('Heading. The
    Heading is...'). We detect that echo: the shortest 1–5 word title that
    reappears within the next few words is the heading; everything from the title
    onward is the body. No echo → ('', original) so the caller keeps the text.
    """
    n = len(words)
    norm = [w.lower().strip(".,;:()") for w in words]
    # Scan longest title first so 'Gram Sabha' wins over the shorter 'Gram' echo.
    for k in range(min(5, n - 1), 0, -1):
        title = norm[:k]
        for start in range(k, min(k + 6, n - k + 1)):
            if norm[start:start + k] == title:
                return " ".join(words[:k]), " ".join(words[k:])
    return "", " ".join(words)


def _normalize_inline_headings(text: str) -> str:
    """
    Turn sentence-boundary inline headings into clean blank-line-delimited blocks.

    '...village. ## Gram Sabha The Gram Sabha meeting...'  →
    '...village.\\n\\n## Gram Sabha\\n\\nThe Gram Sabha meeting...'
    so the heading is absorbed into metadata and a new sub-section chunk begins
    (manual Ch. 2.4 / Failure 7). If no title can be recovered, the stray marker
    is dropped and the text kept as a new paragraph (never lose words).
    """
    def _repl(m: re.Match) -> str:
        marker, rest = m.group(1), m.group(2)
        title, body = _split_inline_title(rest.split())
        if title:
            return f"\n\n{marker} {title}\n\n{body}"
        return f"\n\n{body}"

    return _INLINE_HEADING_BOUNDARY.sub(_repl, text)


# ── Semantic topic-shift detection (optional embeddings) ─────────────────────────

def _adjacent_similarities(sentences: list[str]) -> list[float] | None:
    """
    Cosine similarity between each adjacent sentence pair (Ch. 2.4 / 2.5).

    Uses a locally-cached sentence-transformer if present; returns None when the
    model is unavailable so the caller falls back to the rule-based split.
    """
    if len(sentences) < 2:
        return None
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = getattr(_adjacent_similarities, "_model", None)
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            _adjacent_similarities._model = model
        emb = model.encode(sentences, normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)
        return [float(emb[i] @ emb[i + 1]) for i in range(len(sentences) - 1)]
    except Exception as exc:
        log.debug("Topic-shift embeddings unavailable, using rule fallback: %s", exc)
        return None


def _best_cut(sentences: list[str]) -> int:
    """
    Choose a split index for a long paragraph (Ch. 2.5).

    Valid cuts keep both halves ≥ MIN_SENTENCES and never place a connector
    sentence at the start of the second half (Rule 3). Among valid cuts, the
    lowest adjacent similarity (biggest topic gap) wins; with no embeddings, the
    cut nearest the definitional boundary (after sentence 3–4) is used.
    """
    n = len(sentences)
    candidates = [
        c for c in range(MIN_SENTENCES, n - MIN_SENTENCES + 1)
        if not starts_with_connector(sentences[c])
    ]
    if not candidates:
        # Connectors block every cut — fall back to the midpoint.
        return n // 2

    sims = _adjacent_similarities(sentences)
    if sims:
        # similarity at boundary before index c is sims[c-1]
        return min(candidates, key=lambda c: sims[c - 1])

    # Rule fallback: split after the primary definitional claim (sentence 3 or 4).
    return min(candidates, key=lambda c: abs(c - 4))


def _split_long(sentences: list[str]) -> list[list[str]]:
    """Recursively split a >6-sentence paragraph into 2–6 sentence pieces."""
    if len(sentences) <= MAX_SENTENCES:
        return [sentences]
    cut = _best_cut(sentences)
    return _split_long(sentences[:cut]) + _split_long(sentences[cut:])


# ── Body-text chunking ───────────────────────────────────────────────────────────

def chunk_body_text(
    text: str,
    start_sequence: int = 0,
    known_titles: "set[str] | frozenset" = frozenset(),
) -> list[ChunkDraft]:
    """
    Apply the full Chapter 2 paragraph strategy to a block of body text.

    `start_sequence` is the paragraph_sequence of the first paragraph in `text`
    so positions stay correct across a whole chapter. `known_titles` carries the
    surrounding chapter/section titles so leading running-header contamination
    (Ch. 4 / Failure 7) can be stripped from the first sentence.
    """
    paragraphs = split_paragraphs(text)

    # Stage 1 — paragraph → sentence groups, absorbing headings, splitting longs.
    groups: list[ChunkDraft] = []
    block_titles = set(known_titles)
    current_subsection = ""
    seq = start_sequence
    for para in paragraphs:
        head, body = _absorb_heading(para)
        if head:
            current_subsection = head
            block_titles |= _title_variants(head)
        if not body:
            continue
        body = _strip_header_contamination(body, block_titles)
        if not body:
            continue
        sents = split_sentences(body)
        if not sents:
            continue
        for piece in _split_long(sents):
            groups.append(ChunkDraft(
                verbatim_text=" ".join(piece),
                sentences=piece,
                paragraph_sequence=seq,
                content_type=ContentType.BODY_TEXT,
                sub_section_title=current_subsection,
            ))
        seq += 1

    # Stage 2 — merge short paragraphs / connector-led pieces into the previous
    # chunk (Ch. 2.6 + Ch. 2.5 Rule 3).
    merged: list[ChunkDraft] = []
    for g in groups:
        must_merge = (
            merged
            and (
                g.sentence_count == 1
                or starts_with_connector(g.sentences[0])
                or (g.sentence_count <= MIN_SENTENCES
                    and merged[-1].sentence_count + g.sentence_count <= MAX_SENTENCES
                    and starts_with_connector(g.sentences[0]))
            )
        )
        if must_merge:
            _absorb_into(merged[-1], g)
        else:
            merged.append(g)

    # Stage 3 — final orphan repair: no chunk may have < MIN_SENTENCES (Ch. 2.2).
    repaired: list[ChunkDraft] = []
    for g in merged:
        if g.sentence_count < MIN_SENTENCES and repaired:
            _absorb_into(repaired[-1], g)
        else:
            repaired.append(g)
    # A lone leading orphan: pull the next chunk's first sentences back is unsafe,
    # so merge it forward into the following chunk instead.
    if len(repaired) >= 2 and repaired[0].sentence_count < MIN_SENTENCES:
        _absorb_into(repaired[1], repaired[0], prepend=True)
        repaired.pop(0)

    return repaired


def _absorb_into(target: ChunkDraft, extra: ChunkDraft, prepend: bool = False) -> None:
    """Merge `extra`'s sentences into `target` (Ch. 2.6 merge)."""
    if prepend:
        target.sentences = extra.sentences + target.sentences
    else:
        target.sentences = target.sentences + extra.sentences
    target.verbatim_text = " ".join(target.sentences)
    if extra.sub_section_title and not target.sub_section_title:
        target.sub_section_title = extra.sub_section_title


# ── Table chunking (Ch. 3.4) ──────────────────────────────────────────────────────

def chunk_table(text: str, start_sequence: int = 0) -> list[ChunkDraft]:
    """
    Chunk a pipe/markdown table one row per chunk, header attached (Ch. 3.4).

    Each chunk's verbatim_text is '[Header: ...] [Row: ...]' so the row is
    independently interpretable (prevents Failure 9 — orphaned table data).
    """
    rows = []
    for ln in text.splitlines():
        if "|" not in ln:
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        # Skip markdown separator rows (---|---).
        if all(re.fullmatch(r':?-{2,}:?', c or "") for c in cells if c):
            continue
        if any(cells):
            rows.append(cells)
    if len(rows) < 2:
        return []

    header = rows[0]
    header_str = " | ".join(header)
    drafts: list[ChunkDraft] = []
    seq = start_sequence
    for row in rows[1:]:
        row_str = " | ".join(row)
        verbatim = f"[Header: {header_str}] [Row: {row_str}]"
        drafts.append(ChunkDraft(
            verbatim_text=verbatim,
            sentences=[verbatim],     # a table row is one structural unit
            paragraph_sequence=seq,
            content_type=ContentType.TABLE_ROW,
        ))
        seq += 1
    return drafts
