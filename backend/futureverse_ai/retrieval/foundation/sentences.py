"""
Sentence boundary detection + dehyphenation for the Foundation chunker.

The manual's #1 rule (Ch. 2.3) is that a chunk must never split a sentence or a
proposition. Everything downstream operates on whole sentences, so a reliable
sentence splitter is the foundation of correct chunk boundaries.

Two public helpers:
  dehyphenate(text)      — Ch. 4 Challenge 3: repair words broken across lines.
  split_sentences(text)  — return a list of complete sentences.

Offline-first: tries NLTK punkt if its data is present, otherwise falls back to
a regex splitter that protects common NCERT abbreviations and decimals. No
network access is required at runtime.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Abbreviations whose trailing period must NOT end a sentence. Lower-cased match.
_ABBREVIATIONS = {
    "art", "arts", "no", "nos", "sec", "secs", "fig", "figs", "eq", "vol",
    "e.g", "i.e", "etc", "vs", "viz", "cf", "al", "ed", "pp", "p",
    "dr", "mr", "mrs", "ms", "prof", "govt", "ltd", "pvt", "co", "inc",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "approx", "ch", "chap",
}

# Connector words that link a sentence to the one before it (Ch. 2.5 Rule 3).
# A sentence starting with one of these must never be the first sentence of a
# new chunk — it belongs with the sentence it connects to.
CONNECTORS = {
    "however", "therefore", "thus", "hence", "because", "consequently",
    "moreover", "furthermore", "additionally", "besides", "accordingly",
    "this", "these", "it", "its", "they", "their", "such", "that",
    "so", "then", "also", "meaning",
}

_SENT_END_RE = re.compile(r'([.!?])(["\')\]]?)(\s+)(?=[A-Z(0-9"\'])')
_PLACEHOLDER = "\x01"   # stand-in for a protected period


def dehyphenate(text: str) -> str:
    """
    Join words split by a hard hyphen at a line break (Ch. 4 Challenge 3).

    'consti-\\ntution' → 'constitution'. Only joins when the hyphen sits between
    two alphabetic fragments and the continuation begins lower-case, which is the
    NCERT typesetting pattern; genuine compound hyphens (mid-line) are untouched.
    """
    return re.sub(r'([A-Za-z])-\s*\n\s*([a-z])', r'\1\2', text)


def _protect_abbreviations(text: str) -> str:
    """Replace sentence-internal periods (abbreviations, decimals, initials)."""
    # Decimals: 12.5 → 12<ph>5
    text = re.sub(r'(\d)\.(\d)', rf'\1{_PLACEHOLDER}\2', text)
    # Single-capital initials: 'B. R. Ambedkar'
    text = re.sub(r'\b([A-Z])\.', rf'\1{_PLACEHOLDER}', text)

    def _repl(m: re.Match) -> str:
        word = m.group(1)
        if word.lower() in _ABBREVIATIONS:
            return f"{word}{_PLACEHOLDER}"
        return m.group(0)

    # Word followed by a period — protect only known abbreviations.
    text = re.sub(r'\b([A-Za-z]+)\.', _repl, text)
    return text


def _restore(text: str) -> str:
    return text.replace(_PLACEHOLDER, ".")


def _split_regex(text: str) -> list[str]:
    """Regex sentence splitter with abbreviation/decimal protection."""
    protected = _protect_abbreviations(text)
    # Insert a split marker after every genuine sentence terminator.
    marked = _SENT_END_RE.sub(r'\1\2\n', protected)
    parts = [_restore(s).strip() for s in marked.split("\n")]
    return [s for s in parts if s]


def _split_nltk(text: str) -> list[str] | None:
    """Use NLTK punkt if its data is already downloaded; else return None."""
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        # Probe for punkt data without triggering a download.
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                return None
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        return None


def split_sentences(text: str) -> list[str]:
    """
    Split text into a list of complete sentences.

    Whitespace is normalised first (PDF extraction often injects stray newlines
    mid-sentence). Returns [text] when no terminator is found so a single-clause
    line is still a usable one-sentence unit.
    """
    cleaned = re.sub(r'\s+', ' ', dehyphenate(text)).strip()
    if not cleaned:
        return []

    sentences = _split_nltk(cleaned)
    if sentences is None:
        sentences = _split_regex(cleaned)

    return sentences or [cleaned]


def sentence_count(text: str) -> int:
    return len(split_sentences(text))


def starts_with_connector(sentence: str) -> bool:
    """True if the sentence opens with a cross-sentence connector (Ch. 2.5)."""
    m = re.match(r"[\"'(]*([A-Za-z]+)", sentence.strip())
    if not m:
        return False
    return m.group(1).lower() in CONNECTORS
