"""
Two-pass attribute tagging — Manual Chapter 5.

Pass 1 (LLM): extract explicit propositions from verbatim_text (Ch. 5.2).
Pass 2 (LLM): derive integration attributes + build the attribute_tags array,
              constrained to the committed Q-matrix vocabulary (Ch. 5.3–5.4).
Then, without the LLM:
  • assign cognitive_frame by Rules 1–3 (Ch. 5.5);
  • build proposition_sentence_map with trim_if_confirmed flags (Ch. 5.6 / App. D).

Pass 2 is Q-matrix-constrained: with no committed vocabulary it cannot produce
valid attribute_tags, and the manual marks such a chunk un-servable. The caller
is told via TaggingResult.warnings.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from futureverse_ai.retrieval.foundation.chunker import ChunkDraft
from futureverse_ai.retrieval.foundation.llm import call_llm
from futureverse_ai.retrieval.foundation.prompts import render_pass1, render_pass2
from futureverse_ai.retrieval.foundation.schema import (
    AttributeVocabularyItem,
    CognitiveFrame,
    PropositionMapEntry,
)

log = logging.getLogger(__name__)

_ANALYTICAL_CUES = (
    "implication", "implications", "consequence", "consequences",
    "as a result", "leads to", "results in", "thereby", "structural",
)


@dataclass
class TaggingResult:
    explicit_propositions:          list[str]
    attribute_tags:                 list[str]
    integration_attributes_derived: list[str]
    cognitive_frame:                CognitiveFrame
    proposition_sentence_map:       dict
    warnings:                       list[str] = field(default_factory=list)


# ── Output parsers ────────────────────────────────────────────────────────────

def parse_pass1(raw: str) -> list[str]:
    """Parse the numbered proposition list returned by Pass 1."""
    props: list[str] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip a leading "1.", "1)", "-", "•" enumerator.
        m = re.match(r'^\s*(?:\d+[.)]|[-•*])\s*(.+)$', line)
        text = m.group(1).strip() if m else line
        if len(text) > 3:
            props.append(text)
    return props


def _resolve_attribute(remainder: str, vocab: list[AttributeVocabularyItem]):
    """Match a Pass 2 output fragment to a committed vocabulary item."""
    low = remainder.lower()
    # Prefer an exact attribute_id hit, then a label substring hit.
    for v in vocab:
        if v.attribute_id and v.attribute_id.lower() in low:
            return v
    for v in vocab:
        if v.label and v.label.lower() in low:
            return v
    return None


def _proposition_refs(remainder: str) -> list[int]:
    """Pull 'proposition [N]' / 'propositions [N] and [M]' references (1-based)."""
    refs = re.findall(r'proposition[s]?\s*\[?(\d+)\]?(?:\s*(?:and|,)\s*\[?(\d+)\]?)?',
                      remainder, flags=re.IGNORECASE)
    out: list[int] = []
    for a, b in refs:
        if a:
            out.append(int(a))
        if b:
            out.append(int(b))
    return out


def parse_pass2(raw: str, vocab: list[AttributeVocabularyItem]):
    """
    Parse Pass 2 output into structured tagging facts.

    Returns a dict keyed by attribute_id → {kind, prop_refs, item}, where kind is
    'explicit' or 'derived' (ABSENT lines are dropped).
    """
    findings: dict[str, dict] = {}
    for line in (raw or "").splitlines():
        m = re.match(r'^\s*[-*•]?\s*(EXPLICIT|DERIVED|ABSENT)\s*[:\-—]\s*(.+)$',
                     line.strip(), flags=re.IGNORECASE)
        if not m:
            continue
        kind = m.group(1).lower()
        remainder = m.group(2)
        if kind == "absent":
            continue
        item = _resolve_attribute(remainder, vocab)
        if not item:
            continue
        findings[item.attribute_id] = {
            "kind":      kind,                      # 'explicit' | 'derived'
            "prop_refs": _proposition_refs(remainder),
            "item":      item,
        }
    return findings


# ── Proposition → sentence alignment (for the sentence map) ──────────────────────

_WORD_RE = re.compile(r"[a-z]{3,}")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def align_propositions_to_sentences(
    propositions: list[str],
    sentences: list[str],
) -> list[int]:
    """
    Map each Pass 1 proposition to the sentence index it was drawn from.

    Uses word-overlap (Jaccard) — no model required. Returns a list parallel to
    `propositions`; index i is the best-matching sentence for proposition i.
    """
    sent_tokens = [_tokens(s) for s in sentences]
    result: list[int] = []
    for prop in propositions:
        pt = _tokens(prop)
        best_idx, best_score = 0, -1.0
        for i, st in enumerate(sent_tokens):
            if not st and not pt:
                score = 0.0
            else:
                inter = len(pt & st)
                union = len(pt | st) or 1
                score = inter / union
            if score > best_score:
                best_idx, best_score = i, score
        result.append(best_idx)
    return result


# ── Cognitive frame (Ch. 5.5 Rules 1–3) ─────────────────────────────────────────

def assign_cognitive_frame(
    tagged_items: list[AttributeVocabularyItem],
    verbatim_text: str,
) -> CognitiveFrame:
    """
    Rule 1: any integration attribute present → integrative.
    Rule 2: only foundational attributes → foundational.
    Rule 3 (heuristic): integration present + analytical cue language → analytical.
    Rule 4 (human override) happens later at the Academic Lead gate.
    """
    has_integration = any(v.is_integration for v in tagged_items)
    if not has_integration:
        return CognitiveFrame.FOUNDATIONAL
    low = verbatim_text.lower()
    if any(cue in low for cue in _ANALYTICAL_CUES):
        return CognitiveFrame.ANALYTICAL
    return CognitiveFrame.INTEGRATIVE


# ── Proposition-sentence map (Ch. 5.6 / Appendix D) ──────────────────────────────

def build_proposition_sentence_map(
    findings: dict,
    propositions: list[str],
    prop_to_sentence: list[int],
) -> dict:
    """Construct proposition_sentence_map keyed by attribute_id (Appendix D)."""
    out: dict = {}
    for attr_id, fact in findings.items():
        item        = fact["item"]
        is_derived  = fact["kind"] == "derived"
        refs        = [r for r in fact["prop_refs"] if 1 <= r <= len(propositions)]
        # Convert 1-based proposition refs → sentence indices.
        sent_idx = sorted({prop_to_sentence[r - 1] for r in refs}) if refs else []
        entry = PropositionMapEntry(
            sentence_indices=sent_idx,
            attribute_id=attr_id,
            attribute_label=item.label,
            is_integration=item.is_integration,
            is_derived=is_derived,
            # Integration attribute sentence is never trimmed; foundational is.
            trim_if_confirmed=not item.is_integration,
            confirmation_evidence_quality="secondary" if is_derived else "primary",
            derivation_sources=sent_idx if is_derived else None,
        )
        out[attr_id] = entry.to_dict()
    return out


# ── Orchestration of the two passes for one chunk ────────────────────────────────

def tag_chunk(
    draft: ChunkDraft,
    vocabulary: list[AttributeVocabularyItem],
    *,
    model: str = "",
) -> TaggingResult:
    """Run Pass 1 + Pass 2 + frame + sentence-map for a single body/table chunk."""
    warnings: list[str] = []

    # Pass 1 — explicit propositions.
    raw1 = call_llm(render_pass1(draft.verbatim_text), model=model)
    propositions = parse_pass1(raw1) if raw1 else []
    if not propositions:
        warnings.append("Pass 1 returned no propositions (LLM unavailable or chunk too short)")

    # Pass 2 — integration derivation, constrained to committed vocabulary.
    findings: dict = {}
    if vocabulary:
        raw2 = call_llm(
            render_pass2(draft.verbatim_text, propositions, vocabulary), model=model
        )
        findings = parse_pass2(raw2, vocabulary) if raw2 else {}
    else:
        warnings.append(
            "No Q-matrix vocabulary supplied — Pass 2 skipped; attribute_tags empty "
            "(chunk is NOT servable per Manual Ch. 5.4)"
        )

    attribute_tags = list(findings.keys())
    derived = [a for a, f in findings.items()
               if f["kind"] == "derived" and f["item"].is_integration]
    tagged_items = [f["item"] for f in findings.values()]

    frame = assign_cognitive_frame(tagged_items, draft.verbatim_text)

    prop_to_sentence = align_propositions_to_sentences(propositions, draft.sentences)
    psm = build_proposition_sentence_map(findings, propositions, prop_to_sentence)

    if not attribute_tags and vocabulary:
        warnings.append("Zero attribute tags after Pass 2 (Failure-mode: under-tagging)")

    return TaggingResult(
        explicit_propositions=propositions,
        attribute_tags=attribute_tags,
        integration_attributes_derived=derived,
        cognitive_frame=frame,
        proposition_sentence_map=psm,
        warnings=warnings,
    )
