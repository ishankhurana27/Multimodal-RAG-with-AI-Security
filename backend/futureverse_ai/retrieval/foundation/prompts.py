"""
LLM prompt templates for the two-pass tagging pipeline.

Transcribed verbatim from the manual:
  PASS_1_TEMPLATE — Ch. 5.2 "Pass 1 Agent Prompt Specification"
  PASS_2_TEMPLATE — Ch. 5.3 "Pass 2 Agent Prompt Specification"

Only the {placeholders} are substituted; the instruction text is unchanged so
the agent behaviour matches the spec the Academic Lead validates against.
"""
from __future__ import annotations

from futureverse_ai.retrieval.foundation.schema import AttributeVocabularyItem


PASS_1_TEMPLATE = (
    "You are extracting explicit propositions from an NCERT textbook paragraph. "
    "An explicit proposition is a claim directly stated in the text — not implied, "
    "not inferred.\n\n"
    "TEXT:\n{verbatim_text}\n\n"
    "Rules:\n"
    "1. Extract only what is stated. Do not combine sentences to form new claims.\n"
    "2. Each proposition must be traceable to a specific sentence in the text.\n"
    "3. Format: one proposition per line, complete sentence, present tense.\n"
    "4. Between 3 and 10 propositions. If fewer than 3 are present, the chunk may be too short.\n"
    "5. Do NOT extract propositions that require inference from multiple sentences.\n"
    "6. Do NOT restate the same proposition with different words.\n"
    "Output: numbered list of explicit propositions only. No preamble. No explanation."
)


PASS_2_TEMPLATE = (
    "You are identifying integration attributes in an NCERT paragraph. "
    "An integration attribute is a proposition that: (a) is NOT directly stated in "
    "the text, (b) follows logically from combining two or more of the explicit "
    "propositions, (c) matches one of the committed Q-matrix attributes listed below.\n\n"
    "EXPLICIT PROPOSITIONS FROM PASS 1:\n{explicit_propositions}\n\n"
    "VERBATIM TEXT:\n{verbatim_text}\n\n"
    "COMMITTED Q-MATRIX ATTRIBUTES FOR THIS CLUSTER:\n{attribute_vocabulary}\n\n"
    "Task:\n"
    "1. For each committed Q-matrix attribute, determine: Is this attribute "
    "explicitly stated (Pass 1) OR derivable from combining Pass 1 propositions?\n"
    "2. If derivable (not stated): mark it as a derived integration attribute.\n"
    "3. If neither stated nor derivable: mark it as absent from this chunk.\n"
    "Output format:\n"
    "- EXPLICIT: [attribute_id] — covered by proposition [N]\n"
    "- DERIVED: [attribute_id] — derived by combining propositions [N] and [M]\n"
    "- ABSENT: [attribute_id] — not addressable from this chunk\n"
    "Do not include attributes outside the committed vocabulary."
)


def render_pass1(verbatim_text: str) -> str:
    return PASS_1_TEMPLATE.format(verbatim_text=verbatim_text)


def render_pass2(
    verbatim_text: str,
    explicit_propositions: list[str],
    vocabulary: list[AttributeVocabularyItem],
) -> str:
    props = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(explicit_propositions))
    vocab = "\n".join(
        f"- {v.attribute_id} | {v.label}"
        + (" (integration)" if v.is_integration else " (foundational)")
        for v in vocabulary
    )
    return PASS_2_TEMPLATE.format(
        verbatim_text=verbatim_text,
        explicit_propositions=props or "(none extracted)",
        attribute_vocabulary=vocab or "(no committed attributes provided)",
    )
