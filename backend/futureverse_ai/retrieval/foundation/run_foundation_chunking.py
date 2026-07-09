"""
Runnable demo / smoke test for the Foundation chunking pipeline.

Uses the manual's own Finance Commission example (Ch. 5.2 / Appendix D) plus an
enrichment box, a comparative table, and an exercise block to exercise every
content-type path.

Run from the backend/ directory:
    python -m futureverse_ai.retrieval.foundation.run_foundation_chunking

If Ollama + qwen2.5:7b are running, Pass 1 / Pass 2 produce real tags. If not,
the structural chunking still runs and the LLM stages degrade with warnings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running the file directly as well as via -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from futureverse_ai.retrieval.foundation import (  # noqa: E402
    AttributeVocabularyItem,
    ChapterInput,
    run_chapter,
)
from futureverse_ai.retrieval.foundation.llm import TAGGING_MODEL, is_available  # noqa: E402


SAMPLE_CHAPTER = """\
# Chapter 8: The Finance Commission

The Finance Commission is constituted by the President of India under Article 280 \
of the Constitution. It is constituted every five years or earlier if necessary. \
The primary function of the Finance Commission is to recommend the distribution of \
net proceeds of taxes between the Union and States. It also recommends grants-in-aid \
to States in need of assistance. The recommendations of the Finance Commission are \
addressed to the President of India.

Did you know? The first Finance Commission was constituted in 1951 under the \
chairmanship of K. C. Neogy.

| Dimension | Finance Commission | Planning Commission |
| Constitutional Status | Statutory body under Article 280 | Extra-constitutional body |
| Nature of Recommendations | Advisory - not binding | Advisory - no binding authority |

Answer the following questions. What is the primary function of the Finance Commission?
"""

VOCABULARY = [
    AttributeVocabularyItem("FA1-constitutional-basis-uuid",
                            "Constitutional basis — Article 280", "polity-fincomm", False),
    AttributeVocabularyItem("FA2-quinquennial-uuid",
                            "Quinquennial constitution", "polity-fincomm", False),
    AttributeVocabularyItem("FA3-primary-function-uuid",
                            "Primary function — tax devolution recommendations",
                            "polity-fincomm", False),
    AttributeVocabularyItem("IA6-advisory-not-binding-uuid",
                            "Advisory character — not binding on government",
                            "polity-fincomm", True),
]


def main() -> None:
    # Windows consoles default to cp1252; avoid crashes on any stray unicode.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    chapter = ChapterInput(
        raw_text=SAMPLE_CHAPTER,
        ncert_class=11,
        subject_id="polity-subject-uuid",
        subject_name="Political Science",
        chapter_number=8,
        chapter_title="The Finance Commission",
        topic_cluster_ids=["polity-fincomm"],
        vocabulary=VOCABULARY,
    )

    avail = is_available()
    print("=" * 70)
    print(f"Ollama model '{TAGGING_MODEL}' available: {avail}"
          + ("" if avail else "  -> tagging will be skipped (structure only)"))
    print("=" * 70)

    chunks, report = run_chapter(chapter)

    print(f"\nPRODUCED {report.produced} chunk(s) | EXCLUDED {report.excluded}")
    print(f"By content type: {report.by_type}")
    if report.warnings:
        print("\nWARNINGS:")
        for w in report.warnings:
            print(f"  - {w}")

    print("\n" + "-" * 70)
    for i, c in enumerate(chunks, 1):
        print(f"\n### CHUNK {i}  [{c.content_type.value}]  seq={c.paragraph_sequence}")
        print(f"  source_reference : {c.source_reference}")
        print(f"  cognitive_frame  : {c.cognitive_frame.value}")
        print(f"  sentence_count   : {c.sentence_count}")
        print(f"  attribute_tags   : {c.attribute_tags}")
        print(f"  verbatim_text    : {c.verbatim_text[:160]}"
              + ("..." if len(c.verbatim_text) > 160 else ""))
        if c.explicit_propositions:
            print(f"  explicit_props   : {len(c.explicit_propositions)}")
        if c.proposition_sentence_map:
            print("  proposition_sentence_map:")
            print(json.dumps(c.proposition_sentence_map, indent=4))

    # Full JSON of the first servable chunk for schema inspection.
    if chunks:
        print("\n" + "=" * 70)
        print("FULL ncert_chunks RECORD (chunk 1):")
        print(json.dumps(chunks[0].to_dict(), indent=2))


if __name__ == "__main__":
    main()
