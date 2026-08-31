#!/usr/bin/env python3
"""
Clean extraction "reading view" markdown into pure book content.

The chapter .md files rendered from the extraction JSON contain scaffolding
that must NOT be embedded into the vector DB: block headers (p./leaf/block),
italic metadata lines (heading_level, raw_extraction_note, ...), 'Needs
review' callouts, publisher-ad image records, the TOC section, and render
statistics. This script strips all of it, dedups headings that repeat as
CAPS body text, and outputs clean markdown ready for the `heading` chunker.

Usage:
    python scripts/clean_chapter_md.py ch05_preamble.md            # -> ch05_preamble.clean.md
    python scripts/clean_chapter_md.py chapters\\ --out cleaned\\    # whole folder
    python scripts/clean_chapter_md.py ch05.md --inplace           # overwrite original
"""
import argparse
import re
import sys
from pathlib import Path

# Block-position headers: **p. 86 · leaf 73 · block 2** · Linear Text ...
BLOCK_HEADER_RE = re.compile(r"^\*\*p\.\s*\d+.*block\s*\d+\*\*", re.IGNORECASE)

# Italic metadata lines: *heading_level: H3 · raw_extraction_note: ...*
METADATA_KEYS = (
    "heading_level", "element_type", "continues_on_page", "continues_from_previous_page",
    "raw_extraction_note", "ocr_fix", "chapter_number", "case_reference",
    "constitutional_article_cited", "constitutional_amendment_cited", "quote_source",
    "historical_references", "visual_confirmation", "content_type_rationale",
    "chapter_end", "paragraph_sequence",
)
METADATA_LINE_RE = re.compile(
    r"^\*(?=[^*]).*(?:" + "|".join(METADATA_KEYS) + r")\s*:.*\*\s*$"
)

# Render-header / review scaffolding
SCAFFOLD_RES = [
    re.compile(r"^Rendered from `.*`"),
    re.compile(r"^JSON \d+\s*KB"),
    re.compile(r"^>\s*This is a reading view"),
    re.compile(r"^>\s*[⚠!].*Needs review", re.IGNORECASE),
    re.compile(r"^---\s*$"),
]

# Publisher-ad image records (keep genuine diagrams/maps)
AD_IMAGE_RE = re.compile(
    r"^\[Image:.*(promotional|banner|advertis|courseware|coursewise|publisher|"
    r"click for|play.?button|digital resource).*\]",
    re.IGNORECASE,
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _norm(text: str) -> str:
    """Alnum-only lowercase form, for comparing headings to CAPS body echoes."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def clean_markdown(text: str) -> str:
    out: list[str] = []
    all_headings: set[str] = set()  # normalized forms of every heading seen
    in_toc = False

    for line in text.splitlines():
        stripped = line.strip()

        m = HEADING_RE.match(stripped)
        if m:
            title = m.group(2).strip()
            # Drop the 'Sections in this chapter' TOC section entirely
            if _norm(title) == _norm("Sections in this chapter"):
                in_toc = True
                continue
            in_toc = False
            all_headings.add(_norm(title))
            # Also remember the title without leading list numbers ("1. Sovereign")
            bare = re.sub(r"^(chapter\s*\d+:?\s*|\d+\.?\s*)", "", title, flags=re.IGNORECASE)
            if bare:
                all_headings.add(_norm(bare))
            out.append(line)
            continue

        if in_toc:
            # Skip TOC bullets and blanks until the next heading
            continue

        if not stripped:
            out.append("")
            continue

        if BLOCK_HEADER_RE.match(stripped):
            continue
        if METADATA_LINE_RE.match(stripped):
            continue
        if AD_IMAGE_RE.match(stripped):
            continue
        if any(rx.match(stripped) for rx in SCAFFOLD_RES):
            continue

        # Drop body lines that merely repeat a heading in CAPS
        # (e.g. '## Text of the Preamble' followed by 'TEXT OF THE PREAMBLE',
        #  or 'CHAPTER 5: ...' echoed under '## Chapter Opening')
        n = _norm(stripped)
        bare_body = re.sub(r"^(chapter\s*\d+:?\s*|\d+\.?\s*)", "", stripped, flags=re.IGNORECASE)
        if n in all_headings or (_norm(bare_body) in all_headings and len(stripped) < 80):
            continue

        out.append(line)

    # Collapse 3+ consecutive blank lines to one
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"
    return cleaned


def process_file(src: Path, dst: Path) -> tuple[int, int]:
    original = src.read_text(encoding="utf-8")
    cleaned = clean_markdown(original)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(cleaned, encoding="utf-8")
    return len(original), len(cleaned)


def main():
    ap = argparse.ArgumentParser(description="Strip extraction scaffolding from chapter markdown")
    ap.add_argument("input", help="A .md file or a folder of .md files")
    ap.add_argument("--out", default=None,
                    help="Output file or folder (default: <name>.clean.md alongside input)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite the input file(s)")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"Not found: {src}")

    files = sorted(src.glob("*.md")) if src.is_dir() else [src]
    if not files:
        sys.exit(f"No .md files in {src}")

    for f in files:
        if f.name.endswith(".clean.md") and not args.inplace:
            continue
        if args.inplace:
            dst = f
        elif args.out:
            outp = Path(args.out)
            dst = (outp / f.name) if (src.is_dir() or outp.is_dir() or not outp.suffix) else outp
        else:
            dst = f.with_suffix("").with_suffix("")  # strip .md
            dst = f.parent / (f.stem + ".clean.md")
        before, after = process_file(f, dst)
        pct = 100 * (before - after) / max(before, 1)
        print(f"{f.name}: {before:,} -> {after:,} chars  (-{pct:.0f}% scaffolding)  => {dst}")


if __name__ == "__main__":
    main()
