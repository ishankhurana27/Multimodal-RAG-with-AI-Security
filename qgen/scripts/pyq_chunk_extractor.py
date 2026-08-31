"""
pyq_chunk_extractor.py — retrieve the exact RAG chunks behind every PYQ.

For each past-year question it runs the SAME dual-channel retrieval the
question generator uses (dense vector + Weaviate BM25 over raw_text/text,
merged and deduped), once per RETRIEVAL UNIT — the stem, each numbered
statement, each assertion/reason, each match pair, and each contentful
option — and writes every hit with its FULL chunk text to one local JSON
file.

    # smoke test first (no GPU, no Weaviate): proves parsing + schema
    python qgen\\scripts\\pyq_chunk_extractor.py --self-test

    # see exactly which queries would run, without embedding anything
    python qgen\\scripts\\pyq_chunk_extractor.py --input my_50_questions.json --dry-run

    # the real run
    python qgen\\scripts\\pyq_chunk_extractor.py --input my_50_questions.json ^
        --out backend\\data\\pyq_chunk_map.json --k 8 --md

WHY UNITS, NOT JUST "QUESTION + OPTIONS"
────────────────────────────────────────
In UPSC Prelims only some options carry content. A multi-statement item's
options are key codes ("1 and 2 only", "All three"); an assertion-reason
item's options are relations ("Both A and R are true..."). Embedding those
retrieves noise and buries the real evidence. The content in those formats
lives in the STATEMENTS, the ASSERTION and the REASON, and in match items
in the List-I/List-II pairs — so those become their own retrieval units,
and key-code options are recorded with `queried: false` plus the reason,
rather than silently dropped. Straight select/identify options ARE content
and are queried individually. Nothing is skipped without saying so in the
output: --query-key-codes forces them to be queried anyway.

INPUT — any of these shapes is auto-detected
────────────────────────────────────────────
  • JSON list of question objects, or {"questions": [...]} / {"data": [...]}
  • JSONL (one object per line)
  • normalized SFT shape:  stem / statements[] / options[] / format / answer
  • ingested PYQ shape:    question / options[] / command_word / answer
  • loose shapes: any of question|stem|text|prompt for the stem, and
    options|choices|opts (list, or dict keyed a/b/c/d) for the options
Statements embedded in the stem text ("1. ... 2. ... 3. ...", roman or
arabic) are split out, and the trailing code line ("Which of the statements
given above is/are correct?") is stripped from the last statement.

OUTPUT — one JSON file
──────────────────────
  meta      : run provenance (input, collections, model, k, retriever, counts)
  chunks    : chunk_key -> {text (FULL), metadata, collection, ...} — every
              distinct chunk stored ONCE, so the file has no truncation
              anywhere and can be diffed between runs
  questions : per question — its units, each unit's ranked hits (chunk_key,
              rank, channel, and by default the full text inline too),
              a union_chunks list deduped across all its units, and a
              `coverage` block (does the retrieved evidence actually contain
              the answer?) for quality assurance
Use --no-inline-text to keep chunk text only in the `chunks` map.

RETRIEVAL PARITY. `_dual_channel_retrieve` below is a verbatim copy of
routers/questions.py::_dual_channel_retrieve_sync with one addition: it
labels which channel produced each hit. --router-retriever imports the real
function instead (no labels), and --verify-retriever runs both on the same
queries and reports any divergence, so the copy can never drift unnoticed.
Junk chunks are filtered exactly as the generator filters them
(_is_junk_chunk); --no-junk-filter keeps them.

Stdlib only, apart from the backend's own vector client and (for the real
run) sentence-transformers — both already installed for the backend.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

SCRIPT_VERSION = "pyq-chunks-1"


# ══════════════════════════════════════════════════════════════════════════
# repo location + .env config (never printed)
# ══════════════════════════════════════════════════════════════════════════

def find_repo_root(start: Path) -> Path:
    for cand in [start] + list(start.parents):
        if (cand / "backend" / "futureverse_ai").is_dir():
            return cand
    return start


ROOT = find_repo_root(Path(__file__).resolve().parent)


def load_env(root: Path) -> dict:
    """Read .env for Weaviate / embedding settings. Values are used, never
    logged — the file holds real secrets."""
    env = {}
    path = root / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in env.items():                 # real env wins
        os.environ.setdefault(k, v)
    return env


# ══════════════════════════════════════════════════════════════════════════
# question normalisation
# ══════════════════════════════════════════════════════════════════════════

_STEM_KEYS = ("stem", "question", "text", "prompt", "question_text", "q")
_OPT_KEYS = ("options", "choices", "opts", "option_list")
_ST_KEYS = ("statements", "statement_list", "sts")

# "Which of the statements given above is/are correct?", "Select the correct
# answer using the code given below", "How many of the above are correct?" —
# the instruction tail, not a statement.
_TAIL_RE = re.compile(
    r"(which\s+of\s+(the|these|those)\s+(statements?|above|pairs?|subjects?)|"
    r"how\s+many\s+of\s+(the|these)\s+(above|pairs?|statements?)|"
    r"how\s+many\s+(pairs?|statements?)\s+(given\s+)?above|"
    r"select\s+the\s+correct|choose\s+the\s+correct|"
    r"identify\s+the\s+correct|"
    r"the\s+correct\s+answer\s+(is|using)|"
    r"using\s+the\s+code\s+given|codes?\s*:|"
    r"is\s+required\s+in\s+respect\s+of|"
    r"are\s+correctly\s+matched|"
    r"given\s+above\s+(is|are)\s+correct|"
    r"consider\s+the\s+following\s+(statements?|pairs?)\s*:?\s*$)",
    re.IGNORECASE)

# A real paper often runs the instruction straight on from the last list item
# with no punctuation: "III. Conditions of the Governor's office For a
# constitutional amendment ... in respect of which of the above?" — a
# capitalised clause opener after a lowercase word is the seam. Cutting there
# is only safe when what follows really is the instruction, so the caller
# also requires a tail cue (or a '?') in the discarded remainder.
_INSTR_SEAM_RE = re.compile(
    r"(?<=[a-z’'\)])\s+(For|Which|How|Select|Choose|Identify|Arrange|What|"
    r"With|In|Given|Using|Answer|Consider|Regarding|If)\b")

# Numbered statement markers: "1." "2)" "(3)" "I." "II)" — and lowercase
# roman ("i.", "ii."), which real papers mix with arabic in the same item
# ("... consider the following statements : i. ... 2. ..."). A stray "v."
# from a case citation ("Golaknath v. State") can match the token, but a
# marker only counts inside a run that starts at 1, so it never triggers.
_NUM_MARK_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s;:.\)]))\(?(\d{1,2}|[IVXivx]{1,4})[\.\)]\s+")

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
          "VII": 7, "VIII": 8, "IX": 9, "X": 10}

# match-the-following pair line: "A. Something — 1. Other" / "1. X : Y"
_PAIR_SPLIT_RE = re.compile(r"\s*[—–\-:]{1,2}\s*")

_FORMAT_ALIASES = {
    "select/identify": "select_identify",
    "select_identify": "select_identify",
    "mcq": "select_identify",
    "direct mcq": "select_identify",
    "direct": "select_identify",
    "single correct": "select_identify",
    "consider the following statements": "multi_statement",
    "multi_statement": "multi_statement",
    "multi-statement": "multi_statement",
    "multiple statements": "multi_statement",
    "statement based": "multi_statement",
    "statement_based": "multi_statement",
    "statement-based": "multi_statement",
    "assertion-reason": "assertion_reason",
    "assertion_reason": "assertion_reason",
    "assertion-reasoning": "assertion_reason",
    "assertion reasoning": "assertion_reason",
    "assertion and reason": "assertion_reason",
    "match the following": "match_the_following",
    "match_the_following": "match_the_following",
    "match-the-following": "match_the_following",
    "matching": "match_the_following",
    "arrange in chronological/logical order": "sequence",
    "chronological order": "sequence",
    "sequence": "sequence",
}

_FORMAT_KEYS = ("format", "command_word", "question-type", "question_type",
                "qtype", "type", "q_type", "category")


def _first(d: dict, keys) -> object:
    for k in keys:
        if k in d and d[k] not in (None, "", [], {}):
            return d[k]
    return None


def _as_text_list(raw) -> list:
    """Statements may be plain strings or objects — the normalized SFT rows
    store each as {"text": ..., "truth": true|false}. Stringifying those
    would push "', 'truth': False}" into the query text, so the text field
    is pulled out explicitly.
    """
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw[k] for k in sorted(raw)]
    if isinstance(raw, str):
        raw = [raw]
    out = []
    for s in raw:
        if isinstance(s, dict):
            s = _first(s, ("text", "statement", "value", "content", "s")) or ""
        s = " ".join(str(s).split()).strip()
        if s:
            out.append(s)
    return out


def _as_option_list(raw) -> list:
    """Options may be a list, or a dict keyed a/b/c/d (or 1/2/3/4)."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        def order(k):
            ks = str(k).strip().lower().lstrip("(").rstrip(")").rstrip(".")
            return (0, ord(ks) - 97) if ks[:1].isalpha() and len(ks) == 1 \
                else (1, ks)
        return [str(raw[k]).strip() for k in sorted(raw, key=order)
                if str(raw[k]).strip()]
    if isinstance(raw, (list, tuple)):
        out = []
        for o in raw:
            if isinstance(o, dict):          # [{"label":"a","text":"..."}]
                out.append(str(_first(o, ("text", "option", "value",
                                          "label")) or "").strip())
            else:
                out.append(str(o).strip())
        return [o for o in out if o]
    return [str(raw).strip()]


def resolve_answer(raw: dict, options: list) -> tuple:
    """(letter, answer_text) from whatever shape the file uses.

    Real exports mark the answer as "b", "B", "(b)", "b.", "option_b", the
    1-based index 2, the 0-based index 1, or only as the answer's text —
    and plenty of rows have no answer at all. Every one of those must end
    up either resolved or cleanly empty: this used to do
    `if ans[:1] in "abcd"`, and because `"" in "abcd"` is True in Python, a
    row with no answer crashed the whole run inside ord().
    """
    val = _first(raw, ("answer", "correct", "correct_option", "correct_answer",
                       "key", "ans", "answer_key", "correct_choice"))
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    txt = str(_first(raw, ("answer_text", "correct_text",
                           "answer_option_text")) or "").strip()

    letter = ""
    idx = None
    if isinstance(val, bool):
        val = None
    if isinstance(val, int):
        idx = val
    elif isinstance(val, str):
        s = val.strip().strip("().[]:").rstrip(".").strip()
        low = s.lower()
        if low.startswith("option"):                 # "option_b", "Option B"
            low = low[6:].strip(" _-:")
        if len(low) == 1 and low.isalpha():
            letter = low
        elif low.isdigit():
            idx = int(low)
        elif options:                                # answer given as text
            for i, o in enumerate(options):
                if strip_option_label(o).strip().lower() == low:
                    letter = chr(97 + i)
                    txt = txt or strip_option_label(o)
                    break
        if not letter and idx is None and low[:1].isalpha() and options \
                and len(low) > 1:
            txt = txt or s                           # free-text answer

    if idx is not None and options:
        # 1-based when it fits, else 0-based; anything else is unusable
        if 1 <= idx <= len(options):
            letter = chr(97 + idx - 1)
        elif 0 <= idx < len(options):
            letter = chr(97 + idx)

    if letter and not txt and options:
        i = ord(letter) - 97
        if 0 <= i < len(options):
            txt = strip_option_label(options[i])
    return letter, txt


def strip_option_label(opt: str) -> str:
    """Drop a leading "(a) " / "a. " / "A) " label from an option string."""
    return re.sub(r"^\(?([a-dA-D])[\.\)]\s+", "", opt).strip()


def strip_instruction_tail(body: str, min_keep: int = 3,
                           seam_min_keep: int = 15) -> str:
    """Cut the question's instruction off the end of the last list item.

    Two seams are tried and the earliest safe one wins: an explicit tail
    phrase ("which of the above", "is required in respect of"), and an
    unpunctuated run-on into a capitalised instruction clause — the latter
    only when the discarded part actually reads like an instruction, so a
    statement that merely contains the word "For" is never truncated.

    The explicit phrase needs almost no guard (min_keep): real list items
    can be a single word — "Repeal", "Maithili" — followed immediately by
    "Select the correct answer using the code given below". The riskier
    run-on seam keeps the larger seam_min_keep guard.
    """
    body = (body or "").strip()
    cuts = []
    m = _TAIL_RE.search(body)
    if m and m.start() >= min_keep:
        cuts.append(m.start())
    for s in _INSTR_SEAM_RE.finditer(body):
        rest = body[s.start():]
        if s.start() >= seam_min_keep and (_TAIL_RE.search(rest) or "?" in rest):
            cuts.append(s.start())
            break
    if not cuts:
        return body
    return body[:min(cuts)].strip(" ;,.")


def split_statements(stem: str) -> tuple:
    """(lead_stem, [statements]) — split "1. ... 2. ..." out of a stem.

    Requires at least two consecutive markers so a stray "1947." or a single
    "(1)" citation never triggers a split.
    """
    text = " ".join((stem or "").split())
    marks = list(_NUM_MARK_RE.finditer(text))
    if len(marks) < 2:
        return text, []
    # keep only a run of consecutive markers (1,2,3… or I,II,III…)
    seq = []
    for m in marks:
        tok = m.group(1)
        val = _ROMAN.get(tok.upper()) if not tok.isdigit() else int(tok)
        if val is None:
            continue
        want = 1 if not seq else seq[-1][0] + 1
        if val == want:
            seq.append((val, m))
        elif val == 1:
            seq = [(1, m)]
    if len(seq) < 2:
        return text, []
    lead = text[:seq[0][1].start()].strip(" ;:,")
    statements = []
    for i, (_v, m) in enumerate(seq):
        end = seq[i + 1][1].start() if i + 1 < len(seq) else len(text)
        body = text[m.end():end].strip()
        if i == len(seq) - 1:                # strip the instruction tail
            body = strip_instruction_tail(body)
        body = body.strip(" ;,")
        if body:
            statements.append(body)
    if len(statements) < 2:
        return text, []
    return (lead or text), statements


def split_pairs(stem: str, statements: list) -> list:
    """Match-the-following pairs → ["left : right", ...] (best effort)."""
    src = statements or []
    if not src:
        _lead, src = split_statements(stem)
    pairs = []
    for s in src:
        parts = [p for p in _PAIR_SPLIT_RE.split(s) if p.strip()]
        if len(parts) >= 2:
            pairs.append(f"{parts[0].strip()} : {' '.join(parts[1:]).strip()}")
        elif s.strip():
            pairs.append(s.strip())
    return pairs


_AR_RES = (
    # classic: Assertion (A): ... Reason (R): ...
    re.compile(r"assertion\s*\(?A\)?\s*[:\-–]\s*(.+?)\s*"
               r"reason\s*\(?R\)?\s*[:\-–]\s*(.+)$", re.I | re.S),
    # modern UPSC (2023+): Statement-I : ... Statement-II : ...
    re.compile(r"statement\s*[-–]?\s*(?:I|1)\s*[:\-–]\s*(.+?)\s*"
               r"statement\s*[-–]?\s*(?:II|2)\s*[:\-–]\s*(.+)$", re.I | re.S),
)


def split_assertion_reason(stem: str, statements: list) -> tuple:
    """(assertion, reason) from either explicit statements or stem text.

    Handles both the classic "Assertion (A): … Reason (R): …" wording and
    the modern "Statement-I : … Statement-II : …" wording UPSC has used
    since 2023 — without the latter, the whole stem was being emitted as
    the assertion and the two claims were never queried separately.
    """
    if statements and len(statements) >= 2:
        return statements[0], statements[1]
    text = " ".join((stem or "").split())
    for rx in _AR_RES:
        m = rx.search(text)
        if m:
            a = strip_instruction_tail(m.group(1).strip(" .;:"))
            r = strip_instruction_tail(
                _TAIL_RE.split(m.group(2))[0].strip(" .;:"))
            return a, r
    lead, sts = split_statements(text)
    if len(sts) >= 2:
        return sts[0], sts[1]
    return text, ""


# List-I entries are lettered ("A. Article 16(2) B. Article 29(2) …") while
# List-II entries are numbered. They are separate vocabularies and both are
# worth querying — "Article 16(2)" is among the sharpest queries available —
# so they become their own units rather than being guessed into pairs (the
# true pairing IS the answer, which this tool must never assume).
_LETTER_MARK_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s;:.\)]))\(?([A-H])[\.\)]\s+")


_LIST_HDR_RE = re.compile(r"\bList\s*[-–—]?\s*(I{1,3}|IV|[12])\b", re.I)
_LIST_NUM = {"I": 1, "II": 2, "III": 3, "IV": 4, "1": 1, "2": 2}


def _cut_at_list_header(text: str) -> str:
    """Drop anything from the next List header onward."""
    m = _LIST_HDR_RE.search(text or "")
    return (text[:m.start()] if m else text).strip(" ;,:.")


def split_match_lists(stem: str) -> tuple:
    """(list_I_items, list_II_items) for a match-the-following stem.

    Papers use both layouts — List I lettered with List II numbered, and
    the reverse — so the lists are located by their HEADERS and each side is
    then read with whichever marker style it actually uses. Parsing by
    marker style alone mislabels the inverted layout and lets the List II
    block bleed into the last List I item.
    """
    text = " ".join((stem or "").split())
    heads = [(m.start(), m.end(), _LIST_NUM.get(m.group(1).upper(), 0))
             for m in _LIST_HDR_RE.finditer(text)]

    def items(seg: str) -> list:
        out = split_statements(seg)[1] or split_lettered(seg)
        return [i for i in (_cut_at_list_header(x) for x in out) if i]

    # A stem usually names both lists twice — once in the instruction
    # ("Match List I with List II and select…") and again above the entries.
    # Only the second pair delimits anything, so every candidate pair is
    # tried and the one that actually yields entries on both sides wins.
    best = ([], [])
    for i, h in enumerate(heads):
        if h[2] != 1:
            continue
        nxt = next((g for g in heads if g[2] == 2 and g[0] > h[1]), None)
        if not nxt:
            continue
        end = next((g[0] for g in heads if g[0] > nxt[1] and g[2] == 1),
                   len(text))
        left, right = items(text[h[1]:nxt[0]]), items(text[nxt[1]:end])
        if len(left) >= 2 and len(right) >= 2 \
                and len(left) + len(right) > len(best[0]) + len(best[1]):
            best = (left, right)
    return best


def split_lettered(text: str) -> list:
    """List-I style lettered entries, or [] when there is no A,B,C… run."""
    text = " ".join((text or "").split())
    seq = []
    for m in _LETTER_MARK_RE.finditer(text):
        val = ord(m.group(1)) - 64                      # A -> 1
        want = 1 if not seq else seq[-1][0] + 1
        if val == want:
            seq.append((val, m))
        elif val == 1:
            seq = [(1, m)]
    if len(seq) < 2:
        return []
    out = []
    for i, (_v, m) in enumerate(seq):
        end = seq[i + 1][1].start() if i + 1 < len(seq) else len(text)
        body = text[m.end():end].strip(" ;,")
        if i == len(seq) - 1:
            body = strip_instruction_tail(body)
        # a List-II header ("List II (Provisions) 1. …") ends List-I
        body = re.split(r"\bList\s*[-–]?\s*(?:II|2)\b", body, maxsplit=1,
                        flags=re.IGNORECASE)[0].strip(" ;,:")
        if body:
            out.append(body)
    return out


# ── option classification: content vs key code ───────────────────────────

# Structures that ALWAYS refer to the question's own statements/options
# rather than to anything in the world. Note what is deliberately NOT here:
# a lone number or ordinal. "16", "First", "91st", "Article 350-A" and
# "42nd Amendment" are real answers naming real things, and they are the
# sharpest BM25 queries in the whole corpus — treating them as codes threw
# away the best retrieval signal available.
_KEY_CODE_RES = (
    re.compile(r"^\(?[a-d]\)?$", re.I),
    re.compile(r"^(only\s+)?(one|two|three|four|all|none|both|neither)"
               r"(\s+(of\s+)?(them|the\s+above|these|four|three|two|"
               r"statements?|pairs?|the\s+statements?))?\.?$", re.I),
    re.compile(r"^\d+(\s*(,|and|&|nor)\s*\d+)+(\s+only)?\.?$", re.I),
    re.compile(r"^[IVX]+(\s*(,|and|&|nor)\s*[IVX]+)+(\s+only)?\.?$", re.I),
    re.compile(r"^[\dIVX]+\s+only\.?$", re.I),
    re.compile(r"^(all|none|neither|both|only)\b.{0,60}$", re.I),
    re.compile(r"^[A-D]\s*[-–]\s*\d+(\s*,\s*[A-D]\s*[-–]\s*\d+)+$"),  # A-3, B-2
    re.compile(r"^\d+\s*[-–]\s*[A-D](\s*,\s*\d+\s*[-–]\s*[A-D])+$"),
    # "Statement-I is correct but Statement-II is incorrect", "Both
    # Statement-I and Statement-II are correct and Statement-II is the
    # correct explanation for Statement-I" — the modern UPSC relation
    # options. Long enough to slip past the short-option patterns above.
    re.compile(r"^(both\s+)?(statement|assertion|reason)s?\s*[-–]?\s*[IVX\d]",
               re.I),
    re.compile(r"^(both\s+)?A\s+and\s+R\s+are\b", re.I),
    re.compile(r"^A\s+is\s+(true|false|correct|incorrect)\b", re.I),
    # old-style matching code written as bare positions: "2 4 1 3"
    re.compile(r"^\d(\s+\d){2,}$"),
)

# Formats whose options are answer codes, so a bare "2" means statement 2.
_CODE_FORMATS = frozenset({"multi_statement", "assertion_reason",
                           "match_the_following", "sequence"})

_BARE_NUM_RE = re.compile(r"^\(?([\dIVXivx]{1,4})(st|nd|rd|th)?\)?\.?$")

# Options that name something real: a numbered provision, an ordinal
# schedule/amendment, or a plain quantity.
_ORDINAL_WORDS = (r"first|second|third|fourth|fifth|sixth|seventh|eighth|"
                  r"ninth|tenth|eleventh|twelfth|thirteenth|fourteenth")
_WORLD_REF_RE = re.compile(
    r"\b(article|articles|section|schedule|amendment|part|act|list|clause|"
    r"entry|item)\b\s*[-–]?\s*[\dIVXLC]"
    r"|\b\d{1,3}(st|nd|rd|th)\b"
    r"|^(" + _ORDINAL_WORDS + r")\b"
    r"|\b(" + _ORDINAL_WORDS + r")\s+(schedule|amendment|article|part|list)\b",
    re.IGNORECASE)

# Words that carry no topic content in an answer code. Two groups: the
# vocabulary answer codes are built from, and ordinary function words — a
# real paper writes codes as sentences ("There are two correct statements
# that include statement 3."), so without the function words those score as
# content and get embedded, retrieving pure noise.
_CODE_WORDS = frozenset("""
only and or both neither none all above these those them one two three four
statement statements pair pairs correct incorrect true false is are not
matched match code codes given below option options answer answers
assertion reason explanation of the a an it its as with same different
respectively
there here this that which who whom whose what when how why but if so then
include includes including includes contains contain among amongst
respect regard reference view views conclusion conclusions above-mentioned
aforesaid following follows listed mentioned stated said
are were was be been being has have had does do did can may might must
number numbers count only-one at-least atmost more less than
""".split())

# A statement/pair reference ("statement 3", "pairs given above", "List-I")
# marks an answer code rather than a fact — unless the option also names
# something real (a proper noun, an Article/Section, a year).
_REFERENCE_RE = re.compile(
    r"\b(statements?|pairs?|assertions?|reasons?|options?|lists?|"
    r"list\s*[-–]?\s*[I]{1,3}|codes?)\b", re.IGNORECASE)
_NAMED_RE = re.compile(
    r"\b(?:Article|Articles|Section|Schedule|Amendment|Act|Part)\s+[\dIVXLC]+"
    r"|\b(?:19|20)\d{2}\b"
    r"|\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)+\b")     # multi-word proper noun


def _names_something(o: str) -> bool:
    """True when the option names a real thing. A proper-noun match whose
    first word is code vocabulary ("Both Statement", "All Statements") does
    not count — that phrasing was letting relation options pose as content.
    """
    for m in _NAMED_RE.finditer(o):
        first = re.split(r"[\s\-]", m.group(0), 1)[0].lower()
        if first not in _CODE_WORDS:
            return True
    return False


def classify_option(opt: str, fmt: str = "") -> tuple:
    """("content"|"key_code", reason).

    A key code points at the question's own statements or states a relation
    between them ("1 and 2 only", "Both A and R are true"); embedding it
    retrieves noise. Everything that names something in the world is
    content — including bare numbers and ordinals, which in a select/identify
    item are the answer ("How many spokes?" → "24"; "Which Schedule?" →
    "First"; "Which Article?" → "Article 350-A"). The question's format
    disambiguates the one genuinely ambiguous case: a lone numeral means
    statement 2 in a multi-statement item and the number 2 anywhere else.
    """
    o = strip_option_label(" ".join((opt or "").split()))
    if not o:
        return "key_code", "empty"
    for rx in _KEY_CODE_RES:
        if rx.match(o):
            return "key_code", "answer-code pattern"
    bare = _BARE_NUM_RE.match(o)
    if bare:
        if fmt in _CODE_FORMATS:
            return "key_code", "bare statement reference"
        return "content", "numbered reference"
    if _WORLD_REF_RE.search(o):
        return "content", "numbered reference"
    if _REFERENCE_RE.search(o) and not _names_something(o):
        return "key_code", "refers to statements, names nothing"
    # ONE real word is enough. "Germany", "Calcutta", "Hidayatullah" and
    # "1971 Census" are complete answers and the most precise BM25 queries
    # in the corpus; requiring two content words discarded them.
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-']+", o)
             if len(w) > 2 and w.lower() not in _CODE_WORDS]
    if words:
        return "content", ""
    # No words at all, but a real value — "Rs 10,000 and Rs 2,500". Every
    # purely code-shaped numeric form was already caught above.
    if re.search(r"\d", o):
        return "content", "numeric value"
    return "key_code", "no content words"


# ── the normalised question ──────────────────────────────────────────────

def normalise_question(raw: dict, idx: int) -> dict:
    """Any input shape → one common structure. Never raises: an unparseable
    row still produces a stem unit so nothing is silently dropped."""
    stem_raw = str(_first(raw, _STEM_KEYS) or "").strip()
    options = _as_option_list(_first(raw, _OPT_KEYS))
    statements = _as_text_list(_first(raw, _ST_KEYS))
    statements = [strip_instruction_tail(s) if i == len(statements) - 1 else s
                  for i, s in enumerate(statements)]
    statements = [s for s in statements if s]

    fmt = str(_first(raw, _FORMAT_KEYS) or "").strip().lower()
    fmt = _FORMAT_ALIASES.get(fmt, _FORMAT_ALIASES.get(
        fmt.replace("-", " ").replace("_", " "), fmt or ""))

    lead, embedded = split_statements(stem_raw)
    if not statements and embedded:
        statements = embedded
    stem_lead = lead if statements else " ".join(stem_raw.split())

    if not fmt:                                    # infer from structure
        low = stem_raw.lower()
        if "assertion" in low and "reason" in low:
            fmt = "assertion_reason"
        elif "match" in low or "list-i" in low or "list i" in low:
            fmt = "match_the_following"
        elif statements:
            fmt = "multi_statement"
        else:
            fmt = "select_identify"

    qid = str(_first(raw, ("id", "qid", "question_id", "uid")) or f"q{idx + 1:03d}")
    ans, ans_text = resolve_answer(raw, options)

    return {
        "id": qid,
        "format": fmt,
        "stem": stem_lead,
        "stem_full": " ".join(stem_raw.split()),
        "statements": statements,
        "options": options,
        "answer": ans,
        "answer_text": ans_text,
        "year": raw.get("year"),
        "subject": raw.get("subject"),
        "topic": raw.get("topic"),
        "subtopic": raw.get("subtopic"),
        "explanation": raw.get("explanation") or "",
    }


_ORDINAL_ONLY_RE = re.compile(
    r"^\(?[\dIVXivx]{1,4}(st|nd|rd|th)?\)?\.?$|^(" + _ORDINAL_WORDS + r")$",
    re.IGNORECASE)


def _context_poor(text: str) -> bool:
    """True when a unit's text cannot stand alone as a query.

    "First", "Second", "24", "3rd" name the answer only in the light of the
    stem: on their own they retrieve generic chunks (measured: ~0.76
    similarity against unrelated foundation units, nothing about Schedules).
    A single proper noun like "Calcutta" or "Germany" IS a usable query and
    is deliberately not caught here.
    """
    t = " ".join((text or "").split())
    if not t:
        return True
    if _ORDINAL_ONLY_RE.match(t):
        return True
    return not content_words(t)


def build_units(q: dict, query_key_codes: bool = False,
                stem_context: bool = True) -> list:
    """The retrieval units for one question, in a stable order.

    `query` is always the unit's own text. When that text cannot stand alone
    (see _context_poor), `query_sent` carries the stem-prefixed form that is
    actually searched, so the file still shows both what the unit says and
    what was asked of the index.
    """
    units = []
    stem_for_ctx = " ".join(str(q.get("stem_full") or "").split())

    def add(name, text, queried=True, skip=""):
        text = " ".join(str(text or "").split())
        u = {"unit": name, "query": text,
             "queried": bool(queried and text),
             "skip_reason": skip if not (queried and text) else ""}
        if (u["queried"] and stem_context and name != "stem"
                and _context_poor(text) and stem_for_ctx):
            u["query_sent"] = f"{stem_for_ctx} {text}"
            u["context_added"] = True
        units.append(u)

    add("stem", q["stem_full"])
    fmt = q["format"]

    if fmt == "assertion_reason":
        a, r = split_assertion_reason(q["stem_full"], q["statements"])
        add("assertion", a)
        if r:
            add("reason", r)
    elif fmt == "match_the_following":
        left, right = split_match_lists(q["stem_full"])
        if not (left and right):            # no usable headers — infer
            left = split_lettered(q["stem_full"])
            right = [_cut_at_list_header(s) for s in
                     (q["statements"] or split_statements(q["stem_full"])[1])]
            right = [r for r in right if r]
        if left and right:
            for i, item in enumerate(left):
                add(f"list1_{chr(97 + i)}", item)
            for i, item in enumerate(right, 1):
                add(f"list2_{i}", item)
        else:
            for i, p in enumerate(split_pairs(q["stem_full"],
                                              q["statements"]), 1):
                add(f"pair_{i}", p)
    else:
        for i, s in enumerate(q["statements"], 1):
            add(f"statement_{i}", s)

    for i, opt in enumerate(q["options"]):
        label = chr(97 + i) if i < 26 else str(i + 1)
        kind, why = classify_option(opt, fmt)
        if kind == "content" or query_key_codes:
            add(f"option_{label}", strip_option_label(opt))
            if kind == "key_code":
                units[-1]["option_kind"] = "key_code"
        else:
            add(f"option_{label}", strip_option_label(opt),
                queried=False, skip=f"key_code: {why}")
            units[-1]["option_kind"] = "key_code"
    return units


# ══════════════════════════════════════════════════════════════════════════
# retrieval
# ══════════════════════════════════════════════════════════════════════════

_JUNK_PATTERNS = re.compile(                       # mirrors questions.py
    r"promotional|advertisement|publisher ad|banner|scan (the )?qr|"
    r"omitted (promotional )?image|digital.?edition|coursew(are|ise) on",
    re.IGNORECASE,
)


def make_reranker(model_name: str, mode: str):
    """(score_fn, label). score_fn(query, [texts]) -> [float], higher better.

    A cross-encoder reads the query and the chunk TOGETHER, so it can tell
    "this passage answers that question" from "this passage is about the same
    topic" — which bi-encoder retrieval cannot. That distinction is the whole
    point when the output has to be 1–3 chunks rather than 40.

    Interface matches the backend's own reranker contract
    (get_reranking_function → reranking_function.predict([(query, text), …])),
    so a model set in RAG_RERANKING_MODEL behaves here exactly as in chat.
    """
    if mode == "lexical":
        def lex(query: str, texts: list) -> list:
            qw = content_words(query)
            qe = {e.lower() for e in entities(query)}
            out = []
            for t in texts:
                tw = content_words(t)
                inter = len(qw & tw)
                # F1 over content words, plus credit for named referents that
                # actually appear — an Article number matching is worth more
                # than three generic words overlapping.
                prec = inter / max(1, len(tw))
                rec = inter / max(1, len(qw))
                f1 = 0.0 if not inter else 2 * prec * rec / (prec + rec)
                ent = sum(1 for e in qe if e in t.lower())
                out.append(f1 + 0.15 * ent)
            return out
        return lex, "lexical(no model)"

    from sentence_transformers import CrossEncoder
    print(f"  loading reranker {model_name} "
          f"(first run downloads the model)", flush=True)
    try:
        ce = CrossEncoder(model_name, local_files_only=True)
    except Exception:
        ce = CrossEncoder(model_name)

    cache: dict = {}

    def score(query: str, texts: list) -> list:
        todo, out = [], [None] * len(texts)
        for i, t in enumerate(texts):
            k = (hash(query), hash(t))
            if k in cache:
                out[i] = cache[k]
            else:
                todo.append((i, k, t))
        if todo:
            preds = ce.predict([(query, t) for _i, _k, t in todo],
                               show_progress_bar=False)
            for (i, k, _t), p in zip(todo, preds):
                cache[k] = float(p)
                out[i] = float(p)
        return out

    return score, f"cross-encoder:{model_name}"


def question_rerank_query(q: dict, units: list) -> str:
    """One string describing what the question asks, for question-level
    reranking: the stem plus every claim-bearing unit and contentful option.
    Key codes are left out — "1 and 2 only" describes nothing."""
    parts = [q.get("stem_full") or ""]
    for u in units:
        if not u.get("queried"):
            continue
        if u["unit"] == "stem":
            continue
        parts.append(u["query"])
    return " ".join(p for p in parts if p)[:2000]


def rerank_question(q: dict, chunks: dict, score_fn, per_unit: int,
                    top_k: int) -> dict:
    """Rerank each unit's hits, trim them, then pick the question's best
    chunks overall. Returns counters for reporting."""
    stats = {"unit_pairs": 0, "question_pairs": 0, "dropped": 0}

    # ── per unit: the best evidence for THAT part of the question ─────────
    for u in q["_units"]:
        hits = u.get("hits") or []
        if not hits:
            continue
        texts = [chunks[h["chunk"]]["text"] for h in hits]
        scores = score_fn(u.get("query_sent") or u["query"], texts)
        stats["unit_pairs"] += len(texts)
        for h, s in zip(hits, scores):
            h["rerank"] = round(float(s), 5)
        hits.sort(key=lambda h: -h["rerank"])
        for i, h in enumerate(hits, 1):
            h["rerank_rank"] = i
        if per_unit and len(hits) > per_unit:
            stats["dropped"] += len(hits) - per_unit
            u["hits"] = hits[:per_unit]
        else:
            u["hits"] = hits

    # ── question level: the 1–3 chunks that actually answer it ────────────
    cands = []
    seen = set()
    for u in q["_units"]:
        for h in u.get("hits") or []:
            if h["chunk"] not in seen:
                seen.add(h["chunk"])
                cands.append(h["chunk"])
    if cands:
        qquery = question_rerank_query(q, q["_units"])
        scores = score_fn(qquery, [chunks[c]["text"] for c in cands])
        stats["question_pairs"] += len(cands)
        found_by = {}
        for u in q["_units"]:
            for h in u.get("hits") or []:
                found_by.setdefault(h["chunk"], []).append(u["unit"])
        ranked = sorted(zip(cands, scores), key=lambda kv: -kv[1])
        q["top_chunks"] = [
            {"chunk": c, "rerank": round(float(s), 5), "rank": i,
             "found_by": sorted(set(found_by.get(c, [])))}
            for i, (c, s) in enumerate(ranked[:top_k], 1)]
    else:
        q["top_chunks"] = []
    return stats


def apply_bm25_gate(hits: list, frac: float) -> tuple:
    """Drop BM25 hits scoring below `frac` of the best BM25 score for this
    query. Returns (kept, dropped_count).

    BM25 idf is computed WITHIN a collection, so a small collection inflates
    every score: a 30-object economics collection out-scored nothing, yet its
    own top-8 came back for every query and put the same five economics
    chunks into 44 of 50 polity questions. The gate is relative, so a
    collection that is genuinely the best match for a query is compared
    against itself and keeps everything. Measured on a real 50-question run,
    frac=0.2 removed 89% of that noise while keeping every one of 11
    independently verified answer-bearing chunks.
    """
    if frac <= 0:
        return hits, 0
    best = max((h.get("score") or 0) for h in hits
               if h.get("channel") == "bm25") if any(
        h.get("channel") == "bm25" for h in hits) else 0
    if best <= 0:
        return hits, 0
    kept = [h for h in hits if h.get("channel") != "bm25"
            or (h.get("score") or 0) >= frac * best]
    return kept, len(hits) - len(kept)


def is_junk_chunk(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 180:
        return True
    if _JUNK_PATTERNS.search(t) and len(t) < 600:
        return True
    return False


def _dual_channel_retrieve(client, collection_names, query_embedding,
                           bm25_query, k, filter=None,
                           max_hits: int = 0) -> list:
    """Copy of routers/questions.py::_dual_channel_retrieve_sync, adapted for
    searching MORE THAN ONE collection, plus channel/score/rank labels.

    The router's final `return hits[: k * 2]` is correct for its own use —
    it always passes a single collection, so the cap means "k*2 from this
    collection". Reused across many collections that same line becomes a
    silent corpus lottery: collections are visited in list order, the first
    one fills all k*2 slots, and every later collection is truncated away.
    That is exactly what happened here — a 2203-object foundation corpus
    sorted alphabetically first supplied 976 of 1090 chunks while the
    1427-object collection holding the actual chapters contributed ZERO, with
    no error to show for it. The cap is therefore applied PER COLLECTION,
    which preserves the router's semantics per call, and any overall bound is
    an explicit choice (max_hits).

    `rank` is the hit's position within its own collection+channel result —
    a real relevance rank. Order across collections is list order, not
    relevance, so compare hits by `score` (and only within one channel).
    """
    hits, seen_texts = [], set()

    def _scores(res, n):
        """Whatever relevance value the client exposes, with its direction
        READ OFF THE DATA rather than guessed from the attribute name.

        The client returns these under `distances`, but they arrive ordered
        best-first, so a series that falls as rank worsens is a similarity
        and one that rises is a distance. Labelling by attribute name alone
        called cosine similarities "vector_distance", which inverts the
        meaning of every threshold built on top of it.
        """
        for attr in ("distances", "scores"):
            v = getattr(res, attr, None)
            if v and v[0] and len(v[0]) >= n:
                vals = list(v[0])
                clean = [x for x in vals if isinstance(x, (int, float))]
                kind = "score"
                if len(clean) >= 2:
                    kind = ("distance" if clean[-1] > clean[0]
                            else "similarity")
                return vals, kind
        return [None] * n, ""

    for cname in collection_names:
        local: list = []
        try:                                              # dense channel
            res = client.search(collection_name=cname,
                                vectors=[query_embedding],
                                filter=filter, limit=k)
            if res and res.documents and res.documents[0]:
                metas = res.metadatas[0] if res.metadatas \
                    else [{}] * len(res.documents[0])
                sc, kind = _scores(res, len(res.documents[0]))
                for i, (doc, meta) in enumerate(zip(res.documents[0], metas)):
                    text = ((meta or {}).get("raw_text") or doc or "").strip()
                    key = text[:300]
                    if text and key not in seen_texts:
                        seen_texts.add(key)
                        local.append({"text": text, "metadata": meta or {},
                                      "source": cname, "channel": "vector",
                                      "rank": i + 1,
                                      "score": sc[i] if i < len(sc) else None,
                                      "score_kind": f"vector_{kind}"})
        except Exception as e:
            print(f"    ! vector search failed on {cname}: {e}", flush=True)
        try:                                              # sparse channel
            res = client.bm25_search(collection_name=cname, query=bm25_query,
                                     query_properties=["raw_text", "text"],
                                     limit=k, filter=filter)
            if res and res.documents and res.documents[0]:
                metas = res.metadatas[0] if res.metadatas \
                    else [{}] * len(res.documents[0])
                sc, kind = _scores(res, len(res.documents[0]))
                for i, (doc, meta) in enumerate(zip(res.documents[0], metas)):
                    key = (doc or "")[:300]
                    if doc and key not in seen_texts:
                        seen_texts.add(key)
                        local.append({"text": doc, "metadata": meta or {},
                                      "source": cname, "channel": "bm25",
                                      "rank": i + 1,
                                      "score": sc[i] if i < len(sc) else None,
                                      "score_kind": "bm25_score"})
        except Exception as e:
            print(f"    ! bm25 search failed on {cname}: {e}", flush=True)
        hits.extend(local[: k * 2])        # the router's cap, PER collection
    return hits[:max_hits] if max_hits else hits


def make_embedder(env: dict):
    """Query embedder matching the backend's own settings (engine, model,
    query prefix, L2 normalisation)."""
    engine = (os.environ.get("RAG_EMBEDDING_ENGINE") or "").strip().lower()
    model = os.environ.get("RAG_EMBEDDING_MODEL") \
        or "sentence-transformers/all-MiniLM-L6-v2"
    prefix = os.environ.get("RAG_EMBEDDING_QUERY_PREFIX") or ""

    if engine in ("", "sentence-transformers", "st", "local"):
        from sentence_transformers import SentenceTransformer
        try:
            st = SentenceTransformer(model, local_files_only=True)
        except Exception:
            st = SentenceTransformer(model)

        def embed(text: str):
            return st.encode(f"{prefix}{text}" if prefix else text,
                             normalize_embeddings=True,
                             show_progress_bar=False,
                             convert_to_numpy=True).tolist()
        return embed, f"sentence-transformers:{model}"

    if engine == "ollama":
        import urllib.request
        url = (os.environ.get("RAG_OLLAMA_BASE_URL")
               or os.environ.get("OLLAMA_BASE_URL")
               or "http://localhost:11434").rstrip("/")

        def embed(text: str):
            body = json.dumps({
                "model": model,
                "input": f"{prefix}{text}" if prefix else text}).encode()
            req = urllib.request.Request(
                f"{url}/api/embed", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode())
            emb = (data.get("embeddings") or [data.get("embedding")])[0]
            return list(emb)
        return embed, f"ollama:{model}"

    raise SystemExit(
        f"RAG_EMBEDDING_ENGINE={engine!r} is not supported by this script "
        "(use sentence-transformers or ollama, or pass --collections and "
        "--dry-run to inspect queries only)")


def _weaviate_base() -> str:
    host = os.environ.get("WEAVIATE_HTTP_HOST", "localhost")
    port = os.environ.get("WEAVIATE_HTTP_PORT", "8080")
    secure = (os.environ.get("WEAVIATE_HTTP_SECURE", "false").lower()
              in ("1", "true", "yes"))
    return f"{'https' if secure else 'http'}://{host}:{port}"


def _weaviate_get(path: str, body: dict = None) -> dict:
    import urllib.request
    req = urllib.request.Request(
        _weaviate_base() + path,
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json"} if body else {})
    key = os.environ.get("WEAVIATE_API_KEY", "")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def _weaviate_schema(env: dict) -> dict:
    return _weaviate_get("/v1/schema")


def collection_counts(env: dict) -> dict:
    """collection -> object count, straight from Aggregate meta.count."""
    names = [c["class"] for c in (_weaviate_schema(env).get("classes") or [])
             if c.get("class")]
    if not names:
        return {}
    try:
        q = "{Aggregate{" + " ".join(f"{n}{{meta{{count}}}}" for n in names) \
            + "}}"
        agg = (_weaviate_get("/v1/graphql", {"query": q}).get("data")
               or {}).get("Aggregate") or {}
        return {n: ((agg.get(n) or [{}])[0].get("meta", {}).get("count") or 0)
                for n in names}
    except Exception as e:
        log.warning("collection count query failed: %s", e)
        return {n: None for n in names}


def discover_collections(env: dict, pyq_collection="PyqKnowledge",
                         masters_only: bool = False) -> tuple:
    """(searched, skipped, excluded) collection names.

    Masters (knowledge-base collections) hold every member file's chunks, so
    the per-file collections normally duplicate them and are skipped — here
    six File_* collections all held the same economics chapter a master
    already had. The exception is a master with NO objects, whose content
    exists only per-file; that case is detected from real object counts
    rather than assumed, because a keyword probe cannot tell an empty
    collection from a stopword query. Empty collections are skipped outright,
    and the PYQ collection is always excluded — the point is to find the
    source material for a question, not other questions.
    """
    counts = collection_counts(env)
    names = sorted(counts)
    per_file = [n for n in names if re.match(r"^C?file[_-]", n, re.I)]
    excluded = [n for n in names if n == pyq_collection]
    empty = [n for n in names if counts.get(n) == 0]
    masters = [n for n in names
               if n not in per_file and n not in excluded and n not in empty]
    keep_per_file = [] if (masters or masters_only) else [
        n for n in per_file if n not in empty]
    searched = masters + keep_per_file
    skipped = [n for n in names
               if n not in searched and n not in excluded]
    return searched, skipped, excluded


def list_collections(env: dict, client, sample: int = 200) -> int:
    """What is actually indexed, per collection: exact object count and the
    source files inside it.

    Uses Aggregate meta.count, NOT a keyword probe — Weaviate's English BM25
    analyzer drops stopwords, so probing with a word like "the" returns zero
    hits from a collection holding thousands of chunks and looks exactly like
    an empty index.
    """
    schema = _weaviate_schema(env)
    classes = {c["class"]: [p.get("name") for p in (c.get("properties") or [])]
               for c in (schema.get("classes") or []) if c.get("class")}
    names = sorted(classes)
    counts = {}
    try:
        q = "{Aggregate{" + " ".join(f"{n}{{meta{{count}}}}" for n in names) \
            + "}}"
        data = _weaviate_get("/v1/graphql", {"query": q})
        agg = (data.get("data") or {}).get("Aggregate") or {}
        for n in names:
            rows = agg.get(n) or []
            counts[n] = (rows[0].get("meta", {}).get("count")
                         if rows else 0)
    except Exception as e:
        print(f"  (count query failed: {type(e).__name__}: {e})")

    print(f"\n{len(names)} collection(s) in Weaviate\n")
    for n in names:
        kind = "per-file" if re.match(r"^C?file[_-]", n, re.I) else "master"
        cnt = counts.get(n)
        print(f"  {n[:52]:52s} {('?' if cnt is None else cnt):>7} objects "
              f"[{kind}]")
        if not cnt:
            continue
        field = next((f for f in ("name", "breadcrumb", "chunk_id")
                      if f in (classes.get(n) or [])), None)
        if not field:
            continue
        try:
            data = _weaviate_get("/v1/graphql", {
                "query": "{Get{%s(limit:%d){%s}}}" % (n, sample, field)})
            rows = ((data.get("data") or {}).get("Get") or {}).get(n) or []
            srcs = Counter(str((r or {}).get(field) or "?") for r in rows)
            shown = ", ".join(f"{s[:44]} ({c})" for s, c in
                              srcs.most_common(6))
            more = f" … {len(srcs) - 6} more" if len(srcs) > 6 else ""
            print(f"      {field}s in first {len(rows)}: {shown}{more}")
        except Exception as e:
            print(f"      (sample failed: {type(e).__name__})")
    print("\nA master with 0 objects keeps its content only in its per-file "
          "collections — those are searched by default (--masters-only skips "
          "them).")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# chunk store + coverage QA
# ══════════════════════════════════════════════════════════════════════════

def chunk_key(hit: dict) -> str:
    meta = hit.get("metadata") or {}
    for f in ("chunk_id", "id", "uuid"):
        v = meta.get(f)
        if v:
            return str(v)
    return "sha1:" + hashlib.sha1(
        (hit.get("text") or "").encode("utf-8")).hexdigest()[:16]


def chunk_record(hit: dict) -> dict:
    meta = hit.get("metadata") or {}
    keep = {}
    for f in ("breadcrumb", "name", "header_1", "header_2", "chapter",
              "section_id", "chunk_id", "parent_id", "subject",
              "class_level", "keywords", "kw_domain", "kw_block_type",
              "token_count", "content_type"):
        if meta.get(f) not in (None, ""):
            keep[f] = meta[f]
    return {
        "text": hit.get("text") or "",
        "chars": len(hit.get("text") or ""),
        "collection": hit.get("source") or "",
        "metadata": keep,
    }


_STOP = frozenset("""
the a an of and or in on at to for with is are was were be been being that
this these those which who whom whose what when where how why not no nor
it its as by from into over under between during above below about only
following consider statements statement given correct incorrect select
answer code codes above them all none both neither one two three four
""".split())


def content_words(text: str) -> set:
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-']{2,}", text or "")
            if w.lower() not in _STOP}


_LEADING_DET = frozenset("the a an this that these those his her its their "
                         "our some any".split())


def entities(text: str) -> set:
    """Capitalised runs and numbered references — the checkable referents.

    A capitalised run at the start of a sentence swallows the determiner:
    "The Dinesh Goswami Committee was concerned with…" yielded exactly that
    string, which no chapter contains verbatim, so a question whose evidence
    was retrieved at rank 1 was flagged for thin entity support. Leading
    determiners are stripped so the referent is the thing, not the sentence.
    """
    out = set()
    for m in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b",
                         text or ""):
        run = m.group(0)
        toks = run.split()
        while toks and toks[0].lower() in _LEADING_DET:
            toks = toks[1:]
        run = " ".join(toks)
        if run and run.lower() not in _STOP:
            out.add(run)
    out |= set(re.findall(
        r"\b(?:Article|Articles|Section|Schedule|Amendment|Act)\s+[\dIVXLC]+[A-Z]?\b",
        text or "", re.I))
    return out


_CONTENT_UNIT_RE = re.compile(
    r"^(statement_\d+|assertion|reason|pair_\d+|list1_[a-h]|list2_\d+)$")


def coverage(q: dict, units: list, union_text: str) -> dict:
    """Does the retrieved evidence actually contain what decides the answer?

    This is the QA signal the file exists for. Which text to score against
    depends on the format, and getting that wrong produces false alarms:
      • answer_support is computed ONLY when the marked option is real
        content (select/identify). For a multi-statement or assertion-reason
        item the marked option is a key code ("2 and 3 only", "Both A and R
        are true…") whose words appear in no chapter, so scoring it would
        flag every such question — it is reported as null instead.
      • claim_support covers the units that DO carry the facts: the numbered
        statements, the assertion and reason, the match pairs.
      • stem_entity_support checks the stem's named/numbered referents
        (Article 280, Finance Commission) — the terms retrieval had to hit.
    A question is flagged when whichever signals apply come back thin.
    """
    ut = (union_text or "").lower()

    def frac(items):
        items = [i for i in items if i]
        if not items:
            return None
        return round(sum(1 for i in items if str(i).lower() in ut)
                     / len(items), 3)

    # "absent" is not the same as "key_code": a file with no answer key at
    # all must not look like a set of questions whose answers happen to be
    # unscoreable codes.
    ans_text = q.get("answer_text") or ""
    ans_kind = (classify_option(ans_text, q.get("format", ""))[0]
                if ans_text else "absent")
    ans = frac(content_words(ans_text)) if ans_kind == "content" else None

    claim_w = set()
    for u in units or []:
        if _CONTENT_UNIT_RE.match(u.get("unit", "")):
            claim_w |= content_words(u.get("query") or "")
    for s in q.get("statements") or []:          # explicit statements too
        claim_w |= content_words(s)
    claims = frac(claim_w)

    ent = frac(entities(q.get("stem_full") or ""))

    flags = []
    if not (union_text or "").strip():
        flags.append("no_chunks_retrieved")
    else:
        if ans is not None and ans < 0.5:
            flags.append("answer_text_thin")
        if claims is not None and claims < 0.5:
            flags.append("claims_thin")
        if ent is not None and ent < 0.4:
            flags.append("stem_entities_thin")
        # Nothing scoreable at all (no contentful answer, no statements, no
        # entities) — say so rather than reporting a silent pass.
        if ans is None and claims is None and ent is None:
            flags.append("nothing_scoreable")
    return {"answer_kind": ans_kind, "answer_support": ans,
            "claim_support": claims, "stem_entity_support": ent,
            "flags": flags}


# ══════════════════════════════════════════════════════════════════════════
# input loading
# ══════════════════════════════════════════════════════════════════════════

def load_questions(path: Path) -> list:
    text = path.read_text(encoding="utf-8-sig", errors="replace").strip()
    if not text:
        raise SystemExit(f"{path} is empty")
    rows = None
    if text[:1] in "[{":
        try:
            data = json.loads(text)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                for k in ("questions", "data", "items", "rows", "records",
                          "pyqs"):
                    if isinstance(data.get(k), list):
                        rows = data[k]
                        break
                if rows is None:
                    vals = [v for v in data.values() if isinstance(v, dict)]
                    rows = vals or [data]
        except json.JSONDecodeError:
            rows = None
    if rows is None:                                        # JSONL
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip().rstrip(",")
            if not line or line in "[]":
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i} is neither JSON nor JSONL: {e}")
            if isinstance(obj, dict):
                rows.append(obj)
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        raise SystemExit(f"no question objects found in {path}")
    return rows


# ══════════════════════════════════════════════════════════════════════════
# main run
# ══════════════════════════════════════════════════════════════════════════

def run(args) -> int:
    env = load_env(ROOT)
    in_path = Path(args.input)
    if not in_path.is_absolute() and not in_path.exists():
        alt = ROOT / args.input
        if alt.exists():
            in_path = alt
    if not in_path.exists():
        raise SystemExit(f"input file not found: {args.input}")

    raw_rows = load_questions(in_path)
    if args.limit:
        raw_rows = raw_rows[: args.limit]

    # One malformed row must never cost the other 49. Anything that fails to
    # parse is reported by index and recorded in meta.unparsed — never
    # dropped silently.
    questions, unparsed = [], []
    for i, r in enumerate(raw_rows):
        try:
            q = normalise_question(r, i)
            q["_units"] = build_units(
                q, query_key_codes=args.query_key_codes,
                stem_context=not args.no_stem_context)
            if not any(u["queried"] for u in q["_units"]):
                unparsed.append({"row": i, "id": q.get("id"),
                                 "reason": "no queryable text in the row"})
                continue
            questions.append(q)
        except Exception as e:
            unparsed.append({"row": i,
                             "id": str(r.get("id") or r.get("qid") or ""),
                             "reason": f"{type(e).__name__}: {e}"})
    if unparsed:
        print(f"  ! {len(unparsed)} row(s) could not be parsed and were "
              f"skipped:")
        for u in unparsed[:5]:
            print(f"      row {u['row']} {u['id']}: {u['reason']}")
        if len(unparsed) > 5:
            print(f"      … and {len(unparsed) - 5} more (see meta.unparsed)")
    if not questions:
        raise SystemExit(f"no usable questions in {in_path}")

    n_units = sum(1 for q in questions for u in q["_units"] if u["queried"])
    n_skip = sum(1 for q in questions for u in q["_units"] if not u["queried"])
    fmts = {}
    for q in questions:
        fmts[q["format"]] = fmts.get(q["format"], 0) + 1
    print(f"loaded {len(questions)}/{len(raw_rows)} questions from "
          f"{in_path.name}")
    print(f"  formats: {fmts}")
    n_ans = sum(1 for q in questions if q["answer"])
    if n_ans < len(questions):
        print(f"  note: {len(questions) - n_ans} question(s) have no marked "
              f"answer — retrieval still runs, answer_support is reported "
              f"as null for those")
    n_ctx = sum(1 for q in questions for u in q["_units"]
                if u.get("context_added"))
    print(f"  retrieval units: {n_units} to query, {n_skip} skipped "
          f"(key codes){' — FORCED ON' if args.query_key_codes else ''}"
          + (f", {n_ctx} given stem context" if n_ctx else ""))

    if args.dry_run:
        for q in questions[: args.dry_run_show]:
            print(f"\n[{q['id']}] {q['format']}  {q['stem'][:70]}...")
            for u in q["_units"]:
                mark = "?" if u["queried"] else "-"
                extra = "" if u["queried"] else f"   ({u['skip_reason']})"
                print(f"   {mark} {u['unit']:<14} {u['query'][:64]}{extra}")
        print(f"\ndry run — nothing embedded, nothing written "
              f"({n_units} queries would run)")
        return 0

    # ── live retrieval ───────────────────────────────────────────────────
    sys.path.insert(0, str(ROOT / "backend"))
    from futureverse_ai.retrieval.vector.factory import VECTOR_DB_CLIENT

    retrieve = _dual_channel_retrieve
    retriever_name = "mirror(_dual_channel_retrieve_sync)"
    router_fn = None
    if args.router_retriever or args.verify_retriever:
        try:
            from futureverse_ai.routers.questions import (
                _dual_channel_retrieve_sync as router_fn)
        except Exception as e:
            raise SystemExit(
                "could not import the router's retrieval function "
                f"({type(e).__name__}: {e}).\nThis needs the backend's full "
                "environment (FastAPI, DB models). Drop "
                "--router-retriever/--verify-retriever to use the built-in "
                "copy of the same function.")
        if args.router_retriever:
            retriever_name = "router(_dual_channel_retrieve_sync)"

    if args.collections:
        primary = [c.strip() for c in args.collections.split(",") if c.strip()]
        per_file, excluded = [], []
    elif args.knowledge_id:
        primary, per_file, excluded = [args.knowledge_id], [], []
    else:
        primary, per_file, excluded = discover_collections(
            env, masters_only=args.masters_only)
        if not primary and per_file:
            print("  no master collections found — falling back to per-file")
            primary = per_file
    if not primary:
        raise SystemExit("no collections to search — pass --collections or "
                         "--knowledge-id")
    print(f"  collections searched ({len(primary)}): {', '.join(primary)}")
    if per_file:
        print(f"  skipped ({len(per_file)}): {', '.join(per_file[:6])}"
              f"{'…' if len(per_file) > 6 else ''}"
              "   (duplicates of a master, or empty)")
    if excluded:
        print(f"  excluded: {excluded}")

    embed, embed_name = make_embedder(env)
    print(f"  embedder: {embed_name}   k={args.k}   "
          f"junk filter: {'off' if args.no_junk_filter else 'on'}")

    reranker = rerank_label = None
    if not args.no_rerank:
        try:
            reranker, rerank_label = make_reranker(args.rerank_model,
                                                   args.rerank_mode)
        except Exception as e:
            raise SystemExit(
                f"could not load the reranker ({type(e).__name__}: {e}).\n"
                "Options: --rerank-model <another model>, "
                "--rerank-mode lexical (no model needed), or --no-rerank.")
        print(f"  reranker: {rerank_label}   keep {args.rerank_per_unit} "
              f"per unit, top {args.rerank_top_k} per question")

    chunks: dict = {}
    divergence = []
    bm25_gated_total = 0
    channel_hits = Counter()          # (collection, channel) -> hits
    rr_stats = {"unit_pairs": 0, "question_pairs": 0, "dropped": 0}
    t0 = time.time()
    done = 0
    for q in questions:
        for u in q["_units"]:
            if not u["queried"]:
                u["hits"] = []
                continue
            qtext = u.get("query_sent") or u["query"]
            emb = embed(qtext)
            hits = retrieve(VECTOR_DB_CLIENT, primary, emb, qtext,
                            args.k, max_hits=args.max_hits_per_unit) \
                if retrieve is _dual_channel_retrieve \
                else router_fn(collection_names=primary, query_embedding=emb,
                               bm25_query=qtext, k=args.k)
            hits, gated = apply_bm25_gate(hits, args.bm25_min_frac)
            if gated:
                u["bm25_gated"] = gated
                bm25_gated_total += gated
            if args.verify_retriever and router_fn:
                ref = router_fn(collection_names=primary, query_embedding=emb,
                                bm25_query=qtext, k=args.k)
                a = [h["text"][:300] for h in hits]
                b = [h["text"][:300] for h in ref]
                if a != b:
                    divergence.append(f"{q['id']}/{u['unit']}")
            if not args.no_junk_filter:
                kept = [h for h in hits if not is_junk_chunk(h.get("text"))]
                u["junk_filtered"] = len(hits) - len(kept)
                hits = kept
            out_hits = []
            for pos, h in enumerate(hits, 1):
                key = chunk_key(h)
                if key not in chunks:
                    chunks[key] = chunk_record(h)
                # rank = position inside its own collection+channel result
                # (a real relevance rank); pos = position in the merged list.
                rank = h.get("rank", pos)
                rec = {"chunk": key, "rank": rank,
                       "channel": h.get("channel", ""),
                       "collection": h.get("source", "")}
                if h.get("score") is not None:
                    rec["score"] = h["score"]
                    rec["score_kind"] = h.get("score_kind", "")
                if not args.no_inline_text:
                    rec["text"] = chunks[key]["text"]
                channel_hits[(h.get("source", ""), h.get("channel", ""))] += 1
                out_hits.append(rec)
            u["hits"] = out_hits

        # Rerank BEFORE the union is formed, so the union reflects the chunks
        # that survived reranking rather than everything retrieval touched.
        if reranker:
            st = rerank_question(q, chunks, reranker, args.rerank_per_unit,
                                 args.rerank_top_k)
            for k in rr_stats:
                rr_stats[k] += st[k]
        if args.only_top and q.get("top_chunks"):
            keep = {t["chunk"] for t in q["top_chunks"]}
            for u in q["_units"]:
                u["hits"] = [h for h in (u.get("hits") or [])
                             if h["chunk"] in keep]

        union = {}
        for u in q["_units"]:
            for h in u.get("hits") or []:
                r = h.get("rerank_rank", h.get("rank", 99))
                if h["chunk"] not in union or r < union[h["chunk"]]:
                    union[h["chunk"]] = r
        if q.get("top_chunks"):        # best evidence first, then the rest
            order = [t["chunk"] for t in q["top_chunks"]]
            order += [k for k, _ in sorted(union.items(),
                                           key=lambda kv: kv[1])
                      if k not in order]
            q["union_chunks"] = order
        else:
            q["union_chunks"] = [k for k, _ in sorted(union.items(),
                                                      key=lambda kv: kv[1])]
        q["coverage"] = coverage(
            q, q["_units"],
            "\n".join(chunks[k]["text"] for k in q["union_chunks"]))
        done += 1
        if done % 5 == 0 or done == len(questions):
            print(f"    {done}/{len(questions)} questions  "
                  f"({len(chunks)} distinct chunks, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    # ── write ────────────────────────────────────────────────────────────
    out_questions = []
    for q in questions:
        item = {k: v for k, v in q.items() if not k.startswith("_")}
        item["units"] = q["_units"]
        out_questions.append(item)

    flagged = [q["id"] for q in questions if q["coverage"]["flags"]]

    # Where did the evidence actually come from, and is any single chunk
    # being returned for nearly every question? A chunk that answers most of
    # a set is not evidence, it is a retrieval artefact — and it is invisible
    # unless counted.
    by_coll = Counter(c["collection"] for c in chunks.values())
    hub = Counter(k for q in questions for k in q["union_chunks"])
    hub_thresh = max(3, int(0.5 * len(questions)))
    hubs = [{"chunk": k, "questions": n,
             "source": (chunks[k]["metadata"].get("breadcrumb")
                        or chunks[k]["metadata"].get("name") or ""),
             "collection": chunks[k]["collection"]}
            for k, n in hub.most_common() if n >= hub_thresh]
    print(f"  chunks by collection: {dict(by_coll)}")

    # Per-collection channel health. A collection that answers BM25 queries
    # but never the dense one holds objects WITHOUT VECTORS: semantic
    # retrieval is silently dead for it — no error, just half the engine
    # missing — and everything reading this index, question generation
    # included, is keyword-only on that corpus.
    per_coll_channels = {}
    for coll in primary:
        v = channel_hits.get((coll, "vector"), 0)
        b = channel_hits.get((coll, "bm25"), 0)
        per_coll_channels[coll] = {"vector": v, "bm25": b}
    dead = [c for c, v in per_coll_channels.items()
            if v["bm25"] > 0 and v["vector"] == 0]
    print("  channel health (hits per collection):")
    for c, v in per_coll_channels.items():
        note = ""
        if v["bm25"] and not v["vector"]:
            note = "   <-- NO VECTOR HITS: objects appear to have no " \
                   "embeddings; dense retrieval dead for this collection"
        elif not v["bm25"] and not v["vector"]:
            note = "   (no hits at all)"
        print(f"      {c[:44]:44s} vector {v['vector']:5d}  "
              f"bm25 {v['bm25']:5d}{note}")
    if dead:
        print(f"  ! {len(dead)} collection(s) reachable by keyword only: "
              f"{', '.join(d[:24] for d in dead)}")
        print("    Re-embedding those collections would restore half the "
              "retrieval engine for them.")
    if bm25_gated_total:
        print(f"  bm25 gate ({args.bm25_min_frac:.2f} of best-per-query): "
              f"{bm25_gated_total} weak hit(s) dropped")
    if reranker:
        print(f"  rerank: {rr_stats['unit_pairs']} unit pair(s) + "
              f"{rr_stats['question_pairs']} question pair(s) scored, "
              f"{rr_stats['dropped']} hit(s) trimmed beyond top "
              f"{args.rerank_per_unit}")
    if hubs:
        print(f"  ! {len(hubs)} chunk(s) returned for >= {hub_thresh} of "
              f"{len(questions)} questions — likely retrieval artefacts, not "
              f"evidence:")
        for h in hubs[:5]:
            print(f"      {h['questions']}x  {h['collection']} "
                  f"{h['source'][:52]}")
    payload = {
        "meta": {
            "script": SCRIPT_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "input": str(in_path),
            "questions": len(questions),
            "rows_in_file": len(raw_rows),
            "unparsed": unparsed,
            "formats": fmts,
            "units_queried": n_units,
            "units_skipped_key_code": n_skip,
            "key_codes_queried": bool(args.query_key_codes),
            "collections_searched": primary,
            "collections_skipped_per_file": per_file if per_file is not primary else [],
            "collections_excluded": excluded,
            "embedder": embed_name,
            "k": args.k,
            "retriever": retriever_name,
            "junk_filter": not args.no_junk_filter,
            "inline_text": not args.no_inline_text,
            "distinct_chunks": len(chunks),
            "chunks_by_collection": dict(by_coll),
            "channel_health": per_coll_channels,
            "collections_without_vectors": dead,
            "bm25_min_frac": args.bm25_min_frac,
            "bm25_hits_gated": bm25_gated_total,
            "reranker": rerank_label,
            "rerank_top_k": args.rerank_top_k if reranker else None,
            "rerank_per_unit": args.rerank_per_unit if reranker else None,
            "rerank_pairs_scored": (rr_stats["unit_pairs"]
                                    + rr_stats["question_pairs"]),
            "hub_chunks": hubs,
            "questions_flagged": flagged,
            "retriever_divergence": divergence,
            "elapsed_s": round(time.time() - t0, 1),
        },
        "chunks": chunks,
        "questions": out_questions,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nwrote {out_path}  ({size_mb:.1f} MB, {len(chunks)} distinct "
          f"chunks, {n_units} queries)")
    if divergence:
        print(f"  RETRIEVER DIVERGENCE on {len(divergence)} unit(s): "
              f"{divergence[:5]} — the mirror has drifted from the router")
    if flagged:
        print(f"  QA: {len(flagged)}/{len(questions)} questions flagged for "
              f"thin evidence: {', '.join(flagged[:8])}"
              f"{'…' if len(flagged) > 8 else ''}")
    else:
        print("  QA: every question's retrieved evidence covers its answer")

    if args.md:
        md_path = out_path.with_suffix(".md")
        write_md(md_path, payload, include_text=not args.md_no_text,
                 max_chunks=args.md_max_chunks)
        print(f"wrote {md_path}  (review sheet, "
              f"{md_path.stat().st_size / 1e6:.1f} MB"
              + (", chunk text included" if not args.md_no_text else "")
              + ")")

    # Close the Weaviate connection so the run does not end on a
    # ResourceWarning about a leaked socket.
    for obj in (VECTOR_DB_CLIENT, getattr(VECTOR_DB_CLIENT, "client", None)):
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
                break
            except Exception:
                pass
    return 0


def _fence(text: str) -> str:
    """A code fence longer than any backtick run inside the text, so chunk
    text containing ``` cannot break out of its block."""
    longest = max((len(r) for r in re.findall(r"`+", text or "")), default=0)
    return "`" * max(3, longest + 1)


def write_md(path: Path, payload: dict, include_text: bool = True,
             max_chunks: int = 0) -> None:
    """Review sheet: which units found what, plus the exact chunk text.

    Each question gets short local labels (C1, C2, …) for its distinct
    chunks, ordered by best rank. The unit list references those labels, and
    an evidence section prints each chunk's FULL text once, with the units
    that retrieved it. Chunks are quoted once per question rather than once
    per hit because the same chunk is usually found by several units — the
    text is complete either way, without the repetition.
    """
    m, chunks = payload["meta"], payload["chunks"]
    L = ["# PYQ → RAG chunk map", "",
         f"- input: `{m['input']}`",
         f"- {m['questions']} questions, {m['units_queried']} queries, "
         f"{m['distinct_chunks']} distinct chunks, k={m['k']}",
         f"- embedder: `{m['embedder']}` · retriever: `{m['retriever']}`",
         f"- collections: {', '.join(m['collections_searched'][:6])}",
         f"- flagged for thin evidence: {len(m['questions_flagged'])}", ""]
    for q in payload["questions"]:
        cov = q["coverage"]
        flag = ("  **[" + ", ".join(cov["flags"]) + "]**") if cov["flags"] else ""
        L += [f"## {q['id']} · {q['format']}{flag}", "",
              f"> {q['stem'][:300]}", ""]
        if q.get("answer_text"):
            L.append(f"answer: **{q['answer']}** — {q['answer_text'][:160]}")
        L.append(f"support: answer {cov['answer_support']} "
                 f"({cov['answer_kind']}), claims {cov['claim_support']}, "
                 f"entities {cov['stem_entity_support']}")
        L.append("")

        # distinct chunks for this question, best rank first
        found: dict = {}
        for u in q["units"]:
            for h in u.get("hits", []):
                rec = found.setdefault(h["chunk"], {
                    "rank": h["rank"], "channel": h.get("channel", ""),
                    "score": h.get("score"), "rerank": h.get("rerank"),
                    "units": []})
                rec["units"].append(u["unit"])
                if h["rank"] < rec["rank"]:
                    rec.update(rank=h["rank"], channel=h.get("channel", ""),
                               score=h.get("score"))
                if h.get("rerank") is not None and (
                        rec["rerank"] is None or h["rerank"] > rec["rerank"]):
                    rec["rerank"] = h["rerank"]
        # Reranked chunks lead, in reranked order; a chunk the reranker put
        # first should be the first text a reviewer reads.
        top_order = [t["chunk"] for t in (q.get("top_chunks") or [])]
        rest = sorted((kv for kv in found.items() if kv[0] not in top_order),
                      key=lambda kv: (-(kv[1]["rerank"] if kv[1]["rerank"]
                                        is not None else -1e9),
                                      kv[1]["rank"], kv[0]))
        order = [(c, found[c]) for c in top_order if c in found] + rest
        if max_chunks:
            order = order[:max_chunks]
        label = {k: f"C{i}" for i, (k, _v) in enumerate(order, 1)}

        # the reranked answer first — this is the "most appropriate chunk"
        tops = q.get("top_chunks") or []
        if tops:
            L.append(f"**best evidence** (reranked): "
                     + ", ".join(f"{label.get(t['chunk'], '?')} "
                                 f"(score {t['rerank']:.4g})" for t in tops))
            L.append("")

        for u in q["units"]:
            if not u["queried"]:
                L.append(f"- `{u['unit']}` skipped — {u['skip_reason']}")
                continue
            bits = []
            for h in u["hits"]:
                lab = label.get(h["chunk"])
                if lab:
                    rr = ("" if h.get("rerank") is None
                          else f" rr {h['rerank']:.3g}")
                    bits.append(f"{lab} (r{h['rank']} {h['channel']}{rr})")
            qs_note = ""
            if u.get("query_sent"):
                qs_note = f"  ·  searched as: _{u['query_sent'][:90]}_"
            L.append(f"- `{u['unit']}` → " + (", ".join(bits) or "no hits")
                     + qs_note)
        L.append("")

        if include_text and order:
            L.append("### evidence")
            L.append("")
            for key, rec in order:
                c = chunks.get(key, {})
                md = c.get("metadata", {}) or {}
                where = md.get("breadcrumb") or md.get("name") or ""
                src = md.get("name") or ""
                score = ("" if rec["score"] is None
                         else f" · score {rec['score']:.4g}")
                best = " **← best evidence**" if key in top_order[:1] else (
                    " ← top evidence" if key in top_order else "")
                rr = ("" if rec.get("rerank") is None
                      else f" · rerank {rec['rerank']:.4g}")
                L.append(f"**{label[key]}** · `{src}` · {where}{best}")
                L.append(f"rank {rec['rank']} · {rec['channel']}{score}{rr} · "
                         f"found by: {', '.join(sorted(set(rec['units'])))} · "
                         f"{c.get('chars', 0)} chars · id `{key}`")
                L.append("")
                text = c.get("text") or ""
                f = _fence(text)
                L += [f"{f}text", text, f, ""]
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════
# self-test — offline, no Weaviate, no GPU
# ══════════════════════════════════════════════════════════════════════════

_ST_QUESTIONS = [
    {   # multi-statement, statements embedded in the stem, key-code options
        "id": "t_ms",
        "command_word": "Consider the following statements",
        "question": ("Consider the following statements regarding the "
                     "Directive Principles of State Policy: 1. They are "
                     "enforceable by the courts under Article 32. 2. Article "
                     "39A was inserted by the 42nd Amendment Act, 1976. 3. "
                     "The Principles are borrowed from the Irish "
                     "Constitution. Which of the statements given above "
                     "is/are correct?"),
        "options": ["1 and 2 only", "2 and 3 only", "1 and 3 only",
                    "1, 2 and 3"],
        "answer": "b",
    },
    {   # select/identify with contentful options
        "id": "t_si",
        "format": "select_identify",
        "stem": "Which body recommends the distribution of net tax proceeds "
                "between the Union and the States?",
        "options": ["(a) NITI Aayog", "(b) Finance Commission",
                    "(c) Comptroller and Auditor General",
                    "(d) Reserve Bank of India"],
        "answer": "b",
    },
    {   # assertion-reason, relation options
        "id": "t_ar",
        "format": "assertion_reason",
        "stem": ("Assertion (A): The cash reserve ratio is a monetary policy "
                 "instrument. Reason (R): It determines the share of deposits "
                 "banks keep with the Reserve Bank of India. Select the "
                 "correct answer:"),
        "options": ["Both A and R are true and R explains A",
                    "Both A and R are true but R does not explain A",
                    "A is true but R is false", "A is false but R is true"],
        "answer": "a",
    },
    {   # match the following, pair options
        "id": "t_match",
        "format": "match_the_following",
        "stem": "Match List-I with List-II:",
        "statements": ["Repo rate — Policy rate", "CRR — Reserve requirement",
                       "SLR — Statutory holdings", "MSP — Farm price support"],
        "options": ["A-1, B-2, C-3, D-4", "A-2, B-1, C-4, D-3",
                    "A-3, B-4, C-1, D-2", "A-4, B-3, C-2, D-1"],
        "answer": "a",
    },
    {   # loose shape: no format, no id, dict options
        "text": "The Green Revolution in India primarily increased the "
                "output of which crops, and through which input package?",
        "choices": {"a": "Wheat and rice through HYV seeds and irrigation",
                    "b": "Pulses through dryland farming",
                    "c": "Cotton through export subsidies",
                    "d": "Tea through plantation credit"},
        "answer": "a",
    },
]


class _StubClient:
    """Deterministic fake vector client: returns chunks whose text shares
    words with the query, so unit wiring and schema can be proven offline."""

    CORPUS = [
        ("Cmaster", "DPSP", "The Directive Principles of State Policy are "
         "not enforceable by any court, unlike the Fundamental Rights, but "
         "they are fundamental in the governance of the country. Article 39A, "
         "inserted by the 42nd Amendment Act of 1976, directs the State to "
         "provide free legal aid. The Principles were borrowed from the Irish "
         "Constitution, which had in turn drawn them from the Spanish "
         "Constitution of 1931. " * 2),
        ("Cmaster", "Finance Commission", "The Finance Commission is "
         "constituted by the President under Article 280 every fifth year to "
         "recommend the distribution of the net proceeds of taxes between the "
         "Union and the States, and the principles governing grants-in-aid. "
         "NITI Aayog, by contrast, is an executive think tank with no such "
         "constitutional mandate, and the Comptroller and Auditor General "
         "audits accounts rather than devolving taxes. " * 2),
        ("Cmaster", "Monetary policy", "The cash reserve ratio (CRR) is the "
         "share of net demand and time liabilities that banks must keep with "
         "the Reserve Bank of India, and it is one of the quantitative "
         "instruments of monetary policy along with the repo rate, the "
         "statutory liquidity ratio (SLR) and open market operations. The "
         "minimum support price (MSP) is a farm price support tool, not a "
         "monetary instrument. " * 2),
        ("Cmaster", "Green Revolution", "The Green Revolution raised the "
         "output of wheat and rice sharply through a package of high-yielding "
         "variety seeds, assured irrigation, chemical fertilisers and "
         "institutional credit, concentrated in Punjab, Haryana and western "
         "Uttar Pradesh. Pulses and coarse cereals were largely left out. " * 2),
        ("Cmaster", "Advertisement", "Scan the QR code for our digital "
         "edition and promotional offers."),
    ]

    class _Res:
        def __init__(self, docs, metas):
            self.documents, self.metadatas = [docs], [metas]

    def _score(self, query, text):
        qw = content_words(query)
        tw = content_words(text)
        return len(qw & tw) / max(1, len(qw))

    def _ranked(self, query, limit):
        scored = sorted(self.CORPUS, key=lambda c: -self._score(query, c[2]))
        docs, metas = [], []
        for coll, head, text in scored[:limit]:
            if self._score(query, text) <= 0:
                continue
            docs.append(text)
            metas.append({"raw_text": text, "breadcrumb": head,
                          "name": f"{head}.md",
                          "chunk_id": "ch_" + hashlib.sha1(
                              head.encode()).hexdigest()[:8]})
        return self._Res(docs, metas)

    def search(self, collection_name, vectors, filter=None, limit=5):
        return self._ranked(vectors[0], limit)

    def bm25_search(self, collection_name, query, query_properties=None,
                    limit=5, filter=None):
        return self._ranked(query, limit)


def self_test() -> int:
    checks, fails = 0, []

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(msg)

    qs = [normalise_question(r, i) for i, r in enumerate(_ST_QUESTIONS)]
    by_id = {q["id"]: q for q in qs}

    # format detection / aliasing
    ok(by_id["t_ms"]["format"] == "multi_statement", "ms format")
    ok(by_id["t_si"]["format"] == "select_identify", "si format")
    ok(by_id["t_ar"]["format"] == "assertion_reason", "ar format")
    ok(by_id["t_match"]["format"] == "match_the_following", "match format")
    ok(qs[4]["format"] == "select_identify", "loose-shape format inferred")
    ok(qs[4]["id"] == "q005", "id synthesised for id-less row")

    # statement splitting out of an embedded stem
    ms = by_id["t_ms"]
    ok(len(ms["statements"]) == 3, f"ms statements: {ms['statements']}")
    ok("Article 32" in ms["statements"][0], "statement 1 text")
    ok(ms["statements"][2].rstrip(".").endswith("Irish Constitution"),
       f"instruction tail not stripped: {ms['statements'][2]!r}")
    ok("Which of the statements" not in " ".join(ms["statements"]),
       "tail leaked into statements")
    ok("42nd Amendment" in ms["statements"][1], "statement 2 text")

    # dict options + label stripping + answer_text derivation
    ok(qs[4]["options"][0].startswith("Wheat and rice"), "dict options order")
    ok(by_id["t_si"]["options"][1] == "(b) Finance Commission", "opts kept raw")
    ok(by_id["t_si"]["answer_text"] == "Finance Commission",
       f"answer_text: {by_id['t_si']['answer_text']!r}")

    # answer marking: every real-world shape, and the empty case that used to
    # crash the whole run ("" in "abcd" is True in Python, so ord("") blew up)
    OPTS = ["NITI Aayog", "Finance Commission", "CAG", "RBI"]
    for val, want in (("b", "b"), ("B", "b"), ("(b)", "b"), ("b.", "b"),
                      ("option_b", "b"), ("Option B", "b"), (2, "b"),
                      ("2", "b"), ("Finance Commission", "b"),
                      (["b"], "b"), ("", ""), (None, ""), ("z", "z"),
                      ("99", ""), ({}, "")):
        row = {"stem": "x", "options": OPTS}
        if val is not None:
            row["answer"] = val
        got, _txt = resolve_answer(row, OPTS)
        ok(got == want, f"answer {val!r} → {got!r}, expected {want!r}")
    for bad in ({"stem": "x", "options": OPTS},
                {"stem": "x", "options": OPTS, "answer": ""},
                {"stem": "x", "options": OPTS, "answer": None},
                {"stem": "x", "options": [], "answer": "b"},
                {"stem": "x"}):
        try:
            nq = normalise_question(bad, 0)
            ok(True, "")
            ok(isinstance(nq["answer"], str) and
               isinstance(nq["answer_text"], str),
               f"answer fields not strings for {bad}")
        except Exception as e:
            fails.append(f"normalise_question crashed on {bad}: "
                         f"{type(e).__name__}: {e}")
    ok(resolve_answer({"options": OPTS, "answer": 2}, OPTS)[1] ==
       "Finance Commission", "index answer did not resolve its text")
    ok(resolve_answer({"options": OPTS, "answer_text": "RBI"}, OPTS)[1] ==
       "RBI", "explicit answer_text lost")

    # key-code classification
    for code in ["1 and 2 only", "All three", "None of the above",
                 "A-1, B-2, C-3, D-4", "Both A and R are true and R explains A",
                 "2 only", "I and III only", "(c)", "Neither 1 nor 2",
                 # real rows from pyq_polity_normalized.jsonl that a plain
                 # content-word count let through as "content"
                 "There are two correct statements that include statement 3.",
                 "There is no correct statement.",
                 "All three statements are correct.",
                 "There is only one correct statement.",
                 "The views of both X and Y are correct.",
                 "Only two of the above pairs are correctly matched.",
                 "Statement I is correct but Statement II is incorrect."]:
        ok(classify_option(code)[0] == "key_code", f"not a key code: {code!r}")
    # a bare numeral is statement-2 in a coded format, the number 2 elsewhere
    ok(classify_option("2", "multi_statement")[0] == "key_code",
       "bare numeral in multi_statement should be a statement reference")
    ok(classify_option("2", "select_identify")[0] == "content",
       "bare numeral in select_identify is the answer")
    # numbered references are the sharpest BM25 queries — never discard them
    for named in ["Article 350-A", "42nd Amendment", "91st", "16", "24",
                  "First", "Fourth Schedule", "Schedule VII",
                  "1st Amendment", "Part IV", "Article 19",
                  # single proper nouns and values are complete answers
                  "Germany", "Calcutta", "Hidayatullah", "1971 Census",
                  "3 months", "Rs 10,000 and Rs 2,500", "Union List"]:
        kind, why = classify_option(named, "select_identify")
        ok(kind == "content", f"real answer dropped: {named!r} ({why})")
    # mixed roman/arabic markers in one stem (real 2017 paper shape)
    mixed = ("With reference to the Parliament of India, consider the "
             "following statements : i. A private member's bill is presented "
             "by a Member of Parliament. 2. Such a bill has been passed once "
             "in its history. Which of the statements given above is/are "
             "correct ?")
    _lead, mixed_sts = split_statements(mixed)
    ok(len(mixed_sts) == 2, f"mixed roman/arabic markers: {mixed_sts}")
    ok("Which of the statements" not in " ".join(mixed_sts),
       f"tail leaked in mixed-marker split: {mixed_sts}")
    # modern UPSC (2023+) assertion-reason wording, and its relation options
    modern = ("Consider the following statements in respect of the "
              "Constitution Day : Statement-I : The Constitution Day is "
              "celebrated on 26th November. Statement-II : The Constituent "
              "Assembly adopted the Constitution on that date in 1949.")
    a2, r2 = split_assertion_reason(modern, [])
    ok("26th November" in a2 and "Statement-II" not in a2,
       f"Statement-I not isolated: {a2!r}")
    ok("Constituent Assembly" in r2, f"Statement-II not isolated: {r2!r}")
    for rel in ["Both Statement-I and Statement-II are correct and "
                "Statement-II is the correct explanation for Statement-I",
                "Both Statement-I and Statement-II are correct and "
                "Statement-II is not the correct explanation for Statement-I",
                "Statement-I is correct but Statement-II is incorrect",
                "2 4 1 3", "3 1 4 2"]:
        kind, why = classify_option(rel, "assertion_reason")
        ok(kind == "key_code", f"relation/code read as content: {rel[:44]!r}")

    # match-the-following: List-I (lettered) and List-II (numbered) are
    # separate query vocabularies; the pairing itself is the answer and must
    # never be guessed
    matchq = normalise_question({
        "id": "t_lists", "question-type": "Match the Following",
        "stem": ("Match List I (Article of Indian Constitution) with List II "
                 "(Provisions) and select the correct answer using the codes "
                 "given below the lists: List I (Article) A. Article 16(2) "
                 "B. Article 29(2) C. Article 30(1) D. Article 31 List II "
                 "(Provisions) 1. No person shall be deprived of his property "
                 "save by authority of law 2. No person can be discriminated "
                 "against in public appointment 3. All minorities may "
                 "establish institutions 4. No citizen shall be denied "
                 "admission into any educational institution"),
        "options": ["(a) A-2, B-4, C-3, D-1", "(b) A-3, B-1, C-2, D-4"],
    }, 0)
    mu = {u["unit"]: u["query"] for u in build_units(matchq)}
    ok("list1_a" in mu and "Article 16(2)" in mu["list1_a"],
       f"List-I entries lost: {sorted(mu)}")
    ok(sum(1 for k in mu if k.startswith("list1_")) == 4,
       f"List-I count: {[k for k in mu if k.startswith('list1_')]}")
    ok(sum(1 for k in mu if k.startswith("list2_")) == 4,
       f"List-II count: {[k for k in mu if k.startswith('list2_')]}")
    ok("List II" not in mu.get("list1_d", ""),
       f"List-II header leaked into List-I: {mu.get('list1_d')!r}")
    ok(all(not u["queried"] for u in build_units(matchq)
           if u["unit"].startswith("option_")),
       "match codes should be skipped")

    # inverted layout: List I NUMBERED, List II LETTERED. Parsing by marker
    # style mislabelled the sides and let the List II block bleed into the
    # last List I item.
    inv = normalise_question({
        "id": "t_inv", "command_word": "Match the following",
        "question": ("Match List I with List II and select the correct "
                     "answer: List I (Amendment) 1. The Constitution "
                     "(Seventy-third Amendment) Act, 1992 2. The "
                     "Constitution (Eighty-third Amendment) Act, 2000 "
                     "List II (Contents) A) Establishment of Panchayats "
                     "B) No reservations for Scheduled Castes in Arunachal "
                     "Pradesh Select the correct answer using the codes "
                     "given below:"),
        "options": ["1-A, 2-B", "1-B, 2-A"],
    }, 0)
    iu = {u["unit"]: u["query"] for u in build_units(inv)}
    ok(sum(1 for k in iu if k.startswith("list1_")) == 2,
       f"inverted List-I count: {[k for k in iu if k.startswith('list1_')]}")
    ok(sum(1 for k in iu if k.startswith("list2_")) == 2,
       f"inverted List-II count: {[k for k in iu if k.startswith('list2_')]}")
    ok(all("List II" not in v and "List I" not in v
           for k, v in iu.items() if k.startswith("list")),
       f"list header leaked: {iu}")
    ok(any("Seventy-third" in v for k, v in iu.items()
           if k.startswith("list1_")),
       f"amendments should be List I: {iu}")
    ok(any("Panchayats" in v for k, v in iu.items()
           if k.startswith("list2_")),
       f"contents should be List II: {iu}")
    ok(all("codes given below" not in v for k, v in iu.items()
           if k.startswith("list")),
       f"instruction tail leaked into a list item: {iu}")

    # a case citation must not be read as a roman marker
    cite = ("Consider the following: 1. Golaknath v. State of Punjab was "
            "decided in 1967. 2. It was overruled later.")
    _l2, cite_sts = split_statements(cite)
    ok(len(cite_sts) == 2, f"case citation broke the split: {cite_sts}")
    ok("Golaknath v. State of Punjab" in cite_sts[0],
       f"citation mangled: {cite_sts[0]!r}")
    for content in ["Finance Commission", "(a) NITI Aayog",
                    "Wheat and rice through HYV seeds and irrigation",
                    "Comptroller and Auditor General",
                    # a reference word must not veto a genuinely named option
                    "The statement is protected under Article 19",
                    "Sarkaria Commission on Centre-State relations"]:
        ok(classify_option(content)[0] == "content",
           f"wrongly a key code: {content!r}")

    # run-on instruction tail (real 2025 paper shape): the last list item is
    # followed by the question with no punctuation between them
    runon = ("Conditions of the Governor's office For a constitutional "
             "amendment ratification by the States is required in respect "
             "of which of the above?")
    ok(strip_instruction_tail(runon) == "Conditions of the Governor's office",
       f"run-on tail not cut: {strip_instruction_tail(runon)!r}")
    # ... but a statement that merely starts a clause with such a word and
    # carries no instruction must survive intact
    keep = ("The Governor may reserve a Bill for the President With such "
            "reservation the Bill lapses")
    ok(strip_instruction_tail(keep) == keep,
       f"wrongly truncated: {strip_instruction_tail(keep)!r}")

    # units per format
    u_ms = build_units(ms)
    names = [u["unit"] for u in u_ms]
    ok(names[:4] == ["stem", "statement_1", "statement_2", "statement_3"],
       f"ms units: {names}")
    ok(all(not u["queried"] and "key_code" in u["skip_reason"]
           for u in u_ms if u["unit"].startswith("option_")),
       "ms options should be skipped key codes")
    u_ar = build_units(by_id["t_ar"])
    ok([u["unit"] for u in u_ar][:3] == ["stem", "assertion", "reason"],
       f"ar units: {[u['unit'] for u in u_ar]}")
    ok("cash reserve ratio" in
       [u for u in u_ar if u["unit"] == "assertion"][0]["query"],
       "assertion text")
    ok("Reserve Bank" in
       [u for u in u_ar if u["unit"] == "reason"][0]["query"], "reason text")
    u_m = build_units(by_id["t_match"])
    ok(sum(1 for u in u_m if u["unit"].startswith("pair_")) == 4,
       f"match pairs: {[u['unit'] for u in u_m]}")
    ok("Repo rate : Policy rate" ==
       [u for u in u_m if u["unit"] == "pair_1"][0]["query"], "pair join")
    u_si = build_units(by_id["t_si"])
    ok(sum(1 for u in u_si if u["unit"].startswith("option_")
           and u["queried"]) == 4, "si options must all be queried")
    ok("(b)" not in [u for u in u_si if u["unit"] == "option_b"][0]["query"],
       "option label not stripped")

    # forced key codes
    ok(all(u["queried"] for u in build_units(ms, query_key_codes=True)
           if u["unit"].startswith("option_")), "--query-key-codes ignored")

    # end-to-end against the stub client
    client = _StubClient()
    chunks: dict = {}
    for q in qs:
        q["_units"] = build_units(q)
        union: dict = {}
        for u in q["_units"]:
            if not u["queried"]:
                u["hits"] = []
                continue
            hits = _dual_channel_retrieve(client, ["Cmaster"], u["query"],
                                          u["query"], 3)
            hits = [h for h in hits if not is_junk_chunk(h["text"])]
            u["hits"] = []
            for rank, h in enumerate(hits, 1):
                key = chunk_key(h)
                chunks.setdefault(key, chunk_record(h))
                u["hits"].append({"chunk": key, "rank": rank,
                                  "channel": h["channel"],
                                  "text": chunks[key]["text"]})
                if key not in union or rank < union[key]:
                    union[key] = rank
        q["union_chunks"] = [k for k, _ in sorted(union.items(),
                                                 key=lambda kv: kv[1])]
        q["coverage"] = coverage(
            q, q["_units"],
            "\n".join(chunks[k]["text"] for k in q["union_chunks"]))

    # Multi-collection fairness: with several collections searched, the first
    # one in list order must not consume every slot. The router's single
    # `hits[:k*2]` cap did exactly that — one alphabetically-first corpus
    # supplied 90% of a run's chunks while the collection holding the real
    # source material contributed nothing, silently.
    class _ManyColl:
        """Distinct corpora, as in reality: each collection holds its own
        documents, all of them relevant to the query."""

        class _Res:
            def __init__(self, docs, metas):
                self.documents, self.metadatas = [docs], [metas]

        def _ranked(self, coll, limit):
            docs = [f"{coll} document {i} about the Directive Principles of "
                    f"State Policy and Article 39A, at sufficient length to "
                    f"survive the junk filter. " * 3 for i in range(limit)]
            metas = [{"raw_text": d, "name": f"{coll}_{i}.md"}
                     for i, d in enumerate(docs)]
            return self._Res(docs, metas)

        def search(self, collection_name, vectors, filter=None, limit=5):
            return self._ranked(collection_name, limit)

        def bm25_search(self, collection_name, query, query_properties=None,
                        limit=5, filter=None):
            return self._ranked(collection_name, limit)

    colls = ["Aaa_first", "Bbb_second", "Ccc_third"]
    multi = _dual_channel_retrieve(_ManyColl(), colls,
                                   "Directive Principles enforceable court",
                                   "Directive Principles enforceable court", 2)
    got = {h["source"] for h in multi}
    ok(got == set(colls),
       f"collections starved by the per-call cap: got {sorted(got)}, "
       f"expected {colls}")
    ok(all(h.get("rank", 0) >= 1 for h in multi), "hits missing a rank")
    ok(max((h["rank"] for h in multi), default=0) <= 2,
       "rank should be per collection+channel, not global position")
    bounded = _dual_channel_retrieve(_ManyColl(), colls, "q", "q", 2,
                                     max_hits=3)
    ok(len(bounded) == 3, f"max_hits ignored: {len(bounded)}")

    # score direction is read off the data, not the attribute name
    class _Sim:
        class R:
            documents = [["a " * 100, "b " * 100]]
            metadatas = [[{"raw_text": "a " * 100}, {"raw_text": "b " * 100}]]
            distances = [[0.92, 0.71]]          # falls with rank -> similarity

        def search(self, **kw):
            return self.R()

        def bm25_search(self, **kw):
            return self.R()
    sim = _dual_channel_retrieve(_Sim(), ["X"], "q", "q", 2)
    ok(any(h["score_kind"] == "vector_similarity" for h in sim),
       f"similarity mislabelled: {[h['score_kind'] for h in sim]}")

    # bm25 relative gate: weak hits from an idf-inflated small collection go,
    # the strong ones stay
    gate_hits = [{"channel": "bm25", "score": 100.0, "text": "keep"},
                 {"channel": "bm25", "score": 30.0, "text": "keep"},
                 {"channel": "bm25", "score": 0.9, "text": "drop"},
                 {"channel": "vector", "score": 0.7, "text": "keep"}]
    kept, dropped = apply_bm25_gate(gate_hits, 0.2)
    ok(dropped == 1 and all(h["text"] == "keep" for h in kept),
       f"bm25 gate wrong: kept {[h['text'] for h in kept]}")
    ok(apply_bm25_gate(gate_hits, 0)[1] == 0, "gate should be disablable")
    # a unit whose only hits are from one collection is compared to itself
    solo = [{"channel": "bm25", "score": 0.9, "text": "keep"}]
    ok(apply_bm25_gate(solo, 0.2)[1] == 0,
       "a lone collection must not gate itself away")

    # context-poor units get the stem prefixed; self-sufficient ones do not
    sched = normalise_question({
        "id": "t_sched", "format": "select_identify",
        "stem": "If a new State of the Indian Union is to be created, which "
                "one of the following Schedules must be amended?",
        "options": ["First", "Second", "Third", "Fifth"]}, 0)
    su = {u["unit"]: u for u in build_units(sched)}
    ok(su["option_a"]["query"] == "First", "unit text must stay verbatim")
    ok("Schedules must be amended" in su["option_a"].get("query_sent", ""),
       f"ordinal option not given stem context: {su['option_a']}")
    ok(su["option_a"].get("context_added") is True, "context flag missing")
    ok("query_sent" not in su["stem"], "stem must not prefix itself")
    named = {u["unit"]: u for u in build_units(normalise_question(
        {"id": "t_named", "format": "select_identify", "stem": "Which city?",
         "options": ["Calcutta", "Madras"]}, 0))}
    ok("query_sent" not in named["option_a"],
       "a proper noun stands alone and must not be padded")
    plain = {u["unit"]: u for u in build_units(sched, stem_context=False)}
    ok("query_sent" not in plain["option_a"], "--no-stem-context ignored")

    ok(all(q["union_chunks"] for q in qs),
       "every question must retrieve at least one chunk")
    ok(all(chunks[k]["text"] for q in qs for k in q["union_chunks"]),
       "chunk store must hold full text")
    ok(not any("Scan the QR" in chunks[k]["text"]
               for q in qs for k in q["union_chunks"]),
       "junk chunk was not filtered")
    dpsp = [k for k in by_id["t_ms"]["union_chunks"]
            if "Directive Principles" in chunks[k]["text"]]
    ok(dpsp, "ms question did not retrieve the DPSP chunk")
    fc = by_id["t_si"]["union_chunks"]
    ok(any("Finance Commission" in chunks[k]["text"] for k in fc),
       "si question did not retrieve the Finance Commission chunk")
    ok(by_id["t_si"]["coverage"]["answer_support"] == 1.0,
       f"answer support: {by_id['t_si']['coverage']}")
    ok(by_id["t_si"]["coverage"]["answer_kind"] == "content",
       "si answer should be scoreable content")
    # key-code answers must NOT be scored as answer text — that was a false
    # alarm on every multi-statement and assertion-reason question.
    for qid in ("t_ms", "t_ar", "t_match"):
        cov = by_id[qid]["coverage"]
        ok(cov["answer_kind"] == "key_code" and cov["answer_support"] is None,
           f"{qid} answer wrongly scored: {cov}")
        ok("answer_text_thin" not in cov["flags"],
           f"{qid} falsely flagged on a key-code answer: {cov}")
    ok(by_id["t_ar"]["coverage"]["flags"] == [],
       f"ar flagged unexpectedly: {by_id['t_ar']['coverage']}")
    ok((by_id["t_ar"]["coverage"]["claim_support"] or 0) > 0.5,
       f"ar claims not covered: {by_id['t_ar']['coverage']}")
    ok((by_id["t_ms"]["coverage"]["claim_support"] or 0) > 0.5,
       f"ms claims not covered: {by_id['t_ms']['coverage']}")

    # ── reranking ────────────────────────────────────────────────────────
    lex, lex_label = make_reranker("", "lexical")
    ok("lexical" in lex_label, f"lexical label {lex_label}")
    # the passage that ANSWERS the question must outscore one merely on-topic
    ansr = ("Article 32 confers the right to constitutional remedies and Dr "
            "Ambedkar called it the very heart and soul of the Constitution.")
    topical = ("The Constitution of India is the longest written constitution "
               "in the world and contains many articles.")
    s = lex("Which right did Ambedkar call the heart and soul of the "
            "Constitution?", [topical, ansr])
    ok(s[1] > s[0], f"lexical reranker preferred the topical chunk: {s}")

    # a stub cross-encoder, so the plumbing is tested without a model
    def stub_score(query, texts):
        qw = content_words(query)
        return [len(qw & content_words(t)) for t in texts]

    rq = normalise_question({
        "id": "t_rr", "format": "multi_statement",
        "question": ("Consider the following statements about Article 32: "
                     "1. It is a fundamental right. 2. It can be suspended "
                     "during a National Emergency. Which of the statements "
                     "given above is/are correct?"),
        "options": ["1 only", "2 only", "Both 1 and 2", "Neither 1 nor 2"]},
        0)
    rq["_units"] = build_units(rq)
    rchunks = {
        "good": {"text": "Article 32 is itself a fundamental right and the "
                         "Supreme Court has ruled it a basic feature; it may "
                         "be suspended during a National Emergency under "
                         "Article 359.", "chars": 150, "collection": "X",
                 "metadata": {"name": "ch08.md"}},
        "meh": {"text": "The Constituent Assembly met between 1946 and 1949 "
                        "and adopted the Constitution.", "chars": 80,
                "collection": "X", "metadata": {"name": "ch02.md"}},
        "bad": {"text": "Panchayati Raj institutions were given "
                        "constitutional status by the 73rd Amendment.",
                "chars": 90, "collection": "X",
                "metadata": {"name": "ch77.md"}},
    }
    for u in rq["_units"]:
        u["hits"] = ([{"chunk": k, "rank": i, "channel": "vector"}
                      for i, k in enumerate(rchunks, 1)]
                     if u["queried"] else [])
    st = rerank_question(rq, rchunks, stub_score, per_unit=2, top_k=1)
    ok(len(rq["top_chunks"]) == 1, f"top_k not honoured: {rq['top_chunks']}")
    ok(rq["top_chunks"][0]["chunk"] == "good",
       f"wrong chunk chosen: {rq['top_chunks']}")
    ok(all(len(u["hits"]) <= 2 for u in rq["_units"] if u["queried"]),
       "per-unit trim not applied")
    ok(st["dropped"] > 0 and st["unit_pairs"] > 0, f"rerank stats {st}")
    ok(all(h.get("rerank") is not None for u in rq["_units"]
           for h in u["hits"]), "hits missing rerank scores")
    ok(rq["top_chunks"][0]["found_by"], "top chunk lacks unit attribution")
    # key codes must not pollute the question-level rerank query
    qq = question_rerank_query(rq, rq["_units"])
    ok("Neither 1 nor 2" not in qq and "Article 32" in qq,
       f"question rerank query wrong: {qq[:120]}")
    # per_unit=0 keeps everything
    for u in rq["_units"]:
        u["hits"] = ([{"chunk": k, "rank": i, "channel": "vector"}
                      for i, k in enumerate(rchunks, 1)]
                     if u["queried"] else [])
    rerank_question(rq, rchunks, stub_score, per_unit=0, top_k=3)
    ok(any(len(u["hits"]) == 3 for u in rq["_units"] if u["queried"]),
       "per_unit=0 should keep all hits")

    # entity extraction must name the thing, not the sentence: a leading
    # determiner made "The Dinesh Goswami Committee" the referent, which no
    # chapter contains verbatim, so a question answered at rank 1 was flagged
    ents = entities("The Dinesh Goswami Committee was concerned with")
    ok("Dinesh Goswami Committee" in ents,
       f"determiner not stripped: {sorted(ents)}")
    ok("The Dinesh Goswami Committee" not in ents,
       f"determiner-led form kept: {sorted(ents)}")
    cov_ok = coverage(
        {"stem_full": "The Dinesh Goswami Committee was concerned with",
         "answer_text": "", "format": "select_identify", "statements": []},
        [],
        "The Dinesh Goswami Committee recommended electoral reforms in 1990.")
    ok("stem_entities_thin" not in cov_ok["flags"],
       f"false thin-entity flag: {cov_ok}")

    # a question whose evidence is absent must be flagged, not silently passed
    ghost = normalise_question(
        {"id": "t_ghost", "format": "select_identify",
         "stem": "Which committee recommended the Panchsheel doctrine of "
                 "lunar cartography?",
         "options": ["Kothari Committee", "Sarkaria Commission",
                     "Punchhi Commission", "Rajamannar Committee"],
         "answer": "a"}, 99)
    ghost["_units"] = build_units(ghost)
    gunion: dict = {}
    for u in ghost["_units"]:
        if not u["queried"]:
            continue
        hits = [h for h in _dual_channel_retrieve(
            client, ["Cmaster"], u["query"], u["query"], 3)
            if not is_junk_chunk(h["text"])]
        for rank, h in enumerate(hits, 1):
            key = chunk_key(h)
            chunks.setdefault(key, chunk_record(h))
            gunion.setdefault(key, rank)
    ghost["union_chunks"] = list(gunion)
    ghost["coverage"] = coverage(
        ghost, ghost["_units"],
        "\n".join(chunks[k]["text"] for k in ghost["union_chunks"]))
    ok(ghost["coverage"]["flags"], "absent-evidence question was not flagged")

    # output payload must be JSON-serialisable and round-trip
    payload = {"meta": {"script": SCRIPT_VERSION, "questions": len(qs),
                        "input": "self-test", "formats": {},
                        "units_queried": 0, "distinct_chunks": len(chunks),
                        "questions_flagged": [], "k": 3,
                        "embedder": "stub", "retriever": "stub",
                        "collections_searched": ["Cmaster"]},
               "chunks": chunks,
               "questions": [{k: v for k, v in q.items()
                              if not k.startswith("_")} | {"units": q["_units"]}
                             for q in qs]}
    try:
        rt = json.loads(json.dumps(payload, ensure_ascii=False))
        ok(len(rt["questions"]) == len(qs), "payload question count")
        ok(all("units" in q for q in rt["questions"]), "units in payload")
    except Exception as e:
        fails.append(f"payload not serialisable: {e}")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.md"
        try:
            write_md(p, payload)
            ok(p.stat().st_size > 200, "md review sheet too small")
            md = p.read_text(encoding="utf-8")
            # the exact chunk text must be present, verbatim and complete
            any_text = next((c["text"] for c in chunks.values()
                             if c.get("text")), "")
            ok(any_text and any_text in md,
               "review sheet does not contain the exact chunk text")
            ok("### evidence" in md and "**C1**" in md,
               "evidence section or chunk labels missing")
            ok("found by:" in md, "unit attribution missing")
            # a chunk containing a code fence must not break out of its block
            fenced = dict(payload["chunks"])
            k0 = next(iter(fenced))
            fenced[k0] = dict(fenced[k0],
                              text="before ``` inside ``` after " * 20)
            p2 = Path(td) / "y.md"
            write_md(p2, dict(payload, chunks=fenced))
            body = p2.read_text(encoding="utf-8")
            ok("````text" in body or "`````text" in body,
               "fence not widened for text containing backticks")
            # text can be switched off, and capped
            p3 = Path(td) / "z.md"
            write_md(p3, payload, include_text=False)
            ok("### evidence" not in p3.read_text(encoding="utf-8"),
               "--md-no-text still emitted evidence text")
            p4 = Path(td) / "w.md"
            write_md(p4, payload, max_chunks=1)
            ok(p4.read_text(encoding="utf-8").count("**C2**") == 0,
               "--md-max-chunks not honoured")
        except Exception as e:
            fails.append(f"write_md failed: {e}")

        # input loaders: JSON list, wrapped, JSONL
        for name, body in (
                ("a.json", json.dumps(_ST_QUESTIONS)),
                ("b.json", json.dumps({"questions": _ST_QUESTIONS})),
                ("c.jsonl", "\n".join(json.dumps(r) for r in _ST_QUESTIONS))):
            f = Path(td) / name
            f.write_text(body, encoding="utf-8")
            try:
                ok(len(load_questions(f)) == len(_ST_QUESTIONS),
                   f"loader failed on {name}")
            except SystemExit as e:
                fails.append(f"loader raised on {name}: {e}")

    print(f"pyq_chunk_extractor self-test: {checks} checks, "
          f"{len(fails)} failures")
    for f in fails:
        print(f"  FAIL: {f}")
    return 1 if fails else 0


# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Retrieve the exact RAG chunks behind every PYQ "
                    "(stem, statements, options) into one JSON file.")
    ap.add_argument("--input", help="questions JSON / JSONL")
    ap.add_argument("--out", default="backend/data/pyq_chunk_map.json")
    ap.add_argument("--k", type=int, default=int(
        os.environ.get("RAG_TOP_K", "8")),
        help="hits per channel per unit (default: RAG_TOP_K)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N questions (smoke run)")
    ap.add_argument("--max-hits-per-unit", type=int, default=0,
                    help="bound the merged hits per unit (0 = unbounded; "
                         "each collection is already capped at 2*k)")
    ap.add_argument("--collections", default="",
                    help="comma-separated collection names (skip discovery)")
    ap.add_argument("--knowledge-id", default="",
                    help="search only this knowledge base's master collection")
    ap.add_argument("--masters-only", action="store_true",
                    help="skip per-file collections (a KB with an empty "
                         "master then contributes nothing)")
    ap.add_argument("--list-collections", action="store_true",
                    help="probe every Weaviate collection and exit — shows "
                         "which actually hold content")
    ap.add_argument("--query-key-codes", action="store_true",
                    help="also query key-code options ('1 and 2 only')")
    ap.add_argument("--no-rerank", action="store_true",
                    help="skip cross-encoder reranking (keeps every hit)")
    ap.add_argument("--rerank-model",
                    default=(os.environ.get("RAG_RERANKING_MODEL")
                             or "BAAI/bge-reranker-v2-m3"),
                    help="cross-encoder model (default: RAG_RERANKING_MODEL, "
                         "else BAAI/bge-reranker-v2-m3; "
                         "BAAI/bge-reranker-base is lighter)")
    ap.add_argument("--rerank-mode", default="cross-encoder",
                    choices=["cross-encoder", "lexical"],
                    help="lexical needs no model download but is weaker")
    ap.add_argument("--rerank-top-k", type=int, default=3,
                    help="best chunks per QUESTION (default 3)")
    ap.add_argument("--rerank-per-unit", type=int, default=3,
                    help="hits kept per unit after reranking (default 3, "
                         "0 = keep all)")
    ap.add_argument("--only-top", action="store_true",
                    help="output ONLY the question's top chunks — unit hit "
                         "lists are filtered down to them")
    ap.add_argument("--bm25-min-frac", type=float, default=0.2,
                    help="drop BM25 hits scoring below this fraction of the "
                         "best BM25 score for the same query (0 disables). "
                         "Counters small-collection idf inflation.")
    ap.add_argument("--no-stem-context", action="store_true",
                    help="query bare ordinal options ('First') alone instead "
                         "of prefixing the stem for context")
    ap.add_argument("--no-inline-text", action="store_true",
                    help="chunk text only in the `chunks` map, not per hit")
    ap.add_argument("--no-junk-filter", action="store_true",
                    help="keep ad/banner chunks the generator drops")
    ap.add_argument("--router-retriever", action="store_true",
                    help="import the router's retrieval function itself")
    ap.add_argument("--verify-retriever", action="store_true",
                    help="run both retrievers and report any divergence")
    ap.add_argument("--md", action="store_true",
                    help="also write a .md review sheet next to the JSON")
    ap.add_argument("--md-no-text", action="store_true",
                    help="review sheet lists sources only, without the full "
                         "chunk text")
    ap.add_argument("--md-max-chunks", type=int, default=0,
                    help="cap the chunks quoted per question in the review "
                         "sheet (0 = all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned queries, embed nothing")
    ap.add_argument("--dry-run-show", type=int, default=5,
                    help="questions to print in a dry run")
    ap.add_argument("--self-test", action="store_true",
                    help="offline correctness suite (no Weaviate, no GPU)")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())
    if args.list_collections:
        env = load_env(ROOT)
        # Counts come straight from Weaviate's REST/GraphQL API, so this
        # needs no backend import and no model load at all.
        sys.exit(list_collections(env, None))
    if not args.input:
        ap.error("--input is required (or use --self-test)")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
