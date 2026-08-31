#!/usr/bin/env python3
"""
render.py — deterministic renderer for the semantic question schema (v1).

The fine-tuned model emits truth values, never letters:

    {"format":"multi_statement","stem":"...",
     "statements":[{"text":"...","truth":true},{"text":"...","truth":false}]}

    {"format":"select_identify","stem":"...",
     "choices":[{"text":"...","correct":false},{"text":"...","correct":true},...]}

This module owns everything letter-shaped: the answer-code option set, the
shuffle, the letters, and the answer key. Because the key is COMPUTED from the
truth values rather than emitted by the model, the answer-key defect class that
plagued the pre-fine-tune batches cannot occur here by construction.

Design decisions, each anchored to a measured property of the corpus:

* 9.8% of multi_statement targets are all-false and 23.5% all-true, so both
  ends must render. All-false questions use the "How many of the above
  statements are correct?" variant, where "None" is a standard UPSC option —
  distractor rule D4 ("never none-of-the-above") governs select_identify
  distractors, not answer codes, which are a different vocabulary.
* Training stems had their trailing lead-outs stripped (616 were swallowed into
  the last statement by PDF extraction), so the renderer ADDS the lead-out
  back: that is part of the contract, not an accident.
* A stem that already contains "?" gets "Select the correct answer using the
  code given below." — appending "Which of the statements given above is/are
  correct?" after a question reads as two questions.
* Distractor codes are chosen nearest-first by Hamming distance from the true
  set, so wrong options are near-misses ("1 and 2 only" against a true "1, 2
  and 3") rather than throwaways — same principle as distractor rule D1.

Zero dependencies. `python render.py` runs the exhaustive self-test:
every subset of every statement count 2..5, both variants, multiple seeds.
"""
from __future__ import annotations

import json
import random
import zlib

LETTERS = "abcde"
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


# ─────────────────────────────────────────────────────────────────────────────
# code strings
# ─────────────────────────────────────────────────────────────────────────────
def _join(nums: list[int]) -> str:
    nums = sorted(nums)
    if len(nums) == 1:
        return str(nums[0])
    if len(nums) == 2:
        return f"{nums[0]} and {nums[1]}"
    return ", ".join(map(str, nums[:-1])) + f" and {nums[-1]}"


def subset_code(true_set: frozenset, n: int) -> str:
    """The UPSC answer-code string naming exactly `true_set` out of n statements."""
    s = sorted(true_set)
    if not s:
        return "Neither 1 nor 2" if n == 2 else "None of the above"
    if len(s) == n:
        return "Both 1 and 2" if n == 2 else _join(s)
    if len(s) == 1:
        return f"{s[0]} only"
    return f"{_join(s)} only"


def how_many_code(k: int, n: int) -> str:
    if k == 0:
        return "None"
    if k == n:
        return f"All {WORD[n]}"
    return f"Only {WORD[k]}"


def _rng(seed) -> random.Random:
    if seed is None:
        return random.Random()
    return random.Random(zlib.crc32(repr(seed).encode("utf-8")))


# ─────────────────────────────────────────────────────────────────────────────
# validation of the model's raw object
# ─────────────────────────────────────────────────────────────────────────────
def validate(obj: dict) -> list[str]:
    """Schema-level issues with a model output. Empty list = renderable."""
    issues = []
    fmt = obj.get("format")
    if fmt == "multi_statement":
        st = obj.get("statements")
        if not isinstance(st, list) or len(st) < 2:
            issues.append("multi_statement needs >= 2 statements")
        else:
            if len(st) > 5:
                issues.append("more than 5 statements")
            for i, s in enumerate(st):
                if not isinstance(s, dict) or not (s.get("text") or "").strip():
                    issues.append(f"statement {i+1} has no text")
                elif not isinstance(s.get("truth"), bool):
                    issues.append(f"statement {i+1} truth is not a boolean")
    elif fmt == "select_identify":
        ch = obj.get("choices")
        if not isinstance(ch, list) or not (3 <= len(ch) <= 5):
            issues.append("select_identify needs 3-5 choices")
        else:
            good = [c for c in ch if isinstance(c, dict)
                    and (c.get("text") or "").strip()
                    and isinstance(c.get("correct"), bool)]
            if len(good) != len(ch):
                issues.append("every choice needs text and a boolean 'correct'")
            elif sum(c["correct"] for c in ch) != 1:
                issues.append(f"exactly one choice must be correct "
                              f"(got {sum(c['correct'] for c in ch)})")
    elif fmt == "assertion_reason":
        for field in ("assertion", "reason"):
            if not (obj.get(field) or "").strip():
                issues.append(f"assertion_reason needs a non-empty {field}")
        for field in ("assertion_true", "reason_true", "reason_explains"):
            if not isinstance(obj.get(field), bool):
                issues.append(f"{field} must be a boolean")
        if (obj.get("assertion_true") is False and obj.get("reason_true") is False):
            # Standard UPSC A-R offers no "both false" option, so such a draft
            # has no correct answer and must be regenerated, not rendered.
            issues.append("assertion and reason cannot both be false")
        if obj.get("reason_explains") and not (obj.get("assertion_true")
                                               and obj.get("reason_true")):
            issues.append("reason_explains requires both assertion and reason true")
    elif fmt == "match_the_following":
        pairs = obj.get("pairs")
        if not isinstance(pairs, list) or not (3 <= len(pairs) <= 4):
            issues.append("match_the_following needs 3-4 pairs")
        else:
            for i, pr in enumerate(pairs):
                if not isinstance(pr, dict) or not (pr.get("left") or "").strip() \
                        or not (pr.get("right") or "").strip():
                    issues.append(f"pair {i+1} needs a left and a right item")
            lefts = [str((pr or {}).get("left", "")).strip().lower() for pr in pairs
                     if isinstance(pr, dict)]
            rights = [str((pr or {}).get("right", "")).strip().lower() for pr in pairs
                      if isinstance(pr, dict)]
            if len(set(lefts)) != len(lefts):
                issues.append("duplicate items in List-I")
            if len(set(rights)) != len(rights):
                issues.append("duplicate items in List-II")
    else:
        issues.append(f"unknown format {fmt!r}")
    if fmt != "assertion_reason" and not (obj.get("stem") or "").strip():
        issues.append("empty stem")
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# multi_statement
# ─────────────────────────────────────────────────────────────────────────────
def _pick_subset_distractors(true_set: frozenset, n: int,
                             rng: random.Random) -> list[frozenset]:
    universe = list(range(1, n + 1))
    candidates = []
    for mask in range(2 ** n):
        s = frozenset(x for i, x in enumerate(universe) if mask >> i & 1)
        if s != true_set:
            candidates.append(s)
    # nearest-first: near-miss codes make real distractors
    by_dist: dict[int, list[frozenset]] = {}
    for s in candidates:
        by_dist.setdefault(len(s ^ true_set), []).append(s)
    picked = []
    for d in sorted(by_dist):
        group = sorted(by_dist[d], key=lambda s: tuple(sorted(s)))
        rng.shuffle(group)
        for s in group:
            if len(picked) < 3:
                picked.append(s)
    return picked


def render_multi_statement(obj: dict, seed=None, variant: str = "auto") -> dict:
    issues = validate(obj)
    if issues:
        raise ValueError("; ".join(issues))
    rng = _rng(seed)
    statements = [s["text"].strip() for s in obj["statements"]]
    n = len(statements)
    true_set = frozenset(i + 1 for i, s in enumerate(obj["statements"]) if s["truth"])
    stem = obj["stem"].strip()
    stem_is_question = "?" in stem

    # ── variant ──────────────────────────────────────────────────────────────
    if variant == "auto":
        if stem_is_question or n == 2:
            variant = "subset"
        elif not true_set:
            variant = "how_many"          # "None" is the natural all-false key
        elif len(true_set) == n:
            variant = "how_many" if rng.random() < 0.5 else "subset"
        else:
            variant = "how_many" if rng.random() < 0.35 else "subset"
    if variant == "how_many" and (n == 2 or stem_is_question):
        variant = "subset"                # n=2 has too few counts; "?" reads odd

    # ── options ─────────────────────────────────────────────────────────────
    if variant == "subset":
        correct_code = subset_code(true_set, n)
        codes = [correct_code] + [subset_code(s, n)
                                  for s in _pick_subset_distractors(true_set, n, rng)]
        lead_out = ("Select the correct answer using the code given below."
                    if stem_is_question else
                    "Which of the statements given above is/are correct?")
    else:
        k = len(true_set)
        correct_code = how_many_code(k, n)
        others = sorted((x for x in range(n + 1) if x != k),
                        key=lambda x: (abs(x - k), x))
        grouped: dict[int, list[int]] = {}
        for x in others:
            grouped.setdefault(abs(x - k), []).append(x)
        picked: list[int] = []
        for d in sorted(grouped):
            g = grouped[d]
            rng.shuffle(g)
            for x in g:
                if len(picked) < 3:
                    picked.append(x)
        codes = [correct_code] + [how_many_code(x, n) for x in picked]
        lead_out = "How many of the above statements are correct?"

    assert len(set(codes)) == len(codes) == 4, f"non-unique codes: {codes}"
    rng.shuffle(codes)
    answer_letter = LETTERS[codes.index(correct_code)]

    options = {LETTERS[i]: c for i, c in enumerate(codes)}
    body = "\n".join(f"{i+1}. {t}" for i, t in enumerate(statements))
    display = (f"{stem}\n{body}\n{lead_out}\n"
               + "\n".join(f"({l}) {c}" for l, c in options.items()))
    return {
        "format": "multi_statement",
        "variant": variant,
        "stem": stem,
        "statements": statements,
        "true_set": sorted(true_set),
        "lead_out": lead_out,
        "options": options,
        "answer": answer_letter,
        "answer_text": correct_code,
        "display": display,
    }


# ─────────────────────────────────────────────────────────────────────────────
# select_identify
# ─────────────────────────────────────────────────────────────────────────────
def render_select_identify(obj: dict, seed=None) -> dict:
    issues = validate(obj)
    if issues:
        raise ValueError("; ".join(issues))
    rng = _rng(seed)
    choices = [{"text": c["text"].strip(), "correct": bool(c["correct"])}
               for c in obj["choices"]]
    rng.shuffle(choices)
    idx = next(i for i, c in enumerate(choices) if c["correct"])
    options = {LETTERS[i]: c["text"] for i, c in enumerate(choices)}
    answer_letter = LETTERS[idx]
    stem = obj["stem"].strip()
    display = (f"{stem}\n"
               + "\n".join(f"({l}) {t}" for l, t in options.items()))
    return {
        "format": "select_identify",
        "stem": stem,
        "options": options,
        "answer": answer_letter,
        "answer_text": options[answer_letter],
        "display": display,
    }


# ─────────────────────────────────────────────────────────────────────────────
# assertion_reason
#
# The four options are FIXED and never shuffled — that is the real UPSC form,
# and candidates read them positionally. The model supplies three booleans and
# the renderer computes which option they select, so an A-R key cannot be
# mis-stated by the model any more than a multi_statement key can.
# ─────────────────────────────────────────────────────────────────────────────
AR_OPTIONS = [
    "Both A and R are true and R is the correct explanation of A",
    "Both A and R are true but R is NOT the correct explanation of A",
    "A is true but R is false",
    "A is false but R is true",
]


def render_assertion_reason(obj: dict, seed=None) -> dict:
    issues = validate(obj)
    if issues:
        raise ValueError("; ".join(issues))
    a_true = bool(obj["assertion_true"])
    r_true = bool(obj["reason_true"])
    explains = bool(obj["reason_explains"])
    if a_true and r_true:
        idx = 0 if explains else 1
    elif a_true and not r_true:
        idx = 2
    else:
        idx = 3
    assertion = obj["assertion"].strip()
    reason = obj["reason"].strip()
    options = {LETTERS[i]: t for i, t in enumerate(AR_OPTIONS)}
    display = (f"Assertion (A): {assertion}\n"
               f"Reason (R): {reason}\n"
               "Select the correct answer using the code given below.\n"
               + "\n".join(f"({l}) {t}" for l, t in options.items()))
    return {
        "format": "assertion_reason",
        "stem": f"Assertion (A): {assertion} Reason (R): {reason}",
        "assertion": assertion,
        "reason": reason,
        "truths": {"assertion": a_true, "reason": r_true, "explains": explains},
        "options": options,
        "answer": LETTERS[idx],
        "answer_text": AR_OPTIONS[idx],
        "display": display,
    }


# ─────────────────────────────────────────────────────────────────────────────
# match_the_following
#
# The model supplies correct left->right pairs only. The renderer shuffles
# List-II, builds the answer code, and derives three near-miss codes by
# transposing exactly two assignments — so every distractor differs from the
# key in the minimum possible way, the same near-miss principle the
# multi_statement distractor codes use.
# ─────────────────────────────────────────────────────────────────────────────
def _pair_code(mapping: list[int]) -> str:
    """mapping[i] = the List-II number assigned to List-I item i (1-based)."""
    return ", ".join(f"{LETTERS[i].upper()}-{mapping[i]}" for i in range(len(mapping)))


def render_match_the_following(obj: dict, seed=None) -> dict:
    issues = validate(obj)
    if issues:
        raise ValueError("; ".join(issues))
    rng = _rng(seed)
    pairs = [{"left": str(p["left"]).strip(), "right": str(p["right"]).strip()}
             for p in obj["pairs"]]
    n = len(pairs)

    order = list(range(n))          # position in List-II -> index of the pair
    rng.shuffle(order)
    right_items = [pairs[i]["right"] for i in order]
    # correct mapping: for List-I item i, the 1-based slot its right item sits in
    slot_of = {pair_idx: pos + 1 for pos, pair_idx in enumerate(order)}
    correct = [slot_of[i] for i in range(n)]

    codes, seen = [correct], {tuple(correct)}
    attempts = 0
    while len(codes) < 4 and attempts < 200:
        attempts += 1
        i, j = rng.sample(range(n), 2)
        base = list(rng.choice(codes))
        base[i], base[j] = base[j], base[i]
        if tuple(base) not in seen:
            seen.add(tuple(base))
            codes.append(base)
    while len(codes) < 4:                       # tiny n fallback: any permutation
        perm = list(correct)
        rng.shuffle(perm)
        if tuple(perm) not in seen:
            seen.add(tuple(perm))
            codes.append(perm)

    code_strings = [_pair_code(c) for c in codes]
    correct_string = code_strings[0]
    rng.shuffle(code_strings)
    answer_letter = LETTERS[code_strings.index(correct_string)]
    options = {LETTERS[i]: c for i, c in enumerate(code_strings)}

    stem = (obj.get("stem") or "Match List-I with List-II and select the correct "
                              "answer using the code given below.").strip()
    body = ("List-I\n"
            + "\n".join(f"{LETTERS[i].upper()}. {pairs[i]['left']}" for i in range(n))
            + "\nList-II\n"
            + "\n".join(f"{i+1}. {right_items[i]}" for i in range(n)))
    display = (f"{stem}\n{body}\nSelect the correct answer using the code given below.\n"
               + "\n".join(f"({l}) {c}" for l, c in options.items()))
    return {
        "format": "match_the_following",
        "stem": stem,
        "list_i": [p["left"] for p in pairs],
        "list_ii": right_items,
        "correct_mapping": correct,
        "options": options,
        "answer": answer_letter,
        "answer_text": correct_string,
        "display": display,
    }


def render(obj: dict, seed=None, variant: str = "auto") -> dict:
    if obj.get("format") == "multi_statement":
        return render_multi_statement(obj, seed, variant)
    if obj.get("format") == "select_identify":
        return render_select_identify(obj, seed)
    if obj.get("format") == "assertion_reason":
        return render_assertion_reason(obj, seed)
    if obj.get("format") == "match_the_following":
        return render_match_the_following(obj, seed)
    raise ValueError(f"unknown format {obj.get('format')!r}")


# ─────────────────────────────────────────────────────────────────────────────
# exhaustive self-test
# ─────────────────────────────────────────────────────────────────────────────
def _selftest() -> None:
    checks = 0

    def ms(n, true_set, stem="Consider the following statements regarding X:"):
        return {"format": "multi_statement", "stem": stem,
                "statements": [{"text": f"Statement number {i}",
                                "truth": i in true_set}
                               for i in range(1, n + 1)]}

    for n in range(2, 6):
        for mask in range(2 ** n):
            T = frozenset(i + 1 for i in range(n) if mask >> i & 1)
            for seed in range(4):
                for variant in ("auto", "subset", "how_many"):
                    for stem in ("Consider the following statements:",
                                 "Which of the following is/are audited by the CAG?"):
                        r = render_multi_statement(ms(n, T, stem), seed, variant)
                        # exactly one option is the code for T
                        want = (subset_code(T, n) if r["variant"] == "subset"
                                else how_many_code(len(T), n))
                        hits = [l for l, c in r["options"].items() if c == want]
                        assert hits == [r["answer"]], (n, T, variant, r)
                        assert len(set(r["options"].values())) == 4
                        assert r["answer_text"] == want
                        assert r["true_set"] == sorted(T)
                        # determinism
                        r2 = render_multi_statement(ms(n, T, stem), seed, variant)
                        assert r == r2
                        # a "?" stem never gets the how_many lead-out
                        if "?" in stem:
                            assert r["variant"] == "subset"
                        checks += 1

    # all-false and all-true land on renderable keys through the auto path
    r = render_multi_statement(ms(3, frozenset()), seed=1)
    assert r["answer_text"] == "None" and r["variant"] == "how_many"
    r = render_multi_statement(ms(2, frozenset({1, 2})), seed=1)
    assert r["answer_text"] == "Both 1 and 2"
    checks += 2

    # select_identify: correct choice survives the shuffle from any position
    for k in (3, 4, 5):
        for pos in range(k):
            for seed in range(6):
                obj = {"format": "select_identify", "stem": "Which body is X?",
                       "choices": [{"text": f"Choice {i}", "correct": i == pos}
                                   for i in range(k)]}
                r = render_select_identify(obj, seed)
                assert r["options"][r["answer"]] == f"Choice {pos}"
                assert len(r["options"]) == k
                assert render_select_identify(obj, seed) == r
                checks += 1

    # invalid inputs are refused, not mangled
    bad = [
        {"format": "select_identify", "stem": "s",
         "choices": [{"text": "a", "correct": True}, {"text": "b", "correct": True},
                     {"text": "c", "correct": False}]},          # two correct
        {"format": "select_identify", "stem": "s",
         "choices": [{"text": "a", "correct": False}, {"text": "b", "correct": False},
                     {"text": "c", "correct": False}]},          # none correct
        {"format": "multi_statement", "stem": "s",
         "statements": [{"text": "only one", "truth": True}]},   # 1 statement
        {"format": "multi_statement", "stem": "s",
         "statements": [{"text": "a", "truth": True},
                        {"text": "b", "truth": None}]},          # non-bool truth
        {"format": "match_the_following", "stem": "s"},          # unknown format
    ]
    for obj in bad:
        try:
            render(obj, seed=0)
        except ValueError:
            checks += 1
        else:
            raise AssertionError(f"accepted invalid input: {obj}")

    demo = render_multi_statement(
        {"format": "multi_statement",
         "stem": "Consider the following statements regarding the Finance Commission:",
         "statements": [
             {"text": "It is constituted by the President every fifth year", "truth": True},
             {"text": "Its recommendations are binding on the Government of India", "truth": False},
             {"text": "Article 280 provides for its composition", "truth": True}]},
        seed="demo")
    # ── assertion_reason: every truth combination maps to the right option ──
    AR_CASES = [
        (True, True, True, "a"), (True, True, False, "b"),
        (True, False, False, "c"), (False, True, False, "d"),
    ]
    for a, r, ex, want in AR_CASES:
        d = render({"format": "assertion_reason",
                    "assertion": "Parliament can legislate on a State List subject during a National Emergency",
                    "reason": "A National Emergency suspends the distribution of legislative powers",
                    "assertion_true": a, "reason_true": r, "reason_explains": ex}, seed="t")
        assert d["answer"] == want, (a, r, ex, d["answer"])
        assert d["answer_text"] == AR_OPTIONS["abcd".index(want)]
        assert len(d["options"]) == 4 and "Assertion (A):" in d["display"]
        checks += 1
    assert validate({"format": "assertion_reason", "assertion": "x", "reason": "y",
                     "assertion_true": False, "reason_true": False,
                     "reason_explains": False})            # both-false rejected
    assert validate({"format": "assertion_reason", "assertion": "x", "reason": "y",
                     "assertion_true": True, "reason_true": False,
                     "reason_explains": True})             # explains w/o both true
    checks += 2

    # ── match_the_following: key is always the true mapping, 4 unique codes ──
    for n in (3, 4):
        for seed in range(6):
            pairs = [{"left": f"Item {chr(65+i)}", "right": f"Desc {i+1}"} for i in range(n)]
            d = render({"format": "match_the_following", "stem": "Match List-I with List-II:",
                        "pairs": pairs}, seed=seed)
            assert len(d["options"]) == 4 and len(set(d["options"].values())) == 4
            key = d["options"][d["answer"]]
            # decode the key and confirm every left maps to its own right
            for part in key.split(","):
                lab, num = part.strip().split("-")
                li = "ABCD".index(lab)
                assert d["list_ii"][int(num) - 1] == pairs[li]["right"], (key, seed)
            assert d["answer_text"] == key
            checks += 1
    assert validate({"format": "match_the_following", "stem": "s",
                     "pairs": [{"left": "A", "right": "1"}, {"left": "A", "right": "2"},
                               {"left": "C", "right": "3"}]})        # duplicate lefts
    assert validate({"format": "match_the_following", "stem": "s",
                     "pairs": [{"left": "A", "right": "1"}]})        # too few pairs
    checks += 2

    print(f"self-test: {checks} checks passed")
    print("\n" + demo["display"] + f"\n\nAnswer: ({demo['answer']})")


if __name__ == "__main__":
    _selftest()
