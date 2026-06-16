#!/usr/bin/env python3
"""Convert KMMLU (HF: 'HAERAE-HUB/KMMLU') → {problem, gold, n_options} for verify_mcq.

KMMLU is the native-Korean analogue of MMLU — 45 subjects of Korean-context,
expert-level MCQ (not translated MMLU). The highest-value Korean MCQ source.

Source schema (per row):
  {"question": "...", "A": "...", "B": "...", "C": "...", "D": "...",
   "answer": 2, "Category": "..."}     # answer is 1-BASED (1..4)

Also accepts a generic options-list layout ('options'/'choices') and letter answers,
so it works on KMMLU variants and KMMLU-Pro-style dumps. Tweak FIELD_* if needed.
"""
import argparse, json, string, sys


FIELD_QUESTION = ("question", "prompt", "query")
FIELD_OPTIONS  = ("options", "choices", "option_list")
FIELD_ANSWER   = ("answer", "correct_answer", "label", "target", "gold")
OPTION_COLS    = ("A", "B", "C", "D", "E")        # KMMLU's per-column option layout


def _pick(row, keys):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _options(row):
    """Return the option list, from either an options list or A/B/C/D columns."""
    opts = _pick(row, FIELD_OPTIONS)
    if opts:
        return list(opts)
    cols = [row[c] for c in OPTION_COLS if c in row and row[c] not in (None, "")]
    return cols or None


def convert_row(row: dict) -> dict | None:
    q = _pick(row, FIELD_QUESTION)
    opts = _options(row)
    raw = _pick(row, FIELD_ANSWER)
    if not q or not opts or raw is None:
        return None
    if not (1 < len(opts) <= 26):
        return None
    letters = string.ascii_uppercase[: len(opts)]
    # gold may be a 1-based index (KMMLU), a 0-based index, or a letter
    if isinstance(raw, int) or (isinstance(raw, str) and str(raw).strip().isdigit()):
        idx = int(raw)
        if 1 <= idx <= len(opts):                 # KMMLU is 1-based
            gold = letters[idx - 1]
        elif idx == 0:                            # tolerate a 0-based dump
            gold = letters[0]
        else:
            return None
    else:
        s = str(raw).strip()
        ko = {"가": "A", "나": "B", "다": "C", "라": "D", "마": "E"}
        gold = ko.get(s[:1], s[:1].upper())
        if gold not in letters:
            return None
    rendered = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(opts))
    return {"problem": f"{str(q).strip()}\n\n{rendered}",
            "gold": gold, "n_options": len(opts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", default="all", help="KMMLU subject/config (e.g. 'all')")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("HAERAE-HUB/KMMLU", args.subset, split=args.split)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            if args.limit and n >= args.limit:
                break
            r = convert_row(row)
            if r is None:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    print(f"wrote {n:,} rows -> {args.out}")


def _selftest():
    # KMMLU layout: A/B/C/D columns + 1-based integer answer
    r = convert_row({"question": "조선민주주의인민공화국의 수도는?",
                     "A": "개성", "B": "평양", "C": "평성", "D": "함흥",
                     "answer": 2})
    assert r["gold"] == "B" and "평양" in r["problem"] and r["n_options"] == 4
    assert r["problem"].count(")") >= 4            # rendered A) B) C) D)
    # options-list layout + 1-based answer
    r2 = convert_row({"question": "x", "options": ["가1", "가2", "가3", "가4"], "answer": 3})
    assert r2["gold"] == "C"
    # Korean letter answer (가-마 -> A-E)
    r3 = convert_row({"question": "x", "choices": ["o1", "o2", "o3", "o4"], "answer": "나"})
    assert r3["gold"] == "B"
    # missing fields -> skip
    assert convert_row({"question": "x"}) is None
    # out-of-range 1-based index -> skip
    assert convert_row({"question": "x", "A": "a", "B": "b", "answer": 9}) is None

    # round-trip through verify_mcq with a Korean answer marker
    import os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from verify_mcq import verify
    assert verify("정답: ②", r["gold"], r["n_options"])
    assert not verify("정답: ①", r["gold"], r["n_options"])
    print("PASS kmmlu converter (incl. round-trip through verify_mcq)")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
