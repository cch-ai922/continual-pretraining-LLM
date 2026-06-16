#!/usr/bin/env python3
"""Convert MMLU (HF: 'cais/mmlu') → {problem, gold} for verify_mcq.

Source schema (per row):
  {"question": "...", "subject": "...", "choices": ["c0", "c1", "c2", "c3"],
   "answer": 2}                          # 0-indexed integer
We render the question with labeled options and emit gold as the letter A/B/C/D.
"""
import argparse, json, string, sys


def convert_row(row: dict) -> dict | None:
    q = (row.get("question") or "").strip()
    choices = row.get("choices") or []
    ans = row.get("answer")
    if not q or not choices or ans is None:
        return None
    if not (0 <= int(ans) < len(choices) <= 26):
        return None
    letters = string.ascii_uppercase[: len(choices)]
    rendered = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices))
    return {"problem": f"{q}\n\n{rendered}",
            "gold": letters[int(ans)],
            "n_options": len(choices)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", default="all",
                    help="subject subset (e.g. 'all', 'high_school_mathematics')")
    ap.add_argument("--split", default="test", choices=["test", "validation", "dev", "auxiliary_train"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", args.subset, split=args.split)
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
    row = {"question": "Capital of France?",
           "choices": ["Berlin", "Paris", "Rome", "Madrid"],
           "answer": 1}
    r = convert_row(row)
    assert r["gold"] == "B"
    assert r["n_options"] == 4
    assert "A) Berlin" in r["problem"] and "B) Paris" in r["problem"]
    # 5-option case
    r5 = convert_row({"question": "x", "choices": ["a", "b", "c", "d", "e"], "answer": 4})
    assert r5["gold"] == "E" and r5["n_options"] == 5
    # malformed: skip
    assert convert_row({"question": "x", "choices": [], "answer": 0}) is None
    assert convert_row({"question": "x", "choices": ["a", "b"], "answer": 5}) is None

    # round-trip through verify_mcq
    import os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from verify_mcq import verify
    assert verify("Answer: B", r["gold"], r["n_options"])
    assert not verify("Answer: A", r["gold"], r["n_options"])
    print("PASS mmlu converter")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
