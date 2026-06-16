#!/usr/bin/env python3
"""Convert Big-Math (HF: 'SynthLabsAI/Big-Math-RL-Verified') → {problem, gold} for verify_math.

Big-Math is the cleanest large RL pool: human-filtered, deduped, single-numeric-answer.
Source schema (per row):
  {"problem": "...", "answer": "42", "source": "...", "domain": "..."}
The `answer` is already extracted, so this converter is essentially a rename.
"""
import argparse, json, sys


def convert_row(row: dict) -> dict | None:
    problem = (row.get("problem") or "").strip()
    ans = row.get("answer")
    if not problem or ans is None:
        return None
    return {"problem": problem, "gold": str(ans).strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split=args.split)
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
    assert convert_row({"problem": "Q?", "answer": 42}) == {"problem": "Q?", "gold": "42"}
    assert convert_row({"problem": "Q?", "answer": "3.14"}) == {"problem": "Q?", "gold": "3.14"}
    assert convert_row({"answer": "42"}) is None       # missing problem
    assert convert_row({"problem": "Q?"}) is None       # missing answer
    print("PASS big-math converter")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
