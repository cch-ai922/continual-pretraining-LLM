#!/usr/bin/env python3
"""Convert Hendrycks MATH (HF: 'hendrycks/competition_math') → {problem, gold} for verify_math.

Source schema (per row):
  {"problem": "...", "level": "Level 3", "type": "Algebra",
   "solution": "... Therefore the answer is $\\boxed{42}$ ..."}
The gold answer lives inside the LAST \\boxed{...} in `solution` (braces can nest).
"""
import argparse, json, sys


def extract_boxed(s: str) -> str | None:
    """Find the LAST \\boxed{...} with balanced braces."""
    if not s:
        return None
    idx = s.rfind("\\boxed{")
    if idx < 0:
        return None
    start = idx + len("\\boxed{")
    depth, j = 1, start
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[start:j].strip()
        j += 1
    return None


def convert_row(row: dict) -> dict | None:
    problem = (row.get("problem") or "").strip()
    gold = extract_boxed(row.get("solution") or "")
    if not problem or gold is None:
        return None
    return {"problem": problem, "gold": gold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("hendrycks/competition_math", split=args.split)
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
    # plain
    r = convert_row({"problem": "Solve x.", "solution": "Hence $\\boxed{42}$."})
    assert r == {"problem": "Solve x.", "gold": "42"}, r
    # nested braces inside \boxed{...}
    r = convert_row({"problem": "f?", "solution": "$\\boxed{\\frac{1}{2}}$"})
    assert r["gold"] == "\\frac{1}{2}", r
    # LAST \boxed wins
    r = convert_row({"problem": "x", "solution": "tried $\\boxed{7}$ but actually $\\boxed{12}$."})
    assert r["gold"] == "12"
    # no \boxed -> skip
    assert convert_row({"problem": "x", "solution": "no boxed here"}) is None
    print("PASS MATH converter")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
