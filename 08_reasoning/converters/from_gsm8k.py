#!/usr/bin/env python3
"""Convert GSM8K (HF: 'gsm8k', config 'main') → {problem, gold} for verify_math.

Source schema (per row):
  {"question": "Natalia sold clips ...", "answer": "Natalia sold ...\\n#### 72"}
"""
import argparse, json, re, sys


def convert_row(row: dict) -> dict | None:
    q, a = row.get("question"), row.get("answer", "")
    if not q or "####" not in a:
        return None
    gold = a.split("####")[-1].strip().replace(",", "")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", gold):
        return None
    return {"problem": q.strip(), "gold": gold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=args.split)
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
    r = convert_row({"question": "If 2+2=?", "answer": "2 plus 2 is 4.\n#### 4"})
    assert r == {"problem": "If 2+2=?", "gold": "4"}
    # commas in answer get stripped
    r = convert_row({"question": "Big?", "answer": "...\n#### 1,200"})
    assert r["gold"] == "1200"
    # missing #### marker -> skip
    assert convert_row({"question": "x", "answer": "no marker"}) is None
    # non-numeric gold -> skip
    assert convert_row({"question": "x", "answer": "...\n#### NaN"}) is None
    print("PASS gsm8k converter")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
