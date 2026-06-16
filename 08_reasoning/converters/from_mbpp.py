#!/usr/bin/env python3
"""Convert MBPP (HF: 'mbpp', config 'sanitized') → {problem, gold} for verify_code.

Source schema (per row):
  {"task_id": int, "text": "Write a function to find ...",
   "code": "def f(...): ...", "test_list": ["assert f(1) == 2", ...],
   "test_setup_code": "import math", "challenge_test_list": [...]}
"""
import argparse, json, sys


def convert_row(row: dict) -> dict | None:
    text = (row.get("text") or row.get("prompt") or "").strip()
    test_list = row.get("test_list") or []
    if not text or not test_list:
        return None
    setup = row.get("test_setup_code") or ""
    tests = (setup + "\n" if setup else "") + "\n".join(test_list)
    problem = f"{text}\n\n다음은 실례 테스트입니다:\n```python\n{test_list[0]}\n```"
    return {"problem": problem, "gold": {"tests": tests, "timeout": 5.0}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--config", default="sanitized", choices=["full", "sanitized"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("mbpp", args.config, split=args.split)
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
    row = {"text": "Write a function to add two numbers.",
           "code": "def add(a,b): return a+b",
           "test_list": ["assert add(1,2) == 3", "assert add(0,0) == 0"]}
    r = convert_row(row)
    assert "Write a function to add" in r["problem"]
    assert "assert add(1,2) == 3" in r["gold"]["tests"]
    assert "assert add(0,0) == 0" in r["gold"]["tests"]
    # with setup code
    r2 = convert_row({"text": "Math thing.", "test_list": ["assert sqrt(4) == 2"],
                      "test_setup_code": "from math import sqrt"})
    assert r2["gold"]["tests"].startswith("from math import sqrt")
    # missing tests -> skip
    assert convert_row({"text": "x"}) is None

    # round-trip
    import os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from verify_code import verify
    output = "```python\ndef add(a, b):\n    return a + b\n```"
    assert verify(output, r["gold"])
    print("PASS mbpp converter")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
