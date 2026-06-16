#!/usr/bin/env python3
"""Convert HumanEval (HF: 'openai_humaneval') → {problem, gold} for verify_code.

Source schema (per row):
  {"task_id": "HumanEval/0", "prompt": "def has_close_elements(...)\\n    ...",
   "canonical_solution": "...", "test": "def check(candidate): ...",
   "entry_point": "has_close_elements"}

We treat `prompt` (a partial function with docstring) as the spec; the verifier runs
the model's complete function + `test` (which defines `check`) + `check(entry_point)`.
"""
import argparse, json, sys


def convert_row(row: dict) -> dict | None:
    prompt, test, entry = row.get("prompt"), row.get("test"), row.get("entry_point")
    if not (prompt and test and entry):
        return None
    problem = f"Complete the following Python function:\n\n```python\n{prompt}```"
    tests = f"{test}\n\ncheck({entry})"
    return {"problem": problem, "gold": {"tests": tests, "timeout": 5.0}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")          # HumanEval is test-only
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split=args.split)
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
    row = {"prompt": "def add(a, b):\n    \"\"\"Sum.\"\"\"\n",
           "test": "def check(f):\n    assert f(2, 3) == 5",
           "entry_point": "add"}
    r = convert_row(row)
    assert r["problem"].startswith("Complete the following Python function:")
    assert "def add(a, b):" in r["problem"]
    assert "check(add)" in r["gold"]["tests"]
    assert r["gold"]["timeout"] == 5.0
    # missing entry_point -> skip
    assert convert_row({"prompt": "x", "test": "y"}) is None

    # round-trip via verify_code: model output that mimics a real completion should pass
    import os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from verify_code import verify
    model_output = "```python\ndef add(a, b):\n    return a + b\n```"
    assert verify(model_output, r["gold"]), "round-trip with verify_code should pass"
    bad_output = "```python\ndef add(a, b):\n    return a - b\n```"
    assert not verify(bad_output, r["gold"]), "wrong implementation should fail"
    print("PASS humaneval converter (incl. round-trip through verify_code)")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
