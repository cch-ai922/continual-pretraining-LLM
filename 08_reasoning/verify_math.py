#!/usr/bin/env python3
"""
Stage 8 (verifier) — language-INDEPENDENT correctness check for math problems.

This is the heart of RLVR for reasoning: it doesn't matter what language the chain
of thought is in; a number is right or wrong by arithmetic, not by judgment. The
verifier extracts the final answer from the model's output and compares it to the
gold answer numerically.

Extraction priority (catches both Korean and English formats):
  1. \\boxed{X}                              (LaTeX standard)
  2. '최종답: X' / '정답: X' / '답: X'         (Korean answer markers)
  3. 'the answer is X' / 'answer: X'         (English fallback)
  4. '#### X'                                 (GSM8K convention)
  5. last number in the response             (last-resort fallback)

Numeric equality handles ints, floats, fractions, and tolerates trailing units
like '원' / 'meters'. Pure logic, fully unit-tested (--selftest).
"""
import argparse, json, re
from fractions import Fraction


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
_NUM = r"-?\d+(?:[,\d]*\d)?(?:\.\d+)?(?:/-?\d+)?"     # int/float/fraction, with commas
_PATTERNS = [
    re.compile(r"\\boxed\{\s*([^}]+?)\s*\}"),
    re.compile(rf"최종\s*답\s*[:：]\s*({_NUM})"),
    re.compile(rf"(?:정답|답)\s*[:：]\s*({_NUM})"),
    re.compile(rf"(?:the\s+)?answer\s+is\s*[:：]?\s*({_NUM})", re.I),
    re.compile(rf"answer\s*[:：]\s*({_NUM})", re.I),
    re.compile(rf"####\s*({_NUM})"),
]


def extract_final_answer(text: str):
    """Return the last successful match across all priority patterns, or last number."""
    last_match = None
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            last_match = m.group(1).strip()
    if last_match is not None:
        return last_match
    # last-number fallback (RHS of the last `=`, else the very last number)
    eq = list(re.finditer(rf"=\s*({_NUM})", text))
    if eq:
        return eq[-1].group(1).strip()
    nums = list(re.finditer(_NUM, text))
    return nums[-1].group(0).strip() if nums else None


def _to_number(s):
    """Parse 's' into a Fraction so int/float/fraction can be compared exactly."""
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace(" ", "")
    s = re.sub(r"[^\d./\-]", "", s)            # strip units like '원' / 'meters'
    if not s or s in ("-", ".", "/"):
        return None
    try:
        if "/" in s:
            return Fraction(s)
        if "." in s:
            return Fraction(s).limit_denominator(10**9)
        return Fraction(int(s))
    except Exception:
        return None


def numeric_equal(a, b, tol=1e-4):
    """Compare two values as numbers with tolerance. Handles ints/floats/fractions."""
    fa, fb = _to_number(a), _to_number(b)
    if fa is None or fb is None:
        return False
    return abs(float(fa) - float(fb)) <= tol


def verify(model_output: str, gold_answer) -> bool:
    return numeric_equal(extract_final_answer(model_output), gold_answer)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="JSONL with {prediction, answer} or {response, answer}")
    ap.add_argument("--out", default=None, help="optional: write per-row verdicts JSONL")
    args = ap.parse_args()

    n = ok = 0
    fout = open(args.out, "w", encoding="utf-8") if args.out else None
    for line in open(args.inp, encoding="utf-8"):
        r = json.loads(line); n += 1
        pred = r.get("prediction") or r.get("response") or r.get("output", "")
        gold = r.get("answer") or r.get("gold")
        v = verify(pred, gold)
        ok += int(v)
        if fout:
            fout.write(json.dumps({**r, "verified": v,
                                   "extracted": extract_final_answer(pred)},
                                  ensure_ascii=False) + "\n")
    if fout:
        fout.close()
    print(f"verified {ok:,}/{n:,}  ({ok/max(n,1):.1%})")


# ---------------------------------------------------------------------------
def _selftest():
    # extraction priority
    assert extract_final_answer("Therefore the answer is $\\boxed{42}$.") == "42"
    assert extract_final_answer("문장입니다. 최종답: 42") == "42"
    assert extract_final_answer("풀이과정. 정답: -7.5") == "-7.5"
    assert extract_final_answer("Step 1 ... #### 18") == "18"
    assert extract_final_answer("So x = 3 + 4 = 7. Done.") == "7"   # via '=' fallback
    assert extract_final_answer("First 2 then 3 finally 11.") == "11"  # last-number fallback
    assert extract_final_answer("수자가 전혀 없습니다") is None
    # multiple matches -> take the LAST (model may show intermediate boxes)
    assert extract_final_answer("\\boxed{12} ... \\boxed{42}") == "42"
    # numeric equality across formats
    assert numeric_equal("42", 42) and numeric_equal("42.0", 42)
    assert numeric_equal("1/2", "0.5") and numeric_equal("3/4", 0.75)
    assert numeric_equal("1,200", "1200")
    assert numeric_equal("42 원", 42)
    assert not numeric_equal("42", "43")
    assert not numeric_equal("abc", "42")
    # end-to-end verify
    assert verify("풀이 ... 최종답: 144", 144)
    assert verify("\\boxed{0.25}", "1/4")
    assert not verify("풀이 ... 최종답: 10", 11)
    print("PASS all math-verifier tests")


if __name__ == "__main__":
    import sys
    _selftest() if "--selftest" in sys.argv else main()
