#!/usr/bin/env python3
"""
Stage 8 (verifier) — multiple-choice answer extraction (MMLU/KMMLU/ARC-style).

Handles English (A-Z), Korean syllable labels (가/나/다/라/마 = A/B/C/D/E),
circled numbers (①②③④⑤) and 1-based digits (1-5), in many common formats:

  Answer: B          /  정답: ②
  The answer is C    /  최종답: 다
  (D)                /  답은 4번
  \\boxed{B}          /  보기: 1
  Last standalone label as a final fallback.

Extraction picks the LAST matching marker — models often say "I considered A but…
the answer is C", and we want C, not A.
"""
import argparse, json, re, string, sys


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
# Korean MCQ labels -> A..E. Korean exams use circled ①..⑤, the syllables
# 가/나/다/라/마, or 1-based digits 1..5 (often followed by 번).
_KO_TO_EN = {
    "①": "A", "②": "B", "③": "C", "④": "D", "⑤": "E",
    "가": "A", "나": "B", "다": "C", "라": "D", "마": "E",
    "1": "A", "2": "B", "3": "C", "4": "D", "5": "E",     # 1-based digit options
}
_LETTER = r"[A-Za-z①-⑤가-마1-5]"               # marker-context label (digits allowed)
_FB_LETTER = r"[A-Za-z①-⑤가-마]"               # fallback label (no bare digits -> less noise)

# Ordered priority: stronger markers first. Each captures one label; an optional
# trailing 번 / . / ) is consumed (e.g. "4번", "(B)", "C.").
_PATTERNS = [
    re.compile(rf"\\boxed\{{\s*({_LETTER})\s*\}}"),
    re.compile(rf"(?:최종\s*답|정답|답|보기|선택지)\s*(?:은|는|이|가)?\s*[:：]?\s*\(?\s*({_LETTER})\s*(?:번|\.|\))?", re.U),
    re.compile(rf"(?:the\s+answer\s+is|answer\s+is|answer|option)\s*[:：]?\s*\(?\s*({_LETTER})\s*[\.\)]?", re.I),
    re.compile(rf"\(\s*({_LETTER})\s*\)"),                     # (B) style
]


def _normalize(letter: str) -> str:
    return _KO_TO_EN.get(letter, letter.upper())


def extract_choice(text: str, n_options: int = 4):
    """Return the chosen letter as uppercase English (A..Z) or None.

    n_options bounds the valid letters; defaults to 4 (A-D).
    """
    if not text or n_options < 1 or n_options > 26:
        return None
    valid = set(string.ascii_uppercase[:n_options])
    for pat in _PATTERNS:
        matches = list(pat.finditer(text))
        if matches:
            cand = _normalize(matches[-1].group(1))
            if cand in valid:
                return cand
    # last-resort: last standalone label (circled / Korean syllable / Latin) in the text
    for m in reversed(list(re.finditer(rf"(?<![A-Za-z가-힣])({_FB_LETTER})(?![A-Za-z가-힣])", text))):
        cand = _normalize(m.group(1))
        if cand in valid:
            return cand
    return None


def verify(model_output: str, gold, n_options: int = 4) -> bool:
    """Plug into rejection_sample.py / grpo_train.py.

    `gold` may be: a label ("A".."E" / "가".."마" / "①".."⑤" / 1-based "1".."5"),
    OR a dict {"answer": "B", "n_options": 5} to override option count.
    """
    if isinstance(gold, dict):
        n = int(gold.get("n_options", n_options))
        g = str(gold.get("answer", gold.get("gold", ""))).strip()
    else:
        n, g = n_options, str(gold).strip()
    if not g:
        return False
    gold_letter = _normalize(g[0])  # tolerate trailing punctuation: "B." -> "B"
    extracted = extract_choice(model_output, n)
    return extracted is not None and extracted == gold_letter


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="JSONL with {prediction/response, answer/gold, [n_options]}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-options", type=int, default=4)
    args = ap.parse_args()

    n = ok = 0
    fout = open(args.out, "w", encoding="utf-8") if args.out else None
    for line in open(args.inp, encoding="utf-8"):
        r = json.loads(line); n += 1
        pred = r.get("prediction") or r.get("response") or r.get("output", "")
        gold = r.get("gold", r.get("answer", ""))
        v = verify(pred, gold, args.n_options)
        ok += int(v)
        if fout:
            fout.write(json.dumps({**r, "verified": v,
                                   "extracted": extract_choice(pred, args.n_options)},
                                  ensure_ascii=False) + "\n")
    if fout:
        fout.close()
    print(f"verified {ok:,}/{n:,}  ({ok/max(n,1):.1%})")


# ---------------------------------------------------------------------------
def _selftest():
    # English markers, varied phrasings
    assert extract_choice("Answer: B") == "B"
    assert extract_choice("The answer is C.") == "C"
    assert extract_choice("So the correct option is (D).") == "D"
    assert extract_choice("After analysis, \\boxed{A}") == "A"
    # Korean markers + label styles
    assert extract_choice("정답: ②") == "B"
    assert extract_choice("최종답: 다") == "C"
    assert extract_choice("답은 4번") == "D"
    assert extract_choice("보기: 1") == "A"                  # 1-based digit -> A
    assert extract_choice("정답은 ⑤입니다", n_options=5) == "E"
    assert extract_choice("나 가 맞다") == "A"               # standalone Korean-label fallback (last valid... )
    # Last-match wins (model considers wrong answer first, settles on right one)
    assert extract_choice("I thought A but actually the answer is C.") == "C"
    # No marker, last standalone letter
    assert extract_choice("Looking at it... B.") == "B"
    # Lowercase letters are allowed (normalized to uppercase)
    assert extract_choice("answer is c") == "C"
    # Out-of-range label rejected
    assert extract_choice("Answer: E", n_options=4) is None
    assert extract_choice("Answer: E", n_options=5) == "E"
    # No label at all
    assert extract_choice("모르겠습니다") is None
    # Empty / nonsense
    assert extract_choice("") is None

    # end-to-end verify
    assert verify("정답: ②", "B")
    assert verify("정답: 나", "나")            # gold can be a Korean label too
    assert verify("답은 4번", {"answer": "D", "n_options": 4})
    assert verify("The answer is D.", "D")
    assert not verify("Answer: A", "B")
    # 1-based digit gold (KMMLU style): gold "3" == option C
    assert verify("정답: 다", "3")
    # dict gold with n_options
    assert verify("정답: ⑤", {"answer": "E", "n_options": 5})
    # gold with trailing punctuation
    assert verify("Answer: B", "B.")

    print("PASS all mcq-verifier tests")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
