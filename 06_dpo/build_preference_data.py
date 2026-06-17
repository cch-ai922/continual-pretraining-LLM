#!/usr/bin/env python3
"""
Stage 6 — build Korean DPO preference data WITHOUT native annotators, using cheap
*verifiable* signals that target exactly the failure modes of a freshly-adapted
low-resource model. For each prompt we sample two candidate responses from the SFT
model (e.g. two temperatures) and pick chosen/rejected by these rules, in order:

  1. language correctness : Korean response beats a code-switched/English one
  2. register consistency : single-register Korean (높임말/문체) beats mixed
  3. format adherence     : followed requested format (e.g. JSON/list) beats not
  4. faithfulness         : for grounded prompts, faithful-to-source beats hallucinated
  5. length/health        : non-degenerate (no repetition loops) beats degenerate

Pairs where the two candidates tie on all signals are dropped (no clear signal).
The signal functions below are concrete and unit-tested at the bottom; the
generation step is a hook you connect to vLLM/HF serving of your SFT model.
"""
import argparse, json, os, re, sys, unicodedata

# Sibling-module import (register check lives in 05_sft/). Works whether this
# script is run from the project root or from 06_dpo/ directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "05_sft"))
from register_consistency import is_consistent as _register_consistent


# ---------- verifiable signal functions (pure, testable) ----------
def korean_fraction(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 1.0
    deva = sum(1 for c in letters if "HANGUL" in unicodedata.name(c, ""))
    return deva / len(letters)


def has_repetition(s: str, n: int = 4, thresh: int = 3) -> bool:
    """True if any n-gram repeats >= thresh times (degenerate output)."""
    toks = s.split()
    if len(toks) < n * thresh:
        return False
    grams = {}
    for i in range(len(toks) - n + 1):
        g = tuple(toks[i:i + n])
        grams[g] = grams.get(g, 0) + 1
        if grams[g] >= thresh:
            return True
    return False


def follows_format(s: str, fmt: str | None) -> bool:
    if not fmt:
        return True
    if fmt == "json":
        try:
            json.loads(s.strip()); return True
        except Exception:
            return False
    if fmt == "list":
        return bool(re.search(r"(^|\n)\s*(-|•|\d+[.)])\s+", s))
    return True


_PARTICLES = sorted([
    "입니다", "이였다", "이며", "이고", "이다",   # fused copula endings (noun+이다)
    "에서는", "으로는", "에게서", "께서는", "이라고", "이라는",
    "에서", "에게", "께서", "으로", "라고", "부터", "까지", "보다", "처럼",
    "만큼", "마다", "조차", "마저", "밖에", "이나", "든지", "이라", "에는",
    "은", "는", "이", "가", "을", "를", "의", "에", "도", "와", "과", "만",
    "로", "야", "께", "라", "나",
], key=len, reverse=True)


def _strip_particle(w):
    for p in _PARTICLES:
        if len(w) - len(p) >= 2 and w.endswith(p):   # keep >=2-syll stem (정의/수도 intact)
            return w[: -len(p)]
    return w


_KO_STOP = {"그리고", "그러나", "하지만", "또한", "그래서", "따라서", "즉", "및",
            "등", "것", "수", "더", "매우", "아주", "이", "그", "저", "때문",
            "하다", "있다", "없다", "되다", "이다", "한다", "합니다", "입니다",
            "있습니다", "같은", "위해", "통해", "대한", "대해"}


def _content_words(text: str) -> set:
    out = set()
    for w in text.split():
        w = w.strip("。.,?!\"'()[]{}:;…·")
        w = _strip_particle(w)
        if len(w) >= 2 and korean_fraction(w) > 0.5 and w not in _KO_STOP:
            out.add(w)
    return out


def faithful(s: str, source: str | None) -> bool:
    """Cheap grounding proxy synced with 05_sft: punctuation normalized and common
    Korean function words/particles dropped, so endings like '은/는' don't inflate overlap."""
    if not source:
        return True
    src = _content_words(source)
    if not src:
        return True
    return len(src & _content_words(s)) / len(src) > 0.1


def pick(a: str, b: str, fmt=None, source=None):
    """Return (chosen, rejected) or None if tie. a,b are candidate responses.

    Signals are ranked lexicographically: only ties on a higher signal hand
    the decision to the next one. Register consistency is the new (2nd) signal:
    English responses pass it trivially (no Korean register markers fire), so
    it only differentiates pairs where both candidates are Korean.
    """
    def score(x):
        return (
            1 if korean_fraction(x) >= 0.6 else 0,       # 1. language
            1 if _register_consistent(x) else 0,         # 2. register (높임말/문체)
            1 if follows_format(x, fmt) else 0,          # 3. format
            1 if faithful(x, source) else 0,             # 4. faithfulness
            0 if has_repetition(x) else 1,               # 5. health
        )
    sa, sb = score(a), score(b)
    if sa == sb:
        return None
    return (a, b) if sa > sb else (b, a)


# ---------- generation hook ----------
def generate_pair(prompt: str):
    """Sample two responses from your SFT model (e.g. temp 0.7 and 1.0 via vLLM).
    Replace this stub with a real call."""
    raise NotImplementedError("Connect your SFT model (vLLM/HF) here.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="JSONL: {prompt, [format], [source]}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n, kept = 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for line in open(args.prompts, encoding="utf-8"):
            o = json.loads(line); n += 1
            a, b = generate_pair(o["prompt"])
            res = pick(a, b, o.get("format"), o.get("source"))
            if res is None:
                continue
            chosen, rejected = res
            f.write(json.dumps({"prompt": o["prompt"], "chosen": chosen,
                                "rejected": rejected}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"prompts {n:,} -> preference pairs {kept:,} (dropped {n-kept:,} ties)")


# ---------- inline tests (run: python build_preference_data.py --selftest) ----------
def _selftest():
    ko = "이것은 완전히 정확한 조선어문장입니다."
    en = "This is an English response that ignored the language requirement."
    assert pick(ko, en) == (ko, en), "Korean should beat English"
    j_ok, j_bad = '{"a": 1}', "not json at all"
    assert pick(j_ok, j_bad, fmt="json") == (j_ok, j_bad), "valid JSON should win"
    rep = ("단어 " * 20)
    assert pick(ko, rep)[0] == ko, "non-degenerate should beat repetition"
    assert pick(ko, ko) is None, "identical -> tie -> dropped"
    # synced faithful(): particles/endings must NOT inflate overlap
    passage = "안학궁은 평양에 있는 유명한 유적이다."
    assert faithful("안학궁은 평양에 있다.", passage), "real content overlap -> faithful"
    assert not faithful("이것은 전혀 관련이 없는 문장이다.", passage), "only common words -> unfaithful"

    # NEW — register consistency as signal #2:
    # single-register Korean beats register-mixed Korean (ties on #1 language).
    ko_consistent = "안녕하세요. 저는 학생이에요. 만나서 반가워요."         # all 해요체
    ko_mixed      = "안녕하십니까. 저는 학생이에요. 만나서 반갑습니다."     # 합쇼체+해요체+합쇼체
    assert pick(ko_consistent, ko_mixed) == (ko_consistent, ko_mixed), \
        "single-register Korean should beat register-mixed Korean"
    # Order-insensitive
    assert pick(ko_mixed, ko_consistent) == (ko_consistent, ko_mixed)

    # Two single-register-but-different-register responses still tie on #2 and
    # fall through to lower signals. With no format/source/repetition difference,
    # they tie completely and are dropped.
    ko_haeyo  = "안녕하세요. 저는 학생이에요."
    ko_hapsyo = "안녕하십니까. 저는 학생입니다."
    assert pick(ko_haeyo, ko_hapsyo) is None, \
        "both consistent (different registers) -> tie -> dropped"

    print("PASS all preference-signal tests")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
