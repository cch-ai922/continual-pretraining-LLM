#!/usr/bin/env python3
"""
Stage 7 (teacher loop) — CRITIQUE-AND-REVISE for non-verifiable DPO pairs.

Constitutional AI (Anthropic, Bai et al. 2022) pattern, adapted for Korean:
  1. Model generates an INITIAL response to the prompt.
  2. Model is asked to CRITIQUE its own response against a rubric (faithfulness,
     completeness, clarity, language quality).
  3. Model generates a REVISED response using the critique.
  4. Output: a preference pair {prompt, chosen: revised, rejected: initial}.

The key insight: the *same* model can usually critique better than it can produce.
Asking it "what's wrong with this answer?" surfaces flaws it wouldn't avoid in
one-shot generation — and the revision step then incorporates the critique. This
yields preference pairs for DPO **without any external judge or verifiable signal**,
which is what makes the technique work for advice, explanation, summary, creative
writing — any of the non-verifiable domains where rejection_sample.py can't help.

Three failure modes worth knowing about:
  - "Empty critique"   : model says "the answer is good" → initial ≈ revised → no useful pair
  - "Worse revision"   : revision actually degrades quality → flip the DPO pair (or drop it)
  - "Sycophantic agreement" with self → critique loop loses bite over rounds; one pass is best
"""
import argparse, json, sys, unicodedata


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
DEFAULT_RUBRIC = (
    "1. 답이 완전히 조선어로 되여있는가?\n"
    "2. 답이 사실적으로 정확한가?\n"
    "3. 답이 질문의 내용을 완전히 다루는가 (불완전하지 않은가)?\n"
    "4. 답이 명확하고 론리적으로 구성되여있는가?\n"
    "5. 언어가 자연스럽고 정확한가 (번역투이거나 어색하지 않은가)?"
)


def build_critique_messages(prompt: str, initial: str, rubric: str = DEFAULT_RUBRIC):
    """Ask the model to find specific flaws in its own initial answer."""
    user = (f"아래에 질문 하나와 그 답이 있습니다. 다음의 기준에 따라 답의 "
            f"부족한 점을 지적하시오.\n\n기준:\n{rubric}\n\n"
            f"질문: {prompt}\n\n답:\n{initial}\n\n"
            f"검토: 이 답은 어떤 점에서 더 개선될수 있습니까? "
            f"짧고 명확한 목록으로 제시하시오.")
    return [{"role": "user", "content": user}]


def build_revise_messages(prompt: str, initial: str, critique: str):
    """Ask the model to produce an improved answer that addresses the critique."""
    user = (f"아래에 질문과 초기답, 그리고 그에 대한 검토가 있습니다. "
            f"검토에서 제기된 모든 문제를 반영하여 더 나은 답을 작성하시오.\n\n"
            f"질문: {prompt}\n\n초기 답:\n{initial}\n\n"
            f"검토:\n{critique}\n\n더 나은 답:")
    return [{"role": "user", "content": user}]


def korean_fraction(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "HANGUL" in unicodedata.name(c, "")) / len(letters)


# Phrases that signal the critique found nothing wrong (empty critique = drop pair)
_EMPTY_CRITIQUE_MARKERS = [
    "부족한 점이 없", "답이 좋", "수정할 필요가 없", "답이 정확하",
    "완벽하", "개선이 필요없", "문제가 없", "고칠 점이 없",
]


def is_useful_critique(critique: str, min_len: int = 30) -> bool:
    """Heuristic: did the critique actually surface any flaw?"""
    if not critique or len(critique) < min_len:
        return False
    low = critique.strip()
    return not any(m in low for m in _EMPTY_CRITIQUE_MARKERS)


def is_meaningful_revision(initial: str, revised: str, min_char_diff: int = 30,
                           min_jaccard_diff: float = 0.15) -> bool:
    """Revision must be substantively different from the initial, not just paraphrased."""
    if not revised or not initial:
        return False
    if abs(len(revised) - len(initial)) < min_char_diff and revised.strip() == initial.strip():
        return False
    # token-set Jaccard distance: revision should change ≥ min_jaccard_diff fraction of tokens
    def toks(s):
        return {w.strip("。.,?!\"'()[]{}:;…·") for w in s.split() if len(w) > 1}
    A, B = toks(initial), toks(revised)
    if not A or not B:
        return False
    jaccard = len(A & B) / len(A | B)
    return (1.0 - jaccard) >= min_jaccard_diff


def acceptable_pair(prompt: str, initial: str, critique: str, revised: str,
                    min_korean: float = 0.7) -> bool:
    """All gates a (initial, critique, revised) tuple must pass to become a DPO pair."""
    if korean_fraction(revised) < min_korean:
        return False
    if not is_useful_critique(critique):
        return False
    if not is_meaningful_revision(initial, revised):
        return False
    return True


# ---------------------------------------------------------------------------
# MODEL HOOK
# ---------------------------------------------------------------------------
def chat(messages_batch, model_path, max_new_tokens=1024, temperature=0.7):
    """Single-turn chat completion. Same signature as ai_feedback_judge.chat."""
    raise NotImplementedError("Connect your model in chat(); return list[str].")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the SFT model that critiques itself")
    ap.add_argument("--prompts", required=True, help="JSONL: {prompt}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rubric", default=DEFAULT_RUBRIC)
    args = ap.parse_args()

    prompts = [json.loads(l)["prompt"] for l in open(args.prompts, encoding="utf-8")]
    kept = dropped = 0
    fout = open(args.out, "w", encoding="utf-8")

    for i in range(0, len(prompts), args.batch):
        batch = prompts[i:i + args.batch]
        # 1. initial responses
        initials = chat([[{"role": "user", "content": p}] for p in batch], args.model)
        # 2. critiques
        critiques = chat([build_critique_messages(p, ini, args.rubric)
                          for p, ini in zip(batch, initials)], args.model)
        # 3. revisions
        revisions = chat([build_revise_messages(p, ini, cri)
                          for p, ini, cri in zip(batch, initials, critiques)], args.model)
        for p, ini, cri, rev in zip(batch, initials, critiques, revisions):
            if not acceptable_pair(p, ini, cri, rev):
                dropped += 1; continue
            fout.write(json.dumps({"prompt": p,
                                   "chosen": rev.strip(),
                                   "rejected": ini.strip()},
                                  ensure_ascii=False) + "\n")
            kept += 1
    fout.close()
    print(f"kept {kept:,} DPO pairs | dropped {dropped:,} -> {args.out}")


# ---------------------------------------------------------------------------
def _selftest():
    # message construction
    cm = build_critique_messages("무엇입니까?", "한 답.")
    assert cm[0]["role"] == "user" and "기준" in cm[0]["content"]
    rm = build_revise_messages("무엇입니까?", "한 답.", "어떤 의견")
    assert "더 나은 답" in rm[0]["content"]

    # critique-quality detection
    assert not is_useful_critique("부족한 점이 없습니다.")
    assert not is_useful_critique("답이 완벽합니다.")
    assert is_useful_critique("언어가 약간 번역투입니다. 사실 2에서 '1947'이여야 하며 '1948'이 아닙니다.")
    assert not is_useful_critique("괜찮음.")   # too short

    # revision-meaningfulness detection
    initial = "조선민주주의인민공화국의 수도는 평양이다."
    paraphrased = "조선민주주의인민공화국의 수도는 평양이다."   # identical
    expanded = ("조선민주주의인민공화국의 수도는 평양이며 조선반도 중부이북에 위치한다. "
                "평양은 우리 나라의 중심지로서 조선로동당 중앙위원회와 정부가 여기에 있다.")
    assert not is_meaningful_revision(initial, paraphrased)
    assert is_meaningful_revision(initial, expanded)

    # full gate
    good_critique = "답이 너무 짧습니다; 지리적특성과 중요성을 추가해야 합니다."
    assert acceptable_pair("조선민주주의인민공화국의 수도는 어디입니까?", initial, good_critique, expanded)
    assert not acceptable_pair("조선민주주의인민공화국의 수도는 어디입니까?", initial,
                                "부족한 점이 없습니다.", expanded)        # empty critique
    assert not acceptable_pair("조선민주주의인민공화국의 수도는 어디입니까?", initial,
                                good_critique, initial)               # no real revision
    # English revision should be rejected even if structurally meaningful
    eng_rev = "The capital of South Korea is Seoul located in the central western Korean peninsula."
    assert not acceptable_pair("조선민주주의인민공화국의 수도는 어디입니까?", initial, good_critique, eng_rev)
    print("PASS critique-revise tests")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
