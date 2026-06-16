#!/usr/bin/env python3
"""
Stage 7 (teacher) — AI-feedback judge in Korean, using the STAGE-5 INSTRUCT model,
to produce preference pairs for Stage-6 DPO without native annotators.

For each prompt: take two candidate responses, ask the instruct model which is better
against a rubric, and — crucially — judge in BOTH orders to cancel position bias;
keep the pair only if the two verdicts AGREE. Optionally have the judge reason in
English first (it judges best there) before the Korean verdict.

Output: DPO JSONL {prompt, chosen, rejected}.
Connect your instruct model in `chat()`. Pure logic is unit-tested (--selftest).
"""
import argparse, json, re

RUBRIC = ("도움됨(helpfulness), 사실정확성(correctness), "
          "류창성(fluency), 출처충실성(faithfulness), "
          "그리고 옳바른 언어(정확한 조선어)")


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
def judge_messages(prompt, first, second, rubric=RUBRIC, reason_in_english=False):
    """Chat messages asking which of two responses (shown as 답 A / 답 B) is better."""
    think = ("Briefly reason in English, then " if reason_in_english else "")
    user = (f"아래에 질문 하나와 두개의 답이 있습니다. 다음의 기준에 따라 어느 답이 "
            f"더 나은지 판단하시오: {rubric}.\n\n"
            f"질문: {prompt}\n\n답 A:\n{first}\n\n답 B:\n{second}\n\n"
            f"{think}마지막에 정확히 이 형식으로 판정을 내리시오 — '판정: A' 또는 '판정: B'.")
    return [{"role": "user", "content": user}]


def parse_verdict(text):
    """Return 'A' / 'B' / None. Looks for an explicit verdict marker, else last A/B."""
    m = re.search(r"(?:판정|승자|winner|verdict)\s*[:：]\s*\"?([ABab])", text)
    if m:
        return m.group(1).upper()
    # JSON fallback
    try:
        j = json.loads(text[text.index("{"): text.rindex("}") + 1])
        v = str(j.get("winner") or j.get("판정") or "").strip().upper()
        if v in ("A", "B"):
            return v
    except Exception:
        pass
    return None


def resolve(v_order1, v_order2):
    """v_order1: winning POSITION when shown (A=respA, B=respB).
       v_order2: winning POSITION when shown SWAPPED (A=respB, B=respA).
       Return 'A'/'B' (in terms of the ORIGINAL respA/respB) only if the two agree."""
    m1 = {"A": "A", "B": "B"}.get(v_order1)
    m2 = {"A": "B", "B": "A"}.get(v_order2)   # account for the swap
    return m1 if (m1 and m2 and m1 == m2) else None


# ---------------------------------------------------------------------------
# MODEL HOOK
# ---------------------------------------------------------------------------
def chat(messages_batch, model_path, max_new_tokens=512, temperature=0.0):
    """Batch chat-completion with the INSTRUCT model (greedy for judging). See
    self_instruct.chat() for the vLLM/HF pattern."""
    raise NotImplementedError("Connect your Stage-5 instruct model in chat().")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Stage-5 Korean instruct model (judge)")
    ap.add_argument("--in", dest="inp", required=True,
                    help="JSONL: {prompt, response_a, response_b}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reason-en", action="store_true", help="let the judge reason in English first")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    # build both orderings for every row, judge in one batched pass each
    order1 = [judge_messages(r["prompt"], r["response_a"], r["response_b"], reason_in_english=args.reason_en) for r in rows]
    order2 = [judge_messages(r["prompt"], r["response_b"], r["response_a"], reason_in_english=args.reason_en) for r in rows]

    def run(msgs):
        out = []
        for i in range(0, len(msgs), args.batch):
            out += chat(msgs[i:i + args.batch], args.model)
        return out

    v1 = [parse_verdict(t) for t in run(order1)]
    v2 = [parse_verdict(t) for t in run(order2)]

    kept = ties = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r, a, b in zip(rows, v1, v2):
            winner = resolve(a, b)
            if winner is None:
                ties += 1; continue
            chosen = r["response_a"] if winner == "A" else r["response_b"]
            rejected = r["response_b"] if winner == "A" else r["response_a"]
            f.write(json.dumps({"prompt": r["prompt"], "chosen": chosen,
                                "rejected": rejected}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"preference pairs: kept {kept:,} | dropped (disagree/ambiguous) {ties:,} -> {args.out}")


# ---------------------------------------------------------------------------
def _selftest():
    m = judge_messages("조선의 수도는?", "평양", "개성")
    assert m[0]["role"] == "user" and "답 A" in m[0]["content"] and "답 B" in m[0]["content"]
    assert parse_verdict("어떤 근거 ... 판정: A") == "A"
    assert parse_verdict("verdict: b") == "B"
    assert parse_verdict('{"winner": "B"}') == "B"
    assert parse_verdict("잘 모르겠다") is None
    # position-bias resolution
    assert resolve("A", "B") == "A"     # respA won in both orders -> A
    assert resolve("B", "A") == "B"     # respB won in both orders -> B
    assert resolve("A", "A") is None    # disagreement -> drop
    assert resolve("A", None) is None
    print("PASS all ai-feedback-judge tests")


if __name__ == "__main__":
    import sys
    _selftest() if "--selftest" in sys.argv else main()
