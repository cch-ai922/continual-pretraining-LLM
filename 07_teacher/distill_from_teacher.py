#!/usr/bin/env python3
"""
Stage 7 (teacher loop) — DISTILL chain-of-thought from a strong reasoner.

The realistic path for general (non-verifiable) Korean CoT data. For each prompt:
  1. Ask a strong reasoning model (Claude, GPT-4-class, DeepSeek-R1, Qwen3-235B-thinking,
     or any local R1-distill checkpoint) to produce a Korean step-by-step response.
  2. Filter by language (Korean fraction), length, and structural CoT markers.
  3. Emit as standard SFT JSONL: {messages: [user(prompt), assistant(cot_response)]}.

This is the technique behind almost every open reasoning model released after R1.
Connect your teacher in `chat_teacher()` — works with any API or local serving.

CRITICAL: the teacher's quality bounds the student's. If the teacher hallucinates
or code-switches into English mid-reasoning, those errors propagate into your SFT
set unfiltered. The structural checks here catch the most egregious cases; quality
spot-checks on a 200-sample subset are still essential before scaling up.
"""
import argparse, json, re, unicodedata, sys


SYSTEM_KO_COT = (
    "당신은 전문교원입니다. 아래의 문제에 대답할 때:\n"
    "1. 먼저 사고과정을 단계별로 설명하시오.\n"
    "2. 중요한 매 단계를 새 행에 작성하시오.\n"
    "3. 답을 도출하는데 계산이 필요하면 그 계산을 보여주시오.\n"
    "4. 마지막에 최종답을 분명하게 적으시오.\n"
    "전체 답은 정확한 조선어로 작성하시오."
)


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
def korean_fraction(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "HANGUL" in unicodedata.name(c, "")) / len(letters)


# Markers that indicate the model walked through reasoning rather than blurting
# a one-line answer. Korean connectives + numbered/bulleted structure + English
# "thinking" markers (since some teachers emit those even when asked for Korean).
_COT_MARKERS = [
    "왜냐하면", "따라서", "그러므로", "먼저", "그다음", "다음 단계", "주목하시오",
    "우리는 안다", "가정하자", "최종답", "결론", "단계",
    "step", "first", "then", "therefore", "because",
]
_NUMBERED = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+\S")


def has_cot_structure(response: str, min_markers: int = 2, min_lines: int = 3) -> bool:
    """Heuristic: does this look like step-by-step reasoning, not a one-shot answer?"""
    if not response:
        return False
    marker_hits = sum(1 for m in _COT_MARKERS if m in response)
    line_count = sum(1 for ln in response.splitlines() if ln.strip())
    numbered_steps = len(_NUMBERED.findall(response))
    return (marker_hits >= min_markers) or (numbered_steps >= 2) or (line_count >= min_lines and marker_hits >= 1)


def acceptable_cot(prompt: str, response: str,
                   min_korean: float = 0.7, min_len: int = 60, max_len: int = 6000) -> bool:
    """Length + language purity + structural CoT gate."""
    if not (min_len <= len(response) <= max_len):
        return False
    if korean_fraction(response) < min_korean:
        return False
    return has_cot_structure(response)


# ---------------------------------------------------------------------------
# TEACHER HOOK  (the model-dependent part)
# ---------------------------------------------------------------------------
def chat_teacher(messages_batch, teacher: str = "claude",
                 max_new_tokens: int = 2048, temperature: float = 0.7):
    """Batch chat-completion with a strong reasoner. Returns list[str].

    Examples for popular teachers:

    Anthropic Claude (with extended thinking):
        from anthropic import Anthropic
        client = Anthropic()
        outs = []
        for m in messages_batch:
            r = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=max_new_tokens,
                thinking={"type": "enabled", "budget_tokens": 4000},
                system=m[0]["content"] if m[0]["role"] == "system" else None,
                messages=[x for x in m if x["role"] != "system"],
            )
            text = "".join(b.text for b in r.content if b.type == "text")
            outs.append(text)
        return outs

    DeepSeek-R1 via vLLM (local):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in messages_batch]
        sp = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_new_tokens)
        outs = llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outs]

    OpenAI (GPT-4-class with thinking):
        from openai import OpenAI
        client = OpenAI()
        ... (similar pattern, use reasoning_effort='high')
    """
    raise NotImplementedError(f"Connect teacher='{teacher}' in chat_teacher().")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="JSONL: {prompt}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="claude",
                    help="identifier for chat_teacher dispatch (claude/r1/gpt/...)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--system", default=SYSTEM_KO_COT,
                    help="override the default Korean CoT system prompt")
    args = ap.parse_args()

    prompts = [json.loads(l)["prompt"] for l in open(args.prompts, encoding="utf-8")]
    msgs = [[{"role": "system", "content": args.system},
             {"role": "user", "content": p}] for p in prompts]

    kept = dropped = 0
    fout = open(args.out, "w", encoding="utf-8")
    for i in range(0, len(prompts), args.batch):
        outs = chat_teacher(msgs[i:i + args.batch], teacher=args.teacher)
        for p, r in zip(prompts[i:i + args.batch], outs):
            if not acceptable_cot(p, r):
                dropped += 1; continue
            fout.write(json.dumps({"messages": [
                {"role": "user", "content": p},
                {"role": "assistant", "content": r.strip()}]},
                                  ensure_ascii=False) + "\n")
            kept += 1
    fout.close()
    print(f"distilled {kept:,} CoT examples | dropped {dropped:,} -> {args.out}")


# ---------------------------------------------------------------------------
def _selftest():
    # CoT structure detection
    good = ("먼저 문제를 리해한다. 속도 = 거리 / 시간이므로, "
            "따라서 거리 = 속도 × 시간 = 60 × 2.5 = 150키로메터.\n"
            "그러므로 최종답은 150키로메터이다.")
    assert has_cot_structure(good)
    bad = "150 키로메터."
    assert not has_cot_structure(bad)
    # numbered-list style also accepted
    numbered = "1. 첫번째 단계\n2. 두번째 단계\n3. 세번째 단계"
    assert has_cot_structure(numbered)
    # acceptable_cot combines all gates
    assert acceptable_cot("무엇입니까?", good)
    assert not acceptable_cot("무엇입니까?", "very short")
    # English-heavy response should be rejected
    eng = ("First we understand the problem. Since speed = distance/time, "
           "therefore distance = 60 × 2.5 = 150 km. So the final answer is 150 km.")
    assert not acceptable_cot("무엇입니까?", eng)
    # length bounds
    assert not acceptable_cot("무엇입니까?", good * 200)  # too long
    print("PASS distill-from-teacher tests")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
