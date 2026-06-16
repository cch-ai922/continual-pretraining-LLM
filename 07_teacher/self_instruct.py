#!/usr/bin/env python3
"""
Stage 7 (teacher) — Self-Instruct in Korean, using the STAGE-5 INSTRUCT model.

Now that the model follows instructions, bootstrap a larger, diverse instruction set
from a small native seed pool:
  1. generate new instructions (few-shot off the seed pool)
  2. generate an output for each new instruction
  3. FILTER: dedup vs pool (diversity), language-ID, length/format
  4. add survivors, repeat

Cross-lingual leverage: the model's instruction/output quality rides on reasoning &
knowledge transferred from English in Stage 4, so native Korean generations are better
than a from-scratch Korean model could produce.

Output: chat-format SFT JSONL that feeds back into Stage 5 (build_sft_blend.py).
Connect your instruct model in `chat()`. Pure logic is unit-tested (--selftest).
"""
import argparse, json, re, unicodedata


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
def korean_fraction(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "HANGUL" in unicodedata.name(c, "")) / len(letters)


def gen_instructions_messages(seed_instructions, n=8):
    """Chat messages that ask the instruct model for n NEW, diverse Korean tasks."""
    examples = "\n".join(f"{i+1}. {ins}" for i, ins in enumerate(seed_instructions))
    user = (f"아래에 몇가지 작업(지시)이 있습니다:\n{examples}\n\n"
            f"이와 다르고 다양한 {n}개의 새로운 작업을 조선어로 작성하시오. "
            f"매 작업을 번호와 함께 새 줄에 작성하시오. "
            f"작업은 명확하고 다양한 주제에서 나오도록 하시오.")
    return [{"role": "user", "content": user}]


def parse_instructions(text):
    """Extract '1. ...' / '2) ...' style instructions from the model output."""
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+[.)]\s+(.*\S)", line)
        if m:
            out.append(m.group(1).strip())
    return out


def _tokset(s):
    return {w.strip("。.,?!\"'()[]{}:;…·") for w in s.split() if len(w) > 1}


def too_similar(a, b, thresh=0.7):
    """Token Jaccard similarity (cheap diversity guard; swap for embeddings at scale)."""
    A, B = _tokset(a), _tokset(b)
    if not A or not B:
        return False
    return len(A & B) / len(A | B) >= thresh


def dedup(candidates, existing, thresh=0.7):
    kept = []
    pool = list(existing)
    for c in candidates:
        if any(too_similar(c, e, thresh) for e in pool):
            continue
        kept.append(c); pool.append(c)
    return kept


def answer_messages(instruction):
    return [{"role": "user", "content": instruction}]


def acceptable_output(instruction, output, min_korean=0.7, min_len=2, max_len=4000):
    if not (min_len <= len(output.split()) and len(output) <= max_len):
        return False
    return korean_fraction(output) >= min_korean and korean_fraction(instruction) >= min_korean


# ---------------------------------------------------------------------------
# MODEL HOOK
# ---------------------------------------------------------------------------
def chat(messages_batch, model_path, max_new_tokens=512, temperature=0.9):
    """Batch chat-completion with the INSTRUCT model. Returns list[str].

    vLLM:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in messages_batch]
        llm = LLM(model_path)
        sp = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_new_tokens)
        return [o.outputs[0].text for o in llm.generate(prompts, sp)]
    """
    raise NotImplementedError("Connect your Stage-5 instruct model in chat().")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Stage-5 Korean instruct model")
    ap.add_argument("--seed", required=True, help="JSON list of native Korean seed instructions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--per-prompt", type=int, default=8)
    ap.add_argument("--seeds-per-prompt", type=int, default=6)
    ap.add_argument("--target", type=int, default=50000)
    ap.add_argument("--sim-thresh", type=float, default=0.7)
    args = ap.parse_args()

    import random
    pool = list(json.load(open(args.seed, encoding="utf-8")))
    rng = random.Random(0)
    fout = open(args.out, "w", encoding="utf-8")
    written = 0

    for _ in range(args.rounds):
        if written >= args.target:
            break
        # 1. propose instructions (a batch of few-shot prompts)
        batch = [gen_instructions_messages(rng.sample(pool, min(args.seeds_per_prompt, len(pool))),
                                           args.per_prompt) for _ in range(64)]
        raw = chat(batch, args.model)
        proposed = [ins for r in raw for ins in parse_instructions(r)]
        # 2. dedup for diversity
        fresh = dedup(proposed, pool, args.sim_thresh)
        if not fresh:
            continue
        # 3. generate outputs
        outs = chat([answer_messages(i) for i in fresh], args.model)
        for ins, out in zip(fresh, outs):
            if acceptable_output(ins, out):
                fout.write(json.dumps({"messages": [
                    {"role": "user", "content": ins},
                    {"role": "assistant", "content": out.strip()}]}, ensure_ascii=False) + "\n")
                pool.append(ins); written += 1

    fout.close()
    print(f"wrote {written:,} self-instruct examples -> {args.out} (pool now {len(pool):,})")


# ---------------------------------------------------------------------------
def _selftest():
    msgs = gen_instructions_messages(["시를 쓰시오", "번역하시오"], n=5)
    assert msgs[0]["role"] == "user" and "5" in msgs[0]["content"]
    parsed = parse_instructions("1. 이야기를 쓰시오\n2) 수학문제를 푸시오\n쓸모없는 줄\n3. 요약하시오")
    assert parsed == ["이야기를 쓰시오", "수학문제를 푸시오", "요약하시오"], parsed
    assert too_similar("이야기를 쓰시오", "이야기를 쓰시오")
    assert not too_similar("시를 쓰시오", "수학문제를 푸시오")
    kept = dedup(["새로운것을 말하시오", "시를 쓰시오"], ["시를 쓰시오"])
    assert kept == ["새로운것을 말하시오"], kept
    assert acceptable_output("이야기를 쓰시오", "이것은 충분한 단어를 가진 긴 조선어이야기입니다.")
    assert not acceptable_output("write a story", "this is english output not korean at all")
    print("PASS all self-instruct tests")


if __name__ == "__main__":
    import sys
    _selftest() if "--selftest" in sys.argv else main()
