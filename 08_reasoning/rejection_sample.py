#!/usr/bin/env python3
"""
Stage 8 — rejection-sample verified Korean chain-of-thought solutions.

For each translated Korean math problem:
  1. sample K solutions from the model at high temperature  (diverse reasoning)
  2. extract the final answer with verify_math.extract_final_answer
  3. compare to the gold answer  (language-independent verifier)
  4. KEEP the correct ones as SFT data
  5. (optionally) keep ONE incorrect solution per problem to form DPO pairs

The verified solutions become Korean CoT SFT data — chain-of-thought reasoning in
Korean, validated by an external (correctness) signal that doesn't depend on the
model's own judgment. This is the "RFT" / "STaR" pattern: distill the model's own
best samples back into itself, with verifiable filtering.

Output formats:
  --out-sft : chat JSONL  {messages: [user(problem), assistant(verified CoT)]}
  --out-dpo : DPO JSONL   {prompt(problem), chosen(verified), rejected(unverified)}
"""
import argparse, importlib, inspect, json, sys, os
sys.path.insert(0, os.path.dirname(__file__))


def load_verifier(name: str):
    """Import verify_<name>.py and return its `verify(output, gold) -> bool`.
    Supported: math, code, mcq, logic, format (drop in your own with the same API)."""
    return importlib.import_module(f"verify_{name}").verify


SYSTEM_KO = ("당신은 수학교원입니다. 아래의 문제를 단계별로 풀고 "
             "마지막에 명확히 적으시오: '최종답: <수자>'.")
SYSTEM_BY_VERIFIER = {                                  # task-appropriate system prompt
    "math": SYSTEM_KO,
    "code": ("당신은 프로그람작성자입니다. 문제를 읽고 Python으로 완전한 풀이법을 작성하여 "
             "```python ... ``` 블로크안에 넣으시오."),
    "mcq": ("다음 문제의 가장 알맞는 답을 고르시오. 근거를 제시하고 마지막에 "
            "적으시오: '정답: <보기>'."),
    "logic": ("아래의 수수께끼를 풀고 최종 해답을 명확하게 제시하시오."),
    "format": ("요청된 형식을 엄격히 따라 답하시오."),
}


def build_messages(problem, system):
    return [{"role": "system", "content": system},
            {"role": "user", "content": problem}]


def chat(messages_batch, model_path, k=8, max_new_tokens=512, temperature=0.9):
    """Sample K completions per prompt from the INSTRUCT (or base+system-prompt) model.

    vLLM (recommended):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in messages_batch]
        sp = SamplingParams(n=k, temperature=temperature, top_p=0.95,
                            max_tokens=max_new_tokens)
        outs = llm.generate(prompts, sp)
        return [[o.text for o in r.outputs] for r in outs]   # list[list[str]]
    """
    raise NotImplementedError("Connect your model in chat(); must return list[list[str]] (k per prompt).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True,
                    help="JSONL: {problem, gold, ...}  (from translate_problems.py)")
    ap.add_argument("--out-sft", required=True)
    ap.add_argument("--out-dpo", default=None)
    ap.add_argument("--k", type=int, default=8, help="samples per problem")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--keep-per-problem", type=int, default=2,
                    help="cap correct solutions kept per problem (avoid duplication)")
    ap.add_argument("--verifier", default="math",
                    choices=["math", "code", "mcq", "logic", "format"],
                    help="which verify_<name>.py to dispatch to (all share the same API)")
    ap.add_argument("--system", default=None,
                    help="override the per-verifier default system prompt")
    args = ap.parse_args()

    verify = load_verifier(args.verifier)
    # Some verifiers need extra args beyond (output, gold) — e.g. verify_mcq needs
    # n_options. Pass through any per-problem field whose name matches a verify()
    # parameter, EXCEPT the positional output/gold slots (which we pass explicitly).
    _verify_params = set(inspect.signature(verify).parameters)
    _POSITIONAL = {"model_output", "output", "response", "prediction",
                   "gold", "gold_answer", "answer", "tests", "puzzle", "problem"}
    _extra_keys = _verify_params - _POSITIONAL
    system = args.system or SYSTEM_BY_VERIFIER[args.verifier]
    problems = [json.loads(l) for l in open(args.problems, encoding="utf-8")]
    msgs = [build_messages(p["problem"], system) for p in problems]

    sft = open(args.out_sft, "w", encoding="utf-8")
    dpo = open(args.out_dpo, "w", encoding="utf-8") if args.out_dpo else None
    n_correct = n_problems_solved = 0

    for i in range(0, len(problems), args.batch):
        batch_msgs = msgs[i:i + args.batch]
        batch_probs = problems[i:i + args.batch]
        samples_per = chat(batch_msgs, args.model, k=args.k)
        for prob, samples in zip(batch_probs, samples_per):
            gold = prob.get("gold", prob.get("answer", prob.get("tests")))
            extra = {k: prob[k] for k in _extra_keys if k in prob}
            correct, incorrect = [], []
            for s in samples:
                (correct if verify(s, gold, **extra) else incorrect).append(s)
            if correct:
                n_problems_solved += 1
                for s in correct[: args.keep_per_problem]:
                    sft.write(json.dumps({"messages": [
                        {"role": "user", "content": prob["problem"]},
                        {"role": "assistant", "content": s.strip()}]},
                                         ensure_ascii=False) + "\n")
                    n_correct += 1
                if dpo and incorrect:
                    dpo.write(json.dumps({"prompt": prob["problem"],
                                          "chosen": correct[0].strip(),
                                          "rejected": incorrect[0].strip()},
                                         ensure_ascii=False) + "\n")

    sft.close()
    if dpo:
        dpo.close()
    print(f"problems with ≥1 verified solution: {n_problems_solved:,}/{len(problems):,}  "
          f"({n_problems_solved/max(len(problems),1):.1%})")
    print(f"SFT examples written: {n_correct:,} -> {args.out_sft}")
    if args.out_dpo:
        print(f"DPO pairs written: see {args.out_dpo}")


if __name__ == "__main__":
    main()
