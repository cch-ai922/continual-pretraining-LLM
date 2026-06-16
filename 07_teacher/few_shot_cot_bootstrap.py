#!/usr/bin/env python3
"""
Stage 7 (teacher loop) — STaR-style BOOTSTRAP FROM BASE MODEL (no external teacher).

The historical technique that produced the first reasoning models, BEFORE strong
reasoners existed to distill from. Zelikman et al. 2022 (Self-Taught Reasoner):

  1. Few-shot CoT prompt the BASE model with hand-written (problem, solution) exemplars.
  2. Sample K continuations.
  3. Verify each against the gold answer.
  4. KEEP the correct ones as SFT data for the next round.
  5. For problems where 0 of K were correct: RATIONALIZE. Build a different prompt
     that gives the model the gold answer and asks for reasoning that arrives there.
     Verify the rationalized chain still extracts to the gold; keep if so.
  6. SFT the base model on the verified chains; repeat the whole loop.

Why this works: the base model has *seen* reasoning during pretraining (textbooks,
math.stackexchange, GitHub READMEs, scientific papers). It just doesn't reliably
*produce* it without prompting. Few-shot ELICITS the latent capability; verification
FILTERS for correctness; SFT AMPLIFIES what worked. After 1-2 iterations the model
solves ~2-3× more problems than at K=1 baseline.

Rationalization is the under-appreciated ingredient: it salvages problems the
model "almost solved" by working backward from the answer, which is a much
easier task than forward solving. Without it you only collect data on problems
the model already gets right ~once-in-K, which limits curriculum to the easy tail.
"""
import argparse, importlib, inspect, json, os, sys

DELIM = "\n\n###\n\n"


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
def build_few_shot_prompt(exemplars: list, problem: str) -> str:
    """Standard completion-style few-shot prompt — the BASE model continues from '풀이:\\n'."""
    blocks = [f"문제: {ex['problem']}\n풀이:\n{ex['solution']}" for ex in exemplars]
    blocks.append(f"문제: {problem}\n풀이:\n")
    return DELIM.join(blocks)


def build_rationalize_prompt(exemplars: list, problem: str, gold_answer) -> str:
    """Backward-reasoning prompt: gold answer is given, ask the model to fill in why.

    Reaching a known answer is a much easier task than discovering it, so problems
    the model fails forward-style often succeed when rationalized."""
    blocks = [f"문제: {ex['problem']}\n풀이:\n{ex['solution']}" for ex in exemplars]
    blocks.append(
        f"문제: {problem}\n"
        f"정답이 {gold_answer}임을 알고있습니다. 이 답이 어떻게 나오는지 "
        f"단계별로 설명하시오:\n풀이:\n"
    )
    return DELIM.join(blocks)


def parse_completion(completion: str) -> str:
    """Cut the continuation at the next problem boundary or exemplar delimiter."""
    return (completion.split(DELIM)[0]
                       .split("\n문제:")[0]
                       .split("\n\n문제:")[0]
                       .strip())


# ---------------------------------------------------------------------------
# MODEL HOOK (BASE-MODEL completion, NOT chat)
# ---------------------------------------------------------------------------
def generate(prompts, model_path, k=8, max_new_tokens=512, temperature=0.8):
    """Sample K completions per prompt from the BASE model. Returns list[list[str]].

    Unlike rejection_sample.chat() which uses chat templates, this is raw completion
    — the base model is a completion model, not an instruction-tuned one.

    vLLM template:
        from vllm import LLM, SamplingParams
        llm = LLM(model_path)
        sp = SamplingParams(n=k, temperature=temperature, top_p=0.95,
                            max_tokens=max_new_tokens,
                            stop=["\\n문제:", DELIM, "\\n\\n문제:"])
        outs = llm.generate(prompts, sp)
        return [[o.text for o in r.outputs] for r in outs]
    """
    raise NotImplementedError("Connect your BASE model in generate(); return list[list[str]].")


def load_verifier(name: str):
    return importlib.import_module(f"verify_{name}").verify


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to BASE model (completion, not instruct)")
    ap.add_argument("--exemplars", required=True,
                    help='JSON list of {"problem","solution","answer"}. Ships: '
                         'exemplars_math_cot / code_cot / mcq_cot / logic_cot / '
                         'format_cot.json — pick the one matching --verifier.')
    ap.add_argument("--problems", required=True,
                    help="JSONL: {problem, gold} (from converters/ → translate_problems.py)")
    ap.add_argument("--out-sft", required=True)
    ap.add_argument("--verifier", default="math",
                    choices=["math", "code", "mcq", "logic", "format"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--rationalize", action="store_true",
                    help="enable rationalization for problems where all K samples failed")
    ap.add_argument("--n-exemplars", type=int, default=6,
                    help="how many of the provided exemplars to use in each prompt")
    ap.add_argument("--keep-per-problem", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "08_reasoning"))
    verify = load_verifier(args.verifier)
    # mcq's verify needs n_options; pass through any per-problem field matching a
    # verify() parameter (same signature-based dispatch as rejection_sample.py).
    _vparams = set(inspect.signature(verify).parameters)
    _POSITIONAL = {"model_output", "output", "response", "prediction",
                   "gold", "gold_answer", "answer", "tests", "puzzle", "problem"}
    _extra_keys = _vparams - _POSITIONAL

    all_exemplars = json.load(open(args.exemplars, encoding="utf-8"))
    exemplars = all_exemplars[: args.n_exemplars]
    problems = [json.loads(l) for l in open(args.problems, encoding="utf-8")]

    forward_prompts = [build_few_shot_prompt(exemplars, p["problem"]) for p in problems]

    sft = open(args.out_sft, "w", encoding="utf-8")
    n_solved_forward = n_solved_rationalize = n_total_kept = 0
    failed_problems_idx = []

    for i in range(0, len(forward_prompts), args.batch):
        batch_idx = list(range(i, min(i + args.batch, len(forward_prompts))))
        batch_prompts = [forward_prompts[j] for j in batch_idx]
        samples_per = generate(batch_prompts, args.model, k=args.k)
        for j, samples in zip(batch_idx, samples_per):
            prob = problems[j]
            gold = prob.get("gold", prob.get("answer"))
            extra = {k: prob[k] for k in _extra_keys if k in prob}
            correct = [parse_completion(s) for s in samples if verify(s, gold, **extra)]
            if correct:
                n_solved_forward += 1
                for cot in correct[: args.keep_per_problem]:
                    sft.write(json.dumps({"messages": [
                        {"role": "user", "content": prob["problem"]},
                        {"role": "assistant", "content": cot}]},
                                         ensure_ascii=False) + "\n")
                    n_total_kept += 1
            else:
                failed_problems_idx.append(j)

    print(f"forward pass: solved {n_solved_forward:,}/{len(problems):,} problems  "
          f"({n_solved_forward / max(len(problems), 1):.1%})")

    # ---- Rationalization pass on the problems we couldn't solve forward ----
    if args.rationalize and failed_problems_idx:
        rat_prompts = [build_rationalize_prompt(exemplars, problems[j]["problem"],
                                                problems[j].get("gold", problems[j].get("answer")))
                       for j in failed_problems_idx]
        for i in range(0, len(rat_prompts), args.batch):
            batch_slice = failed_problems_idx[i:i + args.batch]
            batch_prompts = rat_prompts[i:i + args.batch]
            samples_per = generate(batch_prompts, args.model, k=args.k)
            for j, samples in zip(batch_slice, samples_per):
                prob = problems[j]
                gold = prob.get("gold", prob.get("answer"))
                # rationalization MUST still verify against the gold (the model might
                # produce reasoning that doesn't actually reach the stated answer)
                extra = {k: prob[k] for k in _extra_keys if k in prob}
                correct = [parse_completion(s) for s in samples if verify(s, gold, **extra)]
                if correct:
                    n_solved_rationalize += 1
                    for cot in correct[: args.keep_per_problem]:
                        sft.write(json.dumps({"messages": [
                            {"role": "user", "content": prob["problem"]},
                            {"role": "assistant", "content": cot}]},
                                             ensure_ascii=False) + "\n")
                        n_total_kept += 1
        print(f"rationalization recovered: {n_solved_rationalize:,} additional problems  "
              f"({n_solved_rationalize / max(len(failed_problems_idx), 1):.1%} of failures)")

    sft.close()
    total = n_solved_forward + n_solved_rationalize
    print(f"TOTAL solved: {total:,}/{len(problems):,}  ({total / max(len(problems), 1):.1%})")
    print(f"SFT examples written: {n_total_kept:,} -> {args.out_sft}")


# ---------------------------------------------------------------------------
def _selftest():
    exs = [{"problem": "2 + 2 는 무엇입니까?",
            "solution": "단계 1: 2 + 2 = 4\n최종답: 4",
            "answer": "4"}]

    # forward prompt
    p = build_few_shot_prompt(exs, "3 + 5 는 무엇입니까?")
    assert "문제: 2 + 2" in p
    assert p.rstrip().endswith("풀이:")
    assert "3 + 5" in p

    # rationalize prompt: gold is interpolated
    pr = build_rationalize_prompt(exs, "3 + 5 는 무엇입니까?", "8")
    assert "정답이 8임을" in pr and pr.rstrip().endswith("풀이:")

    # completion parsing: cut at next problem boundary
    raw = ("단계 1: 3 + 5 = 8\n최종답: 8\n\n문제: 다음 문제\n풀이:\n다른 내용")
    parsed = parse_completion(raw)
    assert parsed.endswith("최종답: 8"), parsed
    # cut at exemplar delimiter
    raw2 = "단계 1: 8\n최종답: 8" + DELIM + "다음"
    assert parse_completion(raw2) == "단계 1: 8\n최종답: 8"
    # empty/no problem boundary
    assert parse_completion("단계 1: 8\n최종답: 8") == "단계 1: 8\n최종답: 8"

    # end-to-end with the math verifier: a synthetic "model output" that verifies
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "08_reasoning"))
    from verify_math import verify
    good_output = "단계 1: 3 + 5 = 8\n최종답: 8"
    assert verify(good_output, "8")
    assert not verify(good_output, "9")
    print("PASS bootstrap-from-base tests (prompts + parsing + verifier round-trip)")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
