#!/usr/bin/env python3
"""
Proves every shipped few-shot CoT exemplar is CORRECT: each exemplar's `solution`
must pass the matching verifier against its own `answer`. If an exemplar is wrong,
the base model imitates wrong reasoning — so these are checked, not assumed.

Run: python test_cot_exemplars.py
(needs ../08_reasoning on the path; the `code` check runs python in a subprocess.)
"""
import json, os, sys, inspect

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "08_reasoning"))

# exemplar file -> verifier name
FILES = {
    "exemplars_math_cot.json":   "math",
    "exemplars_code_cot.json":   "code",
    "exemplars_mcq_cot.json":    "mcq",
    "exemplars_format_cot.json": "format",
    "exemplars_logic_cot.json":  "logic",
}


def _verify_call(verify, solution, answer, ex):
    """Pass any extra kwarg the verifier accepts and the exemplar provides
    (e.g. n_options for mcq) — same dispatch as rejection_sample.py."""
    params = set(inspect.signature(verify).parameters)
    positional = {"model_output", "output", "response", "prediction",
                  "gold", "gold_answer", "answer", "tests", "puzzle", "problem"}
    extra = {k: ex[k] for k in (params - positional) if k in ex}
    return verify(solution, answer, **extra)


def main():
    import importlib
    total = ok = 0
    for fname, vname in FILES.items():
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} (not present)"); continue
        verify = importlib.import_module(f"verify_{vname}").verify
        exemplars = json.load(open(path, encoding="utf-8"))
        n_ok = 0
        for i, ex in enumerate(exemplars):
            assert {"problem", "solution", "answer"} <= set(ex), f"{fname}[{i}] missing fields"
            v = _verify_call(verify, ex["solution"], ex["answer"], ex)
            total += 1; ok += int(bool(v)); n_ok += int(bool(v))
            if not v:
                print(f"  FAIL {fname}[{i}] — solution does not verify against its answer")
        print(f"  {fname:<28} [{vname:<6}] {n_ok}/{len(exemplars)} exemplars verify")
    print(f"\n{'PASS' if ok == total else 'FAIL'}: {ok}/{total} CoT exemplars verify against their own gold")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
