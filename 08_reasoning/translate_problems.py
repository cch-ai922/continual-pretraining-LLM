#!/usr/bin/env python3
"""
Stage 8 — translate English math problems to Korean for RLVR.

Translates ONLY the problem statement; the gold numerical answer is language-
independent and passes through unchanged. Uses the markdown/LaTeX-preserving
translator from 05_sft so equations, units, and code blocks survive.

Input  JSONL : {"problem": "...", "gold": ...}        (from converters/*.py)
                 (+ any verifier-specific fields like "n_options" for mcq)
Output JSONL : {"problem": "...조선어...", "gold": ...} (problem translated; every
                 other field — gold, n_options, code tests, etc. — passes through
                 UNCHANGED, because correctness is language-independent)
"""
import argparse, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "05_sft"))
import md_translate


# ---- wire your EN->KO translator here ---------------------------------------
def _mt(text: str) -> str:
    """Replace with a real EN->KO MT call (NLLB / vLLM / API). Receives
    clean prose; md_translate already masked numbers in code/LaTeX/inline-math."""
    raise NotImplementedError("Connect your EN->KO MT model in _mt().")


md_translate.translate_text = _mt
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="JSONL: {problem, gold, ...}  (from converters/*.py)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for line in open(args.inp, encoding="utf-8"):
            r = json.loads(line)
            # Translate ONLY the problem statement. Keep every other field
            # (gold, n_options, tests, timeout, ...) verbatim — the verifier
            # checks correctness, which does not depend on the language.
            r["problem"] = md_translate.translate_markdown(r["problem"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"translated {n:,} problems -> {args.out}")


if __name__ == "__main__":
    main()
