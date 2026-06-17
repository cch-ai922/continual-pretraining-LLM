#!/usr/bin/env python3
"""
Stage 9 — EVALUATION harness (the missing safety net).

The whole project's thesis is "become fluent in Korean WITHOUT forgetting English",
and the README's own pitfalls checklist demands: "Evaluated English AND Korean every
checkpoint to catch forgetting early." This script is the tool that does it.

It is a SCORER over a predictions file (the model-generation step is a thin,
swappable driver — same pattern as the rest of the repo). Produce predictions with
your serving stack (vLLM snippet in the docstring below), then score them here.

What it reports
---------------
  * Korean fluency proxy : mean Korean fraction on the Korean slice
  * Code-switching rate  : % of Korean-expected responses that leak Latin script
  * Register consistency : mean 높임말/문체 consistency score on the Korean slice,
                           % of Korean responses that mix registers within a turn,
                           and the dominant-bucket breakdown (what register the
                           model is defaulting to). Mix-within-a-response is the
                           D6 failure mode the playbook flags as non-negotiable.
  * Verifiable accuracy  : dispatches to 08_reasoning/verify_<name>.py (math, mcq,
                           code, logic, format) when rows carry a gold answer
  * English regression   : the SAME metrics on rows tagged lang="en" — this is how
                           you SEE forgetting. Watch the en accuracy across
                           checkpoints; a drop is the alarm this whole pipeline
                           exists to prevent.

Predictions JSONL schema (one row per eval item)
------------------------------------------------
  {
    "prediction": "<model output text>",   # or "response"/"output"
    "lang": "ko" | "en",                    # optional, default "ko"
    "gold": <verifier-specific>,            # optional; enables accuracy
    "n_options": 4,                         # optional; for mcq
    "verifier": "math"                      # optional per-row; else use --verifier
  }

Generate predictions (vLLM) — sketch you adapt:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(MODEL)
    prompts = [tok.apply_chat_template([{"role":"user","content":r["prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
               for r in eval_rows]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=512))
    # write {"prediction": o.outputs[0].text, "lang":..., "gold":..., ...} per row

Run:
    python eval_report.py --in predictions.jsonl --verifier math
    python eval_report.py --in predictions.jsonl            # fluency/code-switch only
    python eval_report.py --selftest
"""
import argparse, importlib, inspect, json, os, sys, unicodedata

# Sibling-stage module (register check lives in 05_sft/). Works whether this
# script is run from project root or from 09_eval/ directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "05_sft"))
from register_consistency import consistency_score as _register_consistency_score


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE  (no model, no torch — runs with --selftest)
# ---------------------------------------------------------------------------
def _alpha(s):
    return [c for c in s if c.isalpha()]


def korean_fraction(s: str) -> float:
    """Hangul letters / all alphabetic letters. 1.0 if there are no letters
    (e.g. a pure-number math answer is not penalised)."""
    letters = _alpha(s)
    if not letters:
        return 1.0
    deva = sum(1 for c in letters if "HANGUL" in unicodedata.name(c, ""))
    return deva / len(letters)


def latin_fraction(s: str) -> float:
    """Basic-Latin letters / all alphabetic letters. High on the Korean slice is a
    code-switching signal (Roman-script Korean or untranslated English)."""
    letters = _alpha(s)
    if not letters:
        return 0.0
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return latin / len(letters)


def is_code_switched(s: str, min_korean: float = 0.6, max_latin: float = 0.15) -> bool:
    """Proxy: a Korean-expected response is code-switched if it is not mostly
    Hangul OR carries a non-trivial Latin-script share. Proper nouns/loanwords
    keep the threshold loose; calibrate on real data (see G0/G5 in the labour
    guide). Responses with no letters at all are never flagged."""
    if not _alpha(s):
        return False
    return korean_fraction(s) < min_korean or latin_fraction(s) > max_latin


def _get_pred(row):
    for k in ("prediction", "response", "output", "text"):
        if row.get(k):
            return row[k]
    return ""


def _load_verifier(name, search_dir):
    if search_dir not in sys.path:
        sys.path.insert(0, search_dir)
    return importlib.import_module(f"verify_{name}").verify


def _verify_one(verify, pred, gold, row):
    """Call verify(pred, gold, **extra) passing only the extra params it accepts
    (e.g. n_options for mcq) — mirrors rejection_sample.py's dispatch."""
    params = set(inspect.signature(verify).parameters)
    positional = {"model_output", "output", "response", "prediction",
                  "gold", "gold_answer", "answer", "tests", "puzzle", "problem"}
    extra = {k: row[k] for k in (params - positional) if k in row}
    return bool(verify(pred, gold, **extra))


def score_rows(rows, verify=None, default_verifier=None, verifier_fns=None,
               min_korean=0.6, max_latin=0.15,
               register_threshold=0.8, register_min_sents=2):
    """Aggregate metrics over rows. Pure function over a list of dicts.

    verifier_fns : optional {name: verify_callable} so per-row "verifier" works
                   without importing anything (used by --selftest).
    Returns a nested dict of metrics, split by language slice.

    Register-consistency metrics are computed for every row, but aggregated only
    over Korean-expected rows whose response has >= register_min_sents register-
    bearing sentences (English text never qualifies, so it sits out cleanly).
    """
    slices = {"ko": [], "en": [], "all": []}
    for row in rows:
        lang = row.get("lang", "ko")
        rec = dict(row)
        pred = _get_pred(row)
        rec["_korean_fraction"] = korean_fraction(pred)
        rec["_latin_fraction"] = latin_fraction(pred)
        rec["_code_switched"] = is_code_switched(pred, min_korean, max_latin)

        # register consistency (deterministic; rows without enough Korean
        # register-bearing sentences come back as n=0 -> dominant=None and are
        # excluded from the aggregate).
        r_score, r_dom, r_n = _register_consistency_score(pred)
        rec["_register_score"] = r_score
        rec["_register_dominant"] = r_dom
        rec["_register_n_sents"] = r_n
        rec["_register_consistent"] = (
            r_n < register_min_sents or r_score >= register_threshold
        )

        # accuracy (optional)
        rec["_verified"] = None
        if "gold" in row:
            vname = row.get("verifier", default_verifier)
            fn = None
            if verifier_fns and vname in verifier_fns:
                fn = verifier_fns[vname]
            elif verify is not None and (vname == default_verifier or vname is None):
                fn = verify
            if fn is not None:
                rec["_verified"] = _verify_one(fn, pred, row["gold"], row)

        slices["all"].append(rec)
        slices.setdefault(lang, []).append(rec)

    def agg(recs):
        if not recs:
            return None
        graded = [r["_verified"] for r in recs if r["_verified"] is not None]
        ko_expected = [r for r in recs if r.get("lang", "ko") == "ko"]
        # Register aggregates: only over ko_expected rows with enough register-
        # bearing Korean sentences to actually judge.
        reg_judgable = [r for r in ko_expected
                        if r["_register_n_sents"] >= register_min_sents]
        bucket_counts = {"합쇼체": 0, "해요체": 0, "문어체": 0, "해체": 0}
        for r in reg_judgable:
            d = r["_register_dominant"]
            if d in bucket_counts:
                bucket_counts[d] += 1
        return {
            "n": len(recs),
            "accuracy": (sum(graded) / len(graded)) if graded else None,
            "n_graded": len(graded),
            "mean_korean_fraction": (sum(r["_korean_fraction"] for r in ko_expected)
                                    / len(ko_expected)) if ko_expected else None,
            "code_switch_rate": (sum(r["_code_switched"] for r in ko_expected)
                                 / len(ko_expected)) if ko_expected else None,
            "n_register_judged": len(reg_judgable),
            "mean_register_score": (sum(r["_register_score"] for r in reg_judgable)
                                    / len(reg_judgable)) if reg_judgable else None,
            "register_mix_rate": (
                sum(1 for r in reg_judgable if not r["_register_consistent"])
                / len(reg_judgable)
            ) if reg_judgable else None,
            "register_buckets": bucket_counts,
        }

    return {"all": agg(slices["all"]),
            "ko": agg(slices.get("ko", [])),
            "en": agg(slices.get("en", [])),
            "_rows": slices["all"]}


def format_report(metrics):
    def line(name, m):
        if not m:
            return f"  {name:<8}: (no rows)"
        acc = f"{m['accuracy']:.1%}" if m["accuracy"] is not None else "  n/a"
        hf = f"{m['mean_korean_fraction']:.1%}" if m["mean_korean_fraction"] is not None else " n/a"
        cs = f"{m['code_switch_rate']:.1%}" if m["code_switch_rate"] is not None else " n/a"
        return (f"  {name:<8}: n={m['n']:<6} acc={acc:>6} (graded {m['n_graded']})"
                f"  korean={hf:>6}  code-switch={cs:>6}")

    def register_line(name, m):
        if not m or m.get("n_register_judged", 0) == 0:
            return f"  {name:<8}: register: (no judgable rows)"
        ms = f"{m['mean_register_score']:.2f}"
        mr = f"{m['register_mix_rate']:.1%}"
        # bucket profile: short labels for compactness in ASCII-only logs
        b = m["register_buckets"]
        bucket_total = sum(b.values()) or 1
        profile = "/".join(
            f"{label}={100*b[bucket]//bucket_total}%"
            for label, bucket in (("hapsyo", "합쇼체"), ("haeyo", "해요체"),
                                  ("muneo", "문어체"), ("hae", "해체"))
        )
        return (f"  {name:<8}: register: n_judged={m['n_register_judged']:<4}  "
                f"mean_score={ms}  mix_rate={mr:>6}  buckets[{profile}]")

    out = ["=== eval report ===",
           line("ALL", metrics["all"]),
           line("Korean", metrics["ko"]),
           line("English", metrics["en"]),
           "",
           register_line("Korean", metrics["ko"]),
           "",
           "  NOTE: track the English row across checkpoints — a falling English",
           "  accuracy is catastrophic forgetting, the failure this pipeline fights.",
           "  Register mix_rate trending up = SFT/DPO 높임말 discipline slipping;",
           "  buckets shifting = the model is changing its default register."]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="predictions JSONL")
    ap.add_argument("--verifier", default=None,
                    choices=[None, "math", "code", "mcq", "logic", "format"],
                    help="default verifier for rows that carry a gold answer")
    ap.add_argument("--min-korean", type=float, default=0.6)
    ap.add_argument("--max-latin", type=float, default=0.15)
    ap.add_argument("--register-threshold", type=float, default=0.8,
                    help="Dominant-bucket fraction required to call a response "
                         "register-consistent (default 0.8)")
    ap.add_argument("--register-min-sents", type=int, default=2,
                    help="Min register-bearing sentences for a response to be "
                         "judged (shorter responses are excluded from the aggregate)")
    ap.add_argument("--out", default=None, help="optional: per-row metrics JSONL")
    args = ap.parse_args()

    verify = None
    if args.verifier:
        verify = _load_verifier(args.verifier,
                                os.path.join(os.path.dirname(__file__), "..", "08_reasoning"))

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    metrics = score_rows(rows, verify=verify, default_verifier=args.verifier,
                         min_korean=args.min_korean, max_latin=args.max_latin,
                         register_threshold=args.register_threshold,
                         register_min_sents=args.register_min_sents)
    print(format_report(metrics))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in metrics["_rows"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nper-row metrics -> {args.out}")


# ---------------------------------------------------------------------------
def _selftest():
    # korean / latin fraction
    assert abs(korean_fraction("안녕 세계") - 1.0) < 1e-9
    assert abs(latin_fraction("hello world") - 1.0) < 1e-9
    assert korean_fraction("12345") == 1.0          # no letters -> not penalised
    assert latin_fraction("12345") == 0.0
    mix = "환자를 hospital 로 데려가라"               # code-switched
    assert 0.0 < korean_fraction(mix) < 1.0 and latin_fraction(mix) > 0.0

    # code-switch detection
    assert is_code_switched("answer 는 이것이다")      # English where Korean exists
    assert is_code_switched("mujhe nahi pata")       # Roman-script Korean
    assert not is_code_switched("이것은 순수한 조선어 문장이다.")
    assert not is_code_switched("42")                # pure number, never flagged

    # accuracy aggregation with a fake verifier + language slices
    fake = {"toy": lambda pred, gold: pred.strip() == str(gold)}
    rows = [
        {"prediction": "평양", "lang": "ko", "gold": "평양", "verifier": "toy"},
        {"prediction": "answer 는 이것 wrong", "lang": "ko", "gold": "정답", "verifier": "toy"},
        {"prediction": "Paris", "lang": "en", "gold": "Paris", "verifier": "toy"},
    ]
    m = score_rows(rows, verifier_fns=fake)
    assert m["all"]["n"] == 3 and m["ko"]["n"] == 2 and m["en"]["n"] == 1
    assert abs(m["ko"]["accuracy"] - 0.5) < 1e-9         # 1 of 2 Korean correct
    assert abs(m["en"]["accuracy"] - 1.0) < 1e-9         # English retained
    assert m["ko"]["code_switch_rate"] > 0.0             # row 2 is code-switched
    assert m["en"]["mean_korean_fraction"] is None        # no Korean-expected rows in en slice

    # real verifier round-trip via _verify_one (verify_math is pure/importable)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "08_reasoning"))
    try:
        vm = importlib.import_module("verify_math").verify
        assert _verify_one(vm, "최종답: 72", "72", {})
        assert not _verify_one(vm, "최종답: 71", "72", {})
        # mcq dispatch passes n_options
        vmcq = importlib.import_module("verify_mcq").verify
        assert _verify_one(vmcq, "정답: E", "E", {"n_options": 5})
        assert not _verify_one(vmcq, "정답: E", "E", {"n_options": 4})  # E invalid w/ 4 opts
    except ModuleNotFoundError:
        print("  (skipped real-verifier round-trip: 08_reasoning not on path)")

    # NEW — register-consistency aggregation. Five rows mix:
    #   1 consistent 해요체, 1 register-mixed (hapsyo+haeyo), 1 consistent 문어체,
    #   1 too-short Korean (excluded), 1 English (excluded).
    # Aggregates run only over Korean rows with >= min_sents register-bearing
    # sentences -- so n_register_judged == 3.
    reg_rows = [
        {"prediction": "안녕하세요. 저는 학생이에요. 만나서 반가워요.", "lang": "ko"},
        {"prediction": "안녕하십니까. 저는 학생이에요. 만나서 반갑습니다.", "lang": "ko"},
        {"prediction": "이것은 책이다. 저것은 펜이다. 그것은 종이이다.", "lang": "ko"},
        {"prediction": "평양", "lang": "ko"},
        {"prediction": "Paris is the capital.", "lang": "en"},
    ]
    rm = score_rows(reg_rows, register_threshold=0.8, register_min_sents=2)
    ko = rm["ko"]
    assert ko["n_register_judged"] == 3, ko
    # 1 of 3 judgable rows is register-mixed
    assert abs(ko["register_mix_rate"] - 1/3) < 1e-9, ko["register_mix_rate"]
    # mean score: 1.0 (consistent 해요체) + 2/3 (mixed) + 1.0 (consistent 문어체) = 2.667/3
    assert abs(ko["mean_register_score"] - (1.0 + 2/3 + 1.0) / 3) < 1e-6, ko["mean_register_score"]
    # dominant-bucket distribution: 1 해요체, 1 합쇼체 (mixed row's dominant), 1 문어체
    assert ko["register_buckets"] == {"합쇼체": 1, "해요체": 1, "문어체": 1, "해체": 0}, ko["register_buckets"]
    # English / too-short rows are excluded from the register aggregate
    assert rm["en"]["n_register_judged"] == 0

    # format_report must not crash on the augmented metrics
    s = format_report(rm)
    assert "register:" in s and "buckets[" in s, s

    # threshold knob: at threshold 0.6 the mixed row (score 0.667) now passes
    rm_loose = score_rows(reg_rows, register_threshold=0.6, register_min_sents=2)
    assert rm_loose["ko"]["register_mix_rate"] == 0.0, rm_loose["ko"]

    print("PASS all eval-harness tests (fractions + code-switch + accuracy + slices + dispatch + register)")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
