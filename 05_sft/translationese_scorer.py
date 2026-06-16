#!/usr/bin/env python3
"""
Stage 5 (filter) — Tier-1 heuristic 번역투 (translationese) detector for Korean.

번역투 = Korean text whose surface betrays it was translated from English.
The signals are morphological and statistical — fully deterministic, no model,
no labels needed at runtime. (~200 hand-labeled examples are *optional* for
recalibration; sensible defaults work without them.)

This is the cheap precision filter to put between
  05_sft/translate_en_sft_to_ko.py     (the source of most 번역투 risk)
and
  05_sft/build_sft_blend.py            (which builds the final SFT pool).

Nine features measured (each higher = more translationese):

  1.  pronoun_per_100c    Overt-pronoun density (그/그녀/그것/그들/이것 + 은/는/...).
                          Native Korean drops these via zero-anaphora; English
                          forces explicit pronoun copying through translation.
  2.  passive_per_100c    `에 의해 / 에 의하여 / 에 의한` — direct calque of
                          English passive voice. Native Korean uses verb-internal
                          passive (-되다, -히다, -리다, -기다) much more often.
  3.  deul_per_100c       Plural-marker `들` density on non-human nouns.
                          English `-s` forces plural marking Korean usually drops.
  4.  connective_per_sent Standalone connective overuse (그리고/그러나/하지만/
                          그래서/따라서/그러므로) at sentence start. Native
                          Korean prefers conjunctive endings (-고/-지만/-니까).
  5.  thing_end_ratio     Sentences ending in `것이다` / `것입니다` / `것이에요`.
                          Formal translated Korean overuses this nominalizer.
  6.  dem_dep_per_100c    Demonstrative + dependent-noun density
                          (이 것 / 그 곳 / 저 때 / 그 사람). English the/this/that
                          calque.
  7.  chain_deep_count    Count of NPs with THREE consecutive `의` joins
                          (`친구의 부모의 집의 ...`). English NP nesting copied
                          structurally.
  8.  chain_med_per_100c  Density of TWO-consecutive `의` joins (weaker signal).
  9.  sent_len_mean       Mean sentence length in chars. Translationese runs
                          long because English sentence structure carries over.

A text "fires" on a feature if its value exceeds the per-feature threshold.
A text is flagged as 번역투 if `n_fired >= --min-signals` (default 3 of 9).
This errs on precision: it's better to drop borderline native text than to
keep clear 번역투 in the SFT pool.

  python translationese_scorer.py --in translated_sft.jsonl --out clean.jsonl
  python translationese_scorer.py --calibrate \
      --native native_ko.jsonl --translated mt_ko.jsonl \
      --save-thresholds th.json
  python translationese_scorer.py --selftest
"""
import argparse, json, re, sys


# ---------------------------------------------------------------------------
# FEATURE EXTRACTORS — small, fast, pure functions.
# ---------------------------------------------------------------------------
_PRONOUN_RE = re.compile(
    r"(?:그|그녀|그들|그것|이것|저것)(?:은|는|이|가|을|를|의|에게|에|도)\b|"
    r"(?:그|그녀|그들|그것|이것|저것)(?:은|는|이|가|을|를|의|에게|에|도)(?=\s)"
)
# The double-form catches both "그는" before whitespace and "그는." before punct.
# To keep it simple we also accept end-of-string via \b (Python \b is unicode-aware
# enough for Hangul boundaries when adjacent to non-letters).

_PASSIVE_RE = re.compile(r"에\s*의해서?\b|에\s*의하여\b|에\s*의한\b")

_DEUL_RE = re.compile(r"들(?:이|은|을|를|의|에서|에게|에|로|과|와)\b")

_CONN_RE = re.compile(
    r"(?:^|[.!?。…]\s*)(그리고|그러나|하지만|그래서|따라서|그러므로|"
    r"이로\s*인해|그런데도)"
)

_THING_END_RE = re.compile(
    r"것(?:이다|입니다|이었다|이었습니다|이에요|예요|입니까|이라|이라고)"
    r"[\.\?!]?\s*$",
    re.M,
)

_DEM_DEP_RE = re.compile(
    r"(?:^|[\s\(])(?:이|그|저)\s+(?:것|곳|때|사람|등|들|이|점|분)"
    r"(?:은|는|이|가|을|를|의|에|에서|로|과|와|도|만)?\b"
)

_CHAIN_DEEP_RE = re.compile(
    r"[가-힣]{1,8}의\s+[가-힣]{1,8}의\s+[가-힣]{1,8}의"
)
_CHAIN_MED_RE = re.compile(
    r"[가-힣]{1,8}의\s+[가-힣]{1,8}의"
)


def _split_sentences(text):
    parts = re.split(r"[.!?。…\n]+", text)
    return [p.strip() for p in parts if p.strip()]


FEATURE_NAMES = (
    "pronoun_per_100c",
    "passive_per_100c",
    "deul_per_100c",
    "connective_per_sent",
    "thing_end_ratio",
    "dem_dep_per_100c",
    "chain_deep_count",
    "chain_med_per_100c",
    "sent_len_mean",
)


def features(text):
    """Return a dict of the 9 features for one text. Empty input -> all zeros."""
    n_chars = max(len(text), 1)
    sents = _split_sentences(text)
    n_sents = max(len(sents), 1)
    sent_lens = [len(s) for s in sents] or [0]

    n_thing = sum(1 for s in sents if _THING_END_RE.search(s + "."))

    return {
        "pronoun_per_100c":    100.0 * len(_PRONOUN_RE.findall(text)) / n_chars,
        "passive_per_100c":    100.0 * len(_PASSIVE_RE.findall(text)) / n_chars,
        "deul_per_100c":       100.0 * len(_DEUL_RE.findall(text)) / n_chars,
        "connective_per_sent": len(_CONN_RE.findall(text)) / n_sents,
        "thing_end_ratio":     n_thing / n_sents,
        "dem_dep_per_100c":    100.0 * len(_DEM_DEP_RE.findall(text)) / n_chars,
        "chain_deep_count":    float(len(_CHAIN_DEEP_RE.findall(text))),
        "chain_med_per_100c":  100.0 * len(_CHAIN_MED_RE.findall(text)) / n_chars,
        "sent_len_mean":       sum(sent_lens) / n_sents,
    }


# Default thresholds — sensible starting points derived from typical native
# Korean prose statistics. Recalibrate with --calibrate on your own paired
# native/translated samples to tighten precision/recall on your data.
# Each threshold = "above this is suspicious."
DEFAULT_THRESHOLDS = {
    "pronoun_per_100c":     0.6,   # ~6 pronouns per 1000 chars
    "passive_per_100c":     0.25,  # ~2-3 instances of 에 의해 per 1000 chars
    "deul_per_100c":        0.8,
    "connective_per_sent":  0.35,  # > 1 in 3 sentences starts with 그리고/그러나/...
    "thing_end_ratio":      0.30,  # 30%+ of sentences end in 것이다
    "dem_dep_per_100c":     0.8,
    "chain_deep_count":     0.5,   # any 3-deep 의-chain is a strong signal
    "chain_med_per_100c":   0.4,
    "sent_len_mean":        80.0,
}


def composite_score(feats, thresholds=None):
    """Return (n_fired, list_of_fired_feature_names)."""
    th = thresholds or DEFAULT_THRESHOLDS
    fired = [k for k in FEATURE_NAMES if feats[k] > th[k]]
    return len(fired), fired


def is_translationese(text, min_signals=3, thresholds=None):
    n, _ = composite_score(features(text), thresholds)
    return n >= min_signals


# ---------------------------------------------------------------------------
# CALIBRATION — set thresholds from a native vs translated reference pair.
# ---------------------------------------------------------------------------
def _percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = int(p * (len(sorted_vals) - 1))
    return sorted_vals[k]


def calibrate(native_texts, translated_texts, p_native=0.95):
    """For each feature, set threshold = p95 of native.

    Print a per-feature separation report so you can spot features that don't
    actually discriminate on your data and drop them if needed.
    """
    n_feats = [features(t) for t in native_texts if t]
    t_feats = [features(t) for t in translated_texts if t]
    out = {}
    print(f"{'feature':<22} {'native_p95':>12} {'trans_median':>14}  separation")
    print("-" * 64)
    for k in FEATURE_NAMES:
        nv = sorted(f[k] for f in n_feats)
        tv = sorted(f[k] for f in t_feats)
        p95_n = _percentile(nv, p_native)
        med_t = _percentile(tv, 0.5)
        out[k] = round(p95_n, 4)
        if p95_n > 0 and med_t > 2 * p95_n:
            sep = "STRONG"
        elif med_t > p95_n:
            sep = "weak"
        else:
            sep = "NONE"
        print(f"{k:<22} {p95_n:>12.4f} {med_t:>14.4f}  {sep}")
    return out


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
def _pick_text(row, mode):
    if mode == "text":
        return row.get("text", "")
    if mode == "assistant":
        for m in row.get("messages", []):
            if m.get("role") == "assistant":
                return m.get("content", "")
        return ""
    if "text" in row:
        return row["text"]
    for m in row.get("messages", []):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def _load_jsonl_texts(path, field):
    out = []
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        t = _pick_text(row, field)
        if t:
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", help="JSONL with {text} OR {messages}")
    ap.add_argument("--out", default=None,
                    help="If given, write only kept (non-translationese) rows")
    ap.add_argument("--field", default="auto",
                    choices=["auto", "text", "assistant"])
    ap.add_argument("--min-signals", type=int, default=3,
                    help="Flag as translationese if at least this many features "
                         "fire (1..9; default 3 is a precision-friendly choice)")
    ap.add_argument("--calibrate", action="store_true",
                    help="Compute per-feature thresholds from --native + "
                         "--translated reference JSONLs and exit")
    ap.add_argument("--native", help="JSONL of clean native Korean")
    ap.add_argument("--translated", help="JSONL of known MT Korean")
    ap.add_argument("--save-thresholds")
    ap.add_argument("--load-thresholds",
                    help="JSON file produced by --save-thresholds; overrides defaults")
    args = ap.parse_args()

    if args.calibrate:
        if not (args.native and args.translated):
            print("--calibrate requires --native and --translated", file=sys.stderr)
            sys.exit(2)
        nat = _load_jsonl_texts(args.native, args.field)
        trn = _load_jsonl_texts(args.translated, args.field)
        th = calibrate(nat, trn)
        if args.save_thresholds:
            json.dump(th, open(args.save_thresholds, "w", encoding="utf-8"),
                      indent=2, ensure_ascii=False)
            print(f"-> wrote thresholds to {args.save_thresholds}")
        return

    if not args.inp:
        print("--in is required (unless using --calibrate)", file=sys.stderr)
        sys.exit(2)

    th = (json.load(open(args.load_thresholds, encoding="utf-8"))
          if args.load_thresholds else DEFAULT_THRESHOLDS)

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    n_total = n_flagged = 0
    fire_counts = {k: 0 for k in FEATURE_NAMES}

    for row in rows:
        text = _pick_text(row, args.field)
        if not text:
            continue
        n_total += 1
        feats = features(text)
        n_fired, fired = composite_score(feats, th)
        for f in fired:
            fire_counts[f] += 1
        if n_fired >= args.min_signals:
            n_flagged += 1
        elif out_f:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if out_f:
        out_f.close()

    rate = 100 * n_flagged / max(n_total, 1)
    print(f"total examples     : {n_total:,}")
    print(f"flagged (>= {args.min_signals} signals): {n_flagged:,} ({rate:.1f}%)")
    print("per-feature fire counts:")
    for k, v in sorted(fire_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<22}: {v:,}")
    if args.out:
        print(f"-> kept rows written to {args.out}")


# ---------------------------------------------------------------------------
def _selftest():
    # 1. Natural Korean — most features should be near zero.
    natural = (
        "조선어를 배우는 것은 즐겁다. 매일 조금씩 공부하면 실력이 늘어난다. "
        "친구와 자연스럽게 대화하면서 표현을 익히는 편이 좋다."
    )
    f = features(natural)
    n, fired = composite_score(f)
    assert n <= 1, (n, fired, f)
    assert not is_translationese(natural)

    # 2. Heavy translationese — pronouns + passive + 의-chain + 것이다 + connectives
    translated = (
        "그는 학교에 갔다. 그는 거기서 친구를 만났다. 그리고 그들은 함께 공부했다. "
        "그러나 그것은 그들의 친구의 부모의 책의 한 페이지였다. "
        "이 사실은 그에 의해 발견된 것이다. "
        "따라서 그는 그것을 그의 노트에 기록한 것이다."
    )
    f = features(translated)
    n, fired = composite_score(f)
    assert n >= 4, (n, fired, f)
    assert is_translationese(translated)

    # 3. Per-feature triggers — each must fire on a designed minimal example.

    # pronouns
    f = features("그는 갔다. 그녀는 왔다. 그들은 만났다. 그것은 책이다.")
    assert f["pronoun_per_100c"] > DEFAULT_THRESHOLDS["pronoun_per_100c"], f

    # passive 에 의해
    f = features("이 결정은 위원회에 의해 내려졌다. 보고서는 그에 의해 작성되었다.")
    assert f["passive_per_100c"] > DEFAULT_THRESHOLDS["passive_per_100c"], f

    # 의-chain deep (3 of 의 in a row)
    f = features("그는 친구의 부모의 집의 문 앞에 섰다.")
    assert f["chain_deep_count"] >= 1, f

    # standalone connectives
    f = features("비가 왔다. 그리고 바람이 불었다. 그러나 우리는 갔다. 따라서 늦었다.")
    assert f["connective_per_sent"] > DEFAULT_THRESHOLDS["connective_per_sent"], f

    # 것이다 endings
    f = features("그것은 책이다. 그가 한 것이다. 우리가 만든 것이다.")
    assert f["thing_end_ratio"] >= 0.5, f

    # demonstrative + dependent noun
    f = features("이 것은 좋다. 그 곳에 갔다. 저 때에 만났다. 그 사람을 보았다.")
    assert f["dem_dep_per_100c"] > DEFAULT_THRESHOLDS["dem_dep_per_100c"], f

    # 4. Calibration sanity — should produce per-feature numbers and not crash
    th = calibrate([natural] * 5, [translated] * 5)
    assert set(th.keys()) == set(FEATURE_NAMES)
    # Native p95 should be lower than the translated text's value on at least
    # the most-discriminating features.
    f_trn = features(translated)
    discriminators = ["pronoun_per_100c", "thing_end_ratio", "passive_per_100c"]
    for k in discriminators:
        assert f_trn[k] > th[k], (k, f_trn[k], th[k])

    # 5. Chat-format extraction
    chat = {"messages": [{"role": "user", "content": "x"},
                         {"role": "assistant", "content": translated}]}
    assert _pick_text(chat, "auto") == translated
    assert _pick_text(chat, "assistant") == translated

    # 6. min-signals knob — looser threshold catches borderline cases
    borderline = "그는 학교에 갔다. 그것은 친구의 부모의 가방의 일부였다."
    f = features(borderline)
    n, _ = composite_score(f)
    assert n >= 2, (n, f)              # at least pronoun + chain fire
    # default min_signals=3 might not flag; loosened min_signals=2 should
    assert is_translationese(borderline, min_signals=2)

    # 7. Empty / very-short input doesn't crash
    f = features("")
    assert all(v == 0 for v in f.values())
    f = features("네.")
    n, _ = composite_score(f)
    assert n == 0

    print("PASS translationese_scorer: natural vs translationese + 6 per-feature "
          "triggers + calibrate + chat-format + min-signals + empty-input")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
