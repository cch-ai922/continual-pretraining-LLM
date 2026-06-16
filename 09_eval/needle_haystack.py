#!/usr/bin/env python3
"""
Stage 9 (long-context) — needle-in-a-haystack (NIAH) retrieval probe.

Qwen3-8B is native 32K. Continual-pretraining at seq_length=4096 (the default in
common_config.py) never exercises positions 4096..32768, so long-context ability
can silently erode even while loss looks fine. This probe MEASURES that, so you
catch it at a checkpoint instead of from users.

How NIAH works: bury a unique fact (the "needle") at a known DEPTH inside a long
filler "haystack" of a target TOKEN LENGTH, then ask the model to retrieve it.
Sweep length x depth, in BOTH Korean and English, every milestone checkpoint. A
cell that drops to ~0 means context is broken at that length/position.

Two modes (model generation is a thin driver you plug in — same as the rest of repo):
  BUILD : emit a probe set of prompts at each (length, depth) cell.
  SCORE : read your model's predictions and print the retrieval grid.

  python needle_haystack.py --build --lang ko --tokenizer ./qwen3-ko-base-hf \
      --lengths 4000,8000,16000,32000 --depths 0,0.25,0.5,0.75,1.0 --out probes_ko.jsonl
  python needle_haystack.py --build --lang en --tokenizer ./qwen3-ko-base-hf --out probes_en.jsonl
  # generate predictions for each probe (vLLM sketch below), then:
  python needle_haystack.py --score --in predictions.jsonl
  python needle_haystack.py --selftest

Generate predictions (vLLM), one row per probe — adapt:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); llm = LLM(MODEL, max_model_len=32768)
    prompts = [tok.apply_chat_template([{"role":"user","content":r["prompt"]}],
               tokenize=False, add_generation_prompt=True) for r in probes]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32))
    # write {"prediction": o.outputs[0].text, **{k:r[k] for k in
    #        ("answer","lang","context_tokens","depth")}} per row

Predictions JSONL schema (one row per probe):
    {"prediction": "...", "answer": "<needle value>", "lang": "ko"|"en",
     "context_tokens": 16000, "depth": 0.5}
"""
import argparse, json, random, re, sys


# ---------------------------------------------------------------------------
# CONTENT: original neutral filler + needle templates (hi / en)
# ---------------------------------------------------------------------------
FILLER = {
    "en": [
        "The river flows gently past the old stone bridge every morning.",
        "Farmers in the valley harvest wheat before the first rains arrive.",
        "A small library near the market stays open until late evening.",
        "Buses leave the station every hour and stop at the central square.",
        "Children play in the park while the bakery prepares fresh bread.",
        "The mountain trail is steep but offers a clear view of the lake.",
        "Engineers repaired the water pump after the long dry season.",
        "Traders set up their stalls along the road before sunrise.",
    ],
    "ko": [
        "강은 매일 아침 오래된 돌다리옆을 조용히 흐른다.",
        "계곡의 농부들은 첫 비가 오기전에 밀을 수확한다.",
        "시장근처의 작은 도서관은 늦은 저녁까지 문을 연다.",
        "뻐스는 매시간 역에서 출발하여 중앙광장에 선다.",
        "아이들이 공원에서 노는 동안 빵집은 갓 구운 빵을 만든다.",
        "산길은 가파롭지만 호수의 맑은 풍경을 보여 준다.",
        "긴 가물이 끝난 뒤 기술자들이 물뽐프를 수리했다.",
        "상인들은 해가 뜨기전에 길가에 매점을 차린다.",
    ],
}

NEEDLE = {
    "en": ("The secret access code for the archive vault is {value}.",
           "What is the secret access code for the archive vault? "
           "Reply with the code only."),
    "ko": ("기록보관소금고의 비밀접근코드는 {value}이다.",
           "기록보관소금고의 비밀접근코드는 무엇입니까? 코드만 쓰시오."),
}


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE  (no model / tokenizer needed — count_tokens is injected)
# ---------------------------------------------------------------------------
def _take_to_budget(sentences, budget, count_tokens, start=0):
    """Cycle through `sentences` (starting at offset `start`) accumulating until
    the next sentence would exceed `budget` tokens. Returns the chosen lines."""
    if budget <= 0 or not sentences:
        return []
    out, used, i = [], 0, start
    n = len(sentences)
    while True:
        s = sentences[i % n]
        c = count_tokens(s)
        if used + c > budget and out:        # keep at least nothing-over-budget
            break
        out.append(s); used += c; i += 1
        if used >= budget:
            break
    return out


def build_haystack(filler, needle, question, depth, target_tokens, count_tokens,
                   preamble="아래의 글을 읽고 질문에 답하시오."):
    """Build one NIAH prompt of ~target_tokens with `needle` placed at `depth`
    (0.0 = very top, 1.0 = just before the question). The needle is ALWAYS
    present and never truncated. Pure function over an injected token counter."""
    fixed = count_tokens(preamble) + count_tokens(needle) + count_tokens(question)
    budget = max(target_tokens - fixed, 0)
    before_budget = int(round(depth * budget))
    after_budget = budget - before_budget
    before = _take_to_budget(filler, before_budget, count_tokens, start=0)
    after = _take_to_budget(filler, after_budget, count_tokens, start=len(before))
    parts = ([preamble] if preamble else []) + before + [needle] + after + ["", question]
    return "\n".join(parts)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip().lower()


def retrieved(prediction: str, answer: str) -> bool:
    """Did the model surface the needle value? Substring match after stripping
    whitespace/case. Needle values are unique tokens, so this is robust."""
    return _norm(answer) in _norm(prediction)


def length_bucket(context_tokens, buckets=(4000, 8000, 16000, 32000)):
    """Snap an observed token length to the nearest reporting bucket."""
    return min(buckets, key=lambda b: abs(b - context_tokens))


def score_rows(rows):
    """Aggregate retrieval into a {lang: {bucket: {depth: acc}}} grid + overall."""
    grid, totals = {}, {}
    for r in rows:
        lang = r.get("lang", "ko")
        ok = retrieved(r.get("prediction", ""), r.get("answer", ""))
        b = length_bucket(r.get("context_tokens", 0))
        d = round(float(r.get("depth", 0.0)), 2)
        grid.setdefault(lang, {}).setdefault(b, {}).setdefault(d, [0, 0])
        cell = grid[lang][b][d]; cell[0] += int(ok); cell[1] += 1
        t = totals.setdefault(lang, [0, 0]); t[0] += int(ok); t[1] += 1
    # collapse [hits,total] -> accuracy
    out = {"by_lang": {}, "overall": {}}
    for lang, buckets in grid.items():
        out["by_lang"][lang] = {b: {d: (c[0] / c[1] if c[1] else None)
                                    for d, c in sorted(ds.items())}
                                for b, ds in sorted(buckets.items())}
    for lang, t in totals.items():
        out["overall"][lang] = t[0] / t[1] if t[1] else None
    return out


def format_grid(scored):
    lines = ["=== needle-in-a-haystack retrieval (accuracy) ==="]
    for lang in sorted(scored["by_lang"]):
        buckets = scored["by_lang"][lang]
        depths = sorted({d for ds in buckets.values() for d in ds})
        header = "  ".join(f"d={d:g}" for d in depths)
        lines.append(f"\n[{lang}]  overall={scored['overall'].get(lang, 0):.0%}")
        lines.append(f"  {'length':>8} | {header}")
        for b in sorted(buckets):
            cells = "  ".join(
                (f"{buckets[b][d]:.0%}" if buckets[b].get(d) is not None else "  -")
                .rjust(4) for d in depths)
            lines.append(f"  {b:>8} | {cells}")
    lines.append("\n  Read DOWN each column: if accuracy collapses past a length,")
    lines.append("  long-context is degrading there. Compare hi vs en slices.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
def _make_counter(tokenizer_path):
    if tokenizer_path:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        return lambda s: len(tok.encode(s, add_special_tokens=False))
    print("[warn] no --tokenizer: counting tokens by whitespace words "
          "(lengths approximate; pass your tokenizer for accurate buckets).",
          file=sys.stderr)
    return lambda s: len(s.split())


def cmd_build(args):
    rng = random.Random(args.seed)
    count_tokens = _make_counter(args.tokenizer)
    filler = FILLER[args.lang]
    if args.filler_file:
        filler = [ln.strip() for ln in open(args.filler_file, encoding="utf-8")
                  if ln.strip()] or filler
    ntmpl, qtmpl = NEEDLE[args.lang]
    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for L in lengths:
            for d in depths:
                for _ in range(args.per_cell):
                    value = str(rng.randint(10000, 99999))   # unique code
                    needle = ntmpl.format(value=value)
                    prompt = build_haystack(filler, needle, qtmpl, d, L, count_tokens)
                    f.write(json.dumps({"prompt": prompt, "answer": value,
                                        "lang": args.lang, "context_tokens": L,
                                        "depth": d}, ensure_ascii=False) + "\n")
                    n += 1
    print(f"wrote {n:,} NIAH probes ({args.lang}) -> {args.out}")


def cmd_score(args):
    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    print(format_grid(score_rows(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer for accurate token lengths")
    ap.add_argument("--filler-file", default=None, help="optional long real-text filler (one unit/line)")
    ap.add_argument("--lengths", default="4000,8000,16000,32000")
    ap.add_argument("--depths", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--per-cell", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--in", dest="inp", help="predictions JSONL (for --score)")
    ap.add_argument("--out", default="probes.jsonl")
    args = ap.parse_args()
    if args.score:
        cmd_score(args)
    elif args.build:
        cmd_build(args)
    else:
        ap.error("pass --build or --score (or --selftest)")


# ---------------------------------------------------------------------------
def _selftest():
    wc = lambda s: len(s.split())   # fake token counter: 1 token per word

    # 1. needle always present; depth=0 -> needle near top, depth=1 -> needle last
    needle = "CODE 74531 HERE"
    q = "what is the code ?"
    filler = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
    p_top = build_haystack(filler, needle, q, 0.0, 60, wc, preamble="read this")
    p_bot = build_haystack(filler, needle, q, 1.0, 60, wc, preamble="read this")
    assert needle in p_top and needle in p_bot
    lines_top = [l for l in p_top.split("\n") if l]
    lines_bot = [l for l in p_bot.split("\n") if l]
    # depth 0: needle is the line right after the preamble (no filler before it)
    assert lines_top[1] == needle, lines_top[:3]
    # depth 1: needle is the last content line before the question
    assert lines_bot[-2] == needle and lines_bot[-1] == q, lines_bot[-3:]

    # 2. length is controlled (~target within one filler unit)
    p_mid = build_haystack(filler, needle, q, 0.5, 60, wc, preamble="read this")
    assert 45 <= wc(p_mid) <= 75, wc(p_mid)

    # 3. depth 0.5 puts roughly half the filler before the needle
    nl = [l for l in p_mid.split("\n") if l]
    idx = nl.index(needle)
    before = idx - 1  # minus preamble
    after = len(nl) - idx - 2  # minus needle + question
    assert abs(before - after) <= 2, (before, after)

    # 4. retrieved(): exact, whitespace/case tolerant, and true negatives
    assert retrieved("The code is 74531.", "74531")
    assert retrieved("정답: 74531", "74531")
    assert retrieved("  7 4 5 3 1  ".replace(" ", ""), "74531")
    assert not retrieved("the code is 99999", "74531")
    assert not retrieved("I don't know", "74531")

    # 5. length bucketing snaps to nearest reporting bucket
    assert length_bucket(15500) == 16000 and length_bucket(4100) == 4000

    # 6. score grid aggregates by lang x length x depth and computes accuracy
    rows = [
        {"prediction": "74531", "answer": "74531", "lang": "ko", "context_tokens": 4000, "depth": 0.5},
        {"prediction": "nope",  "answer": "11111", "lang": "ko", "context_tokens": 4000, "depth": 0.5},
        {"prediction": "22222", "answer": "22222", "lang": "en", "context_tokens": 32000, "depth": 1.0},
    ]
    g = score_rows(rows)
    assert abs(g["by_lang"]["hi"][4000][0.5] - 0.5) < 1e-9    # 1 of 2 retrieved
    assert g["by_lang"]["en"][32000][1.0] == 1.0
    assert abs(g["overall"]["hi"] - 0.5) < 1e-9 and g["overall"]["en"] == 1.0
    _ = format_grid(g)   # must not raise

    print("PASS all needle-in-a-haystack tests (build + depth + length + score grid)")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
