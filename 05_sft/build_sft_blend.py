#!/usr/bin/env python3
"""
Stage 5 — assemble the SFT blend.

Mix translated-English SFT with genuinely native Korean instruction data. Translated
data gives coverage/skills; native data fixes "translationese" and cultural fit.
Keep a small slice of ENGLISH SFT too so the model stays bilingual and instruction-
following in English (cheap insurance against forgetting at the SFT stage).

Recommended starting mix (per-example sampling):
    translated_ko : 70%   native_ko : 20%   english : 10%

Quality filters (applied BEFORE sampling; default ON for Korean pools):
    register consistency   drop rows whose assistant turn mixes 높임말/문체
                           (deterministic, via register_consistency.py)
    translationese (Tier-1) drop rows whose assistant turn is 번역투-shaped
                           (heuristic, via translationese_scorer.py;
                            applied ONLY to translated_ko by default — running
                            it on genuine native data would be overzealous)

English pool is never filtered (Korean checks don't apply).
Disable with --no-filter-register / --no-filter-translationese.
"""
import argparse, json, os, random, sys

# Sibling modules in 05_sft/. Import works whether the script is invoked from
# the project root or 05_sft/ directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from register_consistency import is_consistent, matches_target_register
from translationese_scorer import is_translationese


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if path else []


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
def _extract_assistant_text(row):
    """Pull the assistant turn from a chat-format row, or 'text' if flat."""
    if isinstance(row.get("text"), str):
        return row["text"]
    for m in row.get("messages", []):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def filter_korean_pool(pool, *, do_translationese, do_register,
                       register_threshold, register_min_sents,
                       register_target, translationese_min_signals):
    """Apply the Korean quality filters to a pool.

    Returns (kept_rows, stats_dict). The two filters compose: a row that fires
    EITHER drops it. Per-row reasons are counted so the driver can show the
    breakdown for tuning.
    """
    kept = []
    stats = {"input": len(pool),
             "dropped_register_mix": 0,
             "dropped_register_target": 0,
             "dropped_translationese": 0,
             "kept": 0}
    for row in pool:
        text = _extract_assistant_text(row)
        if do_register and text:
            if not is_consistent(text,
                                 threshold=register_threshold,
                                 min_sents=register_min_sents):
                stats["dropped_register_mix"] += 1
                continue
            if register_target and not matches_target_register(text, register_target):
                stats["dropped_register_target"] += 1
                continue
        if do_translationese and text:
            if is_translationese(text, min_signals=translationese_min_signals):
                stats["dropped_translationese"] += 1
                continue
        kept.append(row)
    stats["kept"] = len(kept)
    return kept, stats


def _print_pool_stats(label, stats, filters_applied):
    if stats["input"] == 0:
        print(f"  {label:<14} (empty)")
        return
    drop = stats["input"] - stats["kept"]
    pct = 100 * drop / stats["input"]
    print(f"  {label:<14} in={stats['input']:>8,}  kept={stats['kept']:>8,}  "
          f"dropped={drop:>6,} ({pct:.1f}%)  "
          f"[reg_mix={stats['dropped_register_mix']:,} "
          f"reg_target={stats['dropped_register_target']:,} "
          f"trans={stats['dropped_translationese']:,}]  "
          f"applied={filters_applied}")


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translated-ko", required=True)
    ap.add_argument("--native-ko", default=None)
    ap.add_argument("--english", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="0.70,0.20,0.10")
    ap.add_argument("--total", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=0)
    # Quality filters (default ON for Korean pools; disable per-filter)
    ap.add_argument("--no-filter-register", action="store_true",
                    help="Disable the 높임말/문체 consistency filter on KO pools")
    ap.add_argument("--no-filter-translationese", action="store_true",
                    help="Disable the 번역투 filter on the translated-KO pool")
    ap.add_argument("--filter-native-translationese", action="store_true",
                    help="Also apply 번역투 filter to native_ko pool (off by default)")
    ap.add_argument("--register-threshold", type=float, default=0.8,
                    help="Dominant-bucket fraction required for consistency (0..1)")
    ap.add_argument("--register-min-sents", type=int, default=2,
                    help="Rows with fewer register-bearing sentences are kept as-is")
    ap.add_argument("--register-target", default=None,
                    choices=[None, "합쇼체", "해요체", "문어체", "해체"],
                    help="If set, also drop rows whose dominant bucket differs")
    ap.add_argument("--translationese-min-signals", type=int, default=3,
                    help="Drop a row if it fires this many translationese signals (of 9)")
    args = ap.parse_args()

    w = [float(x) for x in args.weights.split(",")]
    raw_translated = load(args.translated_ko)
    raw_native     = load(args.native_ko)
    raw_english    = load(args.english)

    do_trans = not args.no_filter_translationese
    do_reg   = not args.no_filter_register

    print("=== Korean quality filters ===")

    # Translated KO: apply BOTH filters (the high-risk source).
    translated_filters = []
    if do_reg:   translated_filters.append("register")
    if do_trans: translated_filters.append("translationese")
    translated_pool, ts = filter_korean_pool(
        raw_translated,
        do_translationese=do_trans, do_register=do_reg,
        register_threshold=args.register_threshold,
        register_min_sents=args.register_min_sents,
        register_target=args.register_target,
        translationese_min_signals=args.translationese_min_signals,
    )
    _print_pool_stats("translated_ko", ts, translated_filters or ["none"])

    # Native KO: register-only by default (translationese filter risks dropping
    # genuine native idiosyncrasies; flip --filter-native-translationese if you
    # want belt-and-suspenders).
    native_filters = []
    if do_reg: native_filters.append("register")
    if args.filter_native_translationese: native_filters.append("translationese")
    native_pool, ns = filter_korean_pool(
        raw_native,
        do_translationese=args.filter_native_translationese,
        do_register=do_reg,
        register_threshold=args.register_threshold,
        register_min_sents=args.register_min_sents,
        register_target=args.register_target,
        translationese_min_signals=args.translationese_min_signals,
    )
    _print_pool_stats("native_ko", ns, native_filters or ["none"])

    # English: untouched
    if raw_english:
        print(f"  {'english':<14} in={len(raw_english):>8,}  "
              f"(English pool — Korean filters skipped)")

    # ----- blend -----
    rng = random.Random(args.seed)
    pools  = [translated_pool, native_pool, raw_english]
    labels = ["translated_ko", "native_ko", "english"]
    counts = [int(args.total * wi) for wi in w]

    print("\n=== blend ===")
    out = []
    for pool, c, label, wi in zip(pools, counts, labels, w):
        if not pool or c <= 0:
            if c > 0 and not pool:
                print(f"  {label:<14} weight={wi:.2f}  (pool empty after filtering — "
                      f"requested {c:,}, sampled 0)")
            continue
        out += [rng.choice(pool) for _ in range(c)]   # sample with replacement
        print(f"  {label:<14} weight={wi:.2f}  sampled={c:,}  pool_size={len(pool):,}")
    rng.shuffle(out)

    with open(args.out, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(out):,} SFT examples -> {args.out}")


# ---------------------------------------------------------------------------
def _selftest():
    # 1. Text extraction works for chat and flat formats.
    chat = {"messages": [{"role": "user", "content": "안녕"},
                         {"role": "assistant", "content": "안녕하세요. 반가워요."}]}
    assert _extract_assistant_text(chat) == "안녕하세요. 반가워요."
    assert _extract_assistant_text({"text": "그냥 텍스트입니다."}) == "그냥 텍스트입니다."
    assert _extract_assistant_text({"messages": [{"role": "user", "content": "x"}]}) == ""

    # 2. Build a tiny pool: consistent / register-mixed / translationese-heavy.
    consistent = {"messages": [
        {"role": "user", "content": "?"},
        {"role": "assistant", "content": "안녕하세요. 저는 학생이에요. 반가워요."}]}
    mixed_reg = {"messages": [
        {"role": "user", "content": "?"},
        {"role": "assistant", "content":
            "안녕하십니까. 저는 학생이에요. 만나서 반갑습니다."}]}
    trans_heavy = {"messages": [
        {"role": "user", "content": "?"},
        {"role": "assistant", "content":
            "그는 학교에 갔다. 그는 거기서 친구를 만났다. 그리고 그들은 함께 공부했다. "
            "그러나 그것은 그들의 친구의 부모의 책의 한 페이지였다. "
            "이 사실은 그에 의해 발견된 것이다. "
            "따라서 그는 그것을 그의 노트에 기록한 것이다."}]}
    pool = [consistent, mixed_reg, trans_heavy]

    # 3. Both filters ON: only the consistent native-style row survives.
    kept, stats = filter_korean_pool(
        pool, do_translationese=True, do_register=True,
        register_threshold=0.8, register_min_sents=2,
        register_target=None, translationese_min_signals=3,
    )
    assert stats["kept"] == 1, stats
    assert stats["dropped_register_mix"] == 1, stats
    assert stats["dropped_translationese"] == 1, stats
    assert kept[0] is consistent

    # 4. Register only — translationese row is register-CONSISTENT (all 문어체),
    #    so it survives this filter.
    kept, stats = filter_korean_pool(
        pool, do_translationese=False, do_register=True,
        register_threshold=0.8, register_min_sents=2,
        register_target=None, translationese_min_signals=3,
    )
    assert stats["kept"] == 2, stats
    assert stats["dropped_register_mix"] == 1
    assert stats["dropped_translationese"] == 0

    # 5. Translationese only — register-mixed row passes; trans_heavy drops.
    kept, stats = filter_korean_pool(
        pool, do_translationese=True, do_register=False,
        register_threshold=0.8, register_min_sents=2,
        register_target=None, translationese_min_signals=3,
    )
    assert stats["kept"] == 2, stats
    assert stats["dropped_register_mix"] == 0
    assert stats["dropped_translationese"] == 1

    # 6. Target-register: consistent row is 해요체. Demanding 합쇼체 drops it.
    kept, stats = filter_korean_pool(
        [consistent], do_translationese=False, do_register=True,
        register_threshold=0.8, register_min_sents=2,
        register_target="합쇼체", translationese_min_signals=3,
    )
    assert stats["kept"] == 0 and stats["dropped_register_target"] == 1, stats
    kept, stats = filter_korean_pool(
        [consistent], do_translationese=False, do_register=True,
        register_threshold=0.8, register_min_sents=2,
        register_target="해요체", translationese_min_signals=3,
    )
    assert stats["kept"] == 1 and stats["dropped_register_target"] == 0, stats

    # 7. Empty pool doesn't crash.
    kept, stats = filter_korean_pool(
        [], do_translationese=True, do_register=True,
        register_threshold=0.8, register_min_sents=2,
        register_target=None, translationese_min_signals=3,
    )
    assert stats == {"input": 0, "dropped_register_mix": 0,
                     "dropped_register_target": 0, "dropped_translationese": 0,
                     "kept": 0}

    # 8. Filters disabled = everything passes.
    kept, stats = filter_korean_pool(
        pool, do_translationese=False, do_register=False,
        register_threshold=0.8, register_min_sents=2,
        register_target=None, translationese_min_signals=3,
    )
    assert stats["kept"] == 3

    print("PASS build_sft_blend: chat+flat extraction, both filters on/off, "
          "target-register, empty pool, no-op mode")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
