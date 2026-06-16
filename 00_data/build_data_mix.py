#!/usr/bin/env python3
"""
Stage 0 — turn corpus sizes + a token budget into Megatron data blend(s).

Two blends:
  * MAIN (seq 4096): the bulk continual-pretrain mix (Stage 4b). Default.
  * LONG (seq 32768): a small long-context RETENTION mix (Stage 4c), emitted only
    when --long-budget-b > 0. Qwen3-8B is native 32K; training only at 4096 leaves
    positions 4096..32768 with zero gradient, so long-context erodes. The long
    bucket exercises those positions on GENUINELY long documents.

This script computes the weights, prints per-source epoch counts (sanity-check for
over-repetition), and writes blend.yaml (and blend_long.yaml if requested).
"""
import argparse, yaml

# --- MAIN bucket (seq 4096). Edit to your real preprocessed prefixes (no .bin/.idx) ---
SOURCES = {
    "korean":    {"path": "/data/mcore/ko_mono_text_document",        "tokens_b": 20.0},
    "english":  {"path": "/data/mcore/en_mono_text_document",        "tokens_b": 400.0},
    "parallel": {"path": "/data/mcore/parallel_enko_text_document",  "tokens_b": 1.2},
}
DEFAULT_WEIGHTS = {"korean": 0.45, "english": 0.45, "parallel": 0.10}

# --- LONG bucket (seq 32768). ONLY genuinely long documents (books, long articles,
# theme-coherent concatenations). Parallel sentence pairs are short -> not useful
# here. English-long is REPLAY to keep English long-context from regressing. ---
LONG_SOURCES = {
    "korean_long":   {"path": "/data/mcore/ko_long_text_document",  "tokens_b": 3.0},
    "english_long": {"path": "/data/mcore/en_long_text_document",  "tokens_b": 50.0},
}
DEFAULT_LONG_WEIGHTS = {"korean_long": 0.60, "english_long": 0.40}


def _parse_weights(s, default):
    w = dict(default)
    if s:
        for kv in s.split(","):
            k, v = kv.split("="); w[k.strip()] = float(v)
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}     # normalize


def compute_blend(sources, weights, budget_b, label):
    print(f"\n=== {label} blend (budget {budget_b:g}B tokens) ===")
    print(f"{'source':<14}{'weight':>8}{'tokens(B)':>12}{'epochs':>9}")
    flat = []
    for name, w in weights.items():
        seen = budget_b * w
        epochs = seen / sources[name]["tokens_b"]
        flag = "  <-- high, watch overfitting" if epochs > 4 else ""
        print(f"{name:<14}{w:>8.3f}{seen:>12.2f}{epochs:>9.2f}{flag}")
        flat += [round(w, 4), sources[name]["path"]]
    structured = [[sources[n]["path"], weights[n]] for n in weights]
    return flat, structured


def write_yaml(path, budget_b, weights, flat, structured, seq_length):
    cfg = {
        "total_tokens_b": budget_b,
        "seq_length": seq_length,
        "weights": weights,
        "data_path": flat,                                   # Megatron --data-path style
        "blend": structured,                                 # Bridge-style structured blend
    }
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    print(f"wrote {path}")
    print("flat data_path:", " ".join(str(x) for x in flat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-b", type=float, default=60.0, help="MAIN (4096) continual tokens (B)")
    ap.add_argument("--out", default="blend.yaml")
    ap.add_argument("--weights", default=None,
                    help="MAIN override, e.g. 'korean=0.45,english=0.45,parallel=0.10'")
    # long-context retention bucket (Stage 4c) -- off unless a budget is given
    ap.add_argument("--long-budget-b", type=float, default=0.0,
                    help="LONG (32768) retention tokens (B); 0 disables. Try ~3-5B.")
    ap.add_argument("--long-out", default="blend_long.yaml")
    ap.add_argument("--long-weights", default=None,
                    help="LONG override, e.g. 'korean_long=0.6,english_long=0.4'")
    args = ap.parse_args()

    w = _parse_weights(args.weights, DEFAULT_WEIGHTS)
    flat, structured = compute_blend(SOURCES, w, args.budget_b, "MAIN (seq 4096)")
    write_yaml(args.out, args.budget_b, w, flat, structured, seq_length=4096)

    if args.long_budget_b > 0:
        lw = _parse_weights(args.long_weights, DEFAULT_LONG_WEIGHTS)
        lflat, lstructured = compute_blend(LONG_SOURCES, lw, args.long_budget_b,
                                           "LONG (seq 32768, retention)")
        write_yaml(args.long_out, args.long_budget_b, lw, lflat, lstructured,
                   seq_length=32768)
        print("\nNOTE: feed blend_long.yaml to 04_pretrain/stage2c_longctx.py. The long "
              "bucket MUST be genuinely long docs AND run with position/attention reset "
              "DISABLED, or you train short-context at a long seq_length (no benefit).")


if __name__ == "__main__":
    main()
