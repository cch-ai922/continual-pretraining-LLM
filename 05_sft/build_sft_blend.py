#!/usr/bin/env python3
"""
Stage 5 — assemble the SFT blend.

Mix translated-English SFT with genuinely native Korean instruction data. Translated
data gives coverage/skills; native data fixes "translationese" and cultural fit.
Keep a small slice of ENGLISH SFT too so the model stays bilingual and instruction-
following in English (cheap insurance against forgetting at the SFT stage).

Recommended starting mix (per-example sampling):
    translated_ko : 70%   native_ko : 20%   english : 10%
"""
import argparse, json, random


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if path else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translated-ko", required=True)
    ap.add_argument("--native-ko", default=None)
    ap.add_argument("--english", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="0.70,0.20,0.10")
    ap.add_argument("--total", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    w = [float(x) for x in args.weights.split(",")]
    pools = [load(args.translated_ko), load(args.native_ko), load(args.english)]
    rng = random.Random(args.seed)

    counts = [int(args.total * wi) for wi in w]
    out = []
    for pool, c in zip(pools, counts):
        if not pool or c <= 0:
            continue
        out += [rng.choice(pool) for _ in range(c)]   # sample with replacement
    rng.shuffle(out)

    with open(args.out, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(out):,} SFT examples (mix {args.weights}) -> {args.out}")


if __name__ == "__main__":
    main()
