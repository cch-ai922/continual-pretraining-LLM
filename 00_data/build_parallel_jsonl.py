#!/usr/bin/env python3
"""
Stage 0 — turn 25M EN-KO sentence pairs into pretraining documents.

Mixing parallel data as plain bilingual documents is the cheapest, most reliable
way to pull Korean and English representations into alignment (which is what
carries English knowledge into Korean). We emit a MIX of two formats so the model
sees translation both as continuation and as an instruction:

  (a) bilingual concatenation:   "<en text>\n<hi text>"   (and the reverse)
  (b) light instruction form:    "Translate to Korean:\n<en>\n\n<hi>"

Input: TSV or JSONL with en / hi fields.  Output: JSONL with a "text" field.
"""
import argparse, json, random

TEMPLATES = [
    "{en}\n{ko}",
    "{ko}\n{en}",
    "Translate to Korean:\n{en}\n\n{ko}",
    "다음의 영어문장을 조선어로 번역하시오.\n{en}\n\n{ko}",
    "Translate to English:\n{ko}\n\n{en}",
]


def read_pairs(path):
    if path.endswith(".jsonl"):
        for line in open(path, encoding="utf-8"):
            o = json.loads(line); yield o["en"], o["ko"]
    else:  # assume TSV: en<TAB>hi
        for line in open(path, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                yield parts[0], parts[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for en, ko in read_pairs(args.inp):
            en, ko = en.strip(), ko.strip()
            if not en or not ko:
                continue
            tmpl = rng.choice(TEMPLATES)
            f.write(json.dumps({"text": tmpl.format(en=en, ko=ko)}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n:,} parallel documents to {args.out}")


if __name__ == "__main__":
    main()
