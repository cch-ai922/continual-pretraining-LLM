#!/usr/bin/env python3
"""
Stage 5 — translate your 2M English SFT examples into Korean WITHOUT destroying the
markdown / LaTeX in the assistant responses, by reusing the structure-preserving
translator we built earlier (md_translate.py, vendored alongside this file).

Plug your MT system (e.g. an NLLB/IndicTrans model, or the bridge base model once
it can translate) into md_translate.translate_text. We translate user+assistant
turns; we KEEP code blocks, inline code, URLs, and all math intact.

Input/Output: chat JSONL, e.g. {"messages":[{"role":"user","content":...}, ...]}

After translating, ALWAYS run a language-ID pass (flag low-Korean outputs) before
training — weak MT silently leaves English chunks behind. (--lid-check)
"""
import argparse, json, sys
import md_translate


# ---- wire your translator here -------------------------------------------------
def _mt(text: str) -> str:
    """Replace with a real EN->KO translator call.
    Receives clean prose (md/LaTeX already masked by md_translate)."""
    raise NotImplementedError("Connect your EN->KO MT model in _mt().")


md_translate.translate_text = _mt
# -------------------------------------------------------------------------------

ROLES_TO_TRANSLATE = {"user", "assistant", "system"}


def korean_fraction(s: str) -> float:
    import unicodedata
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 1.0
    deva = sum(1 for c in letters if "HANGUL" in unicodedata.name(c, ""))
    return deva / len(letters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lid-check", action="store_true",
                    help="write low-Korean rows to <out>.flagged.jsonl instead of <out>")
    ap.add_argument("--min-korean", type=float, default=0.6)
    args = ap.parse_args()

    fout = open(args.out, "w", encoding="utf-8")
    fflag = open(args.out + ".flagged.jsonl", "w", encoding="utf-8") if args.lid_check else None
    n, flagged = 0, 0

    for line in open(args.inp, encoding="utf-8"):
        ex = json.loads(line)
        ok = True
        for msg in ex.get("messages", []):
            if msg.get("role") in ROLES_TO_TRANSLATE and msg.get("content"):
                msg["content"] = md_translate.translate_markdown(msg["content"])
                if args.lid_check and msg["role"] == "assistant":
                    if korean_fraction(msg["content"]) < args.min_korean:
                        ok = False
        out = json.dumps(ex, ensure_ascii=False) + "\n"
        if ok or not args.lid_check:
            fout.write(out); n += 1
        else:
            fflag.write(out); flagged += 1

    fout.close()
    if fflag:
        fflag.close()
        print(f"kept {n:,} | flagged low-Korean {flagged:,} -> {args.out}.flagged.jsonl",
              file=sys.stderr)
    print(f"wrote {n:,} translated SFT examples -> {args.out}")


if __name__ == "__main__":
    main()
