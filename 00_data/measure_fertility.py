#!/usr/bin/env python3
"""
Stage 0 — measure how well Qwen3's existing 151k tokenizer handles Korean.

Decision rule: if the base tokenizer's fertility (tokens per word) on Korean is
close to a dedicated Korean tokenizer (say within ~1.4x), DO NOT bother extending
the vocab. Embedding surgery on a hybrid + MTP + multimodal model is the riskiest
part of this whole project, so only pay that cost if the numbers justify it.

Usage:
    python measure_fertility.py --model Qwen/Qwen3-8B --korean-text data/ko_sample.txt
    # optionally compare against a freshly trained Korean SPM:
    python measure_fertility.py --model Qwen/Qwen3-8B --korean-text data/ko_sample.txt \
        --compare-bpe ko_bpe.json
"""
import argparse, unicodedata


def basic_word_count(text: str) -> int:
    # whitespace words; fine for a relative fertility metric in Hangul
    return sum(1 for w in text.split() if w.strip())


def hangul_char_count(text: str) -> int:
    return sum(1 for ch in text if "HANGUL" in unicodedata.name(ch, ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or path of base model/tokenizer")
    ap.add_argument("--korean-text", required=True, help="UTF-8 file of representative Korean text")
    ap.add_argument("--compare-bpe", default=None,
                    help="optional ko_bpe.json (byte-level BPE) to compare against")
    ap.add_argument("--max-chars", type=int, default=2_000_000)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    text = open(args.korean_text, encoding="utf-8").read()[: args.max_chars]
    n_words = basic_word_count(text)
    n_chars = len(text)
    n_hangul = hangul_char_count(text)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tok.encode(text, add_special_tokens=False)
    n_tok = len(ids)

    print(f"=== Korean fertility on {args.model} ===")
    print(f"chars               : {n_chars:,}  (Hangul: {n_hangul:,})")
    print(f"words (whitespace)  : {n_words:,}")
    print(f"base tokens         : {n_tok:,}")
    print(f"tokens / word       : {n_tok / max(n_words,1):.3f}")
    print(f"tokens / char       : {n_tok / max(n_chars,1):.3f}")

    # how often Korean falls back to single-byte tokens (a bad sign)
    pieces = tok.convert_ids_to_tokens(ids[: min(len(ids), 200000)])
    byteish = sum(1 for p in pieces if len(p) <= 2)  # rough: very short pieces
    print(f"very-short pieces   : {byteish/len(pieces):.1%} of a sample "
          f"(high % => poor Korean support => extension likely worth it)")

    if args.compare_bpe:
        from tokenizers import Tokenizer
        hi = Tokenizer.from_file(args.compare_bpe)
        n_tok2 = len(hi.encode(text).ids)
        ratio = (n_tok / max(n_words,1)) / (n_tok2 / max(n_words,1))
        print(f"\n=== vs dedicated Korean byte-level BPE ({args.compare_bpe}) ===")
        print(f"korean-bpe tokens/word: {n_tok2 / max(n_words,1):.3f}")
        print(f"base/korean fertility : {ratio:.2f}x")
        print(">> EXTEND the tokenizer (merge vocab+merges)." if ratio > 1.4 else
              ">> Base tokenizer is fine — SKIP extension (Stages 1,2,4a).")


if __name__ == "__main__":
    main()
