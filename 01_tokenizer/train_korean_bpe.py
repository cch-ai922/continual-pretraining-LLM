#!/usr/bin/env python3
"""
Stage 1 — train a BYTE-LEVEL BPE on Korean that is COMPATIBLE with Qwen's tokenizer.

Why not SentencePiece: Qwen's tokenizer is byte-level BPE (GPT-2/tiktoken style).
SentencePiece pieces live in a different alphabet (the U+2581 space marker etc.),
so they cannot be spliced into Qwen's merge table. We train in the SAME byte-level
space — and, crucially, reuse QWEN'S OWN PRE-TOKENIZER so the merges align exactly
on the same pre-token boundaries and byte alphabet.

Output: ko_bpe.json (a tokenizers.Tokenizer with vocab + merges) for extend_tokenizer.py.
"""
import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="UTF-8 Korean text file(s)", nargs="+")
    ap.add_argument("--base", default="Qwen/Qwen3-8B",
                    help="reuse this tokenizer's pre-tokenizer/decoder for compatibility")
    ap.add_argument("--vocab-size", type=int, default=16000,
                    help="size of the Korean BPE; only the NEW pieces get merged into Qwen")
    ap.add_argument("--out", default="ko_bpe.json")
    args = ap.parse_args()

    tok = Tokenizer(models.BPE())

    # Mirror Qwen's byte-level pipeline so the merges are splice-compatible.
    try:
        from transformers import AutoTokenizer
        base = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
        tok.pre_tokenizer = base.backend_tokenizer.pre_tokenizer
        tok.decoder = base.backend_tokenizer.decoder
        print(f"reusing pre-tokenizer/decoder from {args.base}")
    except Exception as e:
        print(f"[warn] couldn't load base ({e}); falling back to plain ByteLevel")
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 byte chars
        special_tokens=[],
        show_progress=True,
    )
    tok.train(args.corpus, trainer)
    tok.save(args.out)
    print(f"trained Korean byte-level BPE ({tok.get_vocab_size()} pieces) -> {args.out}")


if __name__ == "__main__":
    main()
