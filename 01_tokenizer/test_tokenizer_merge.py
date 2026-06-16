#!/usr/bin/env python3
"""
Verifies extend_tokenizer.merge_bpe: that splicing vocab+merges makes new Korean
tokens get produced by the BPE ALGORITHM (compositional merges), not by an
added-token side channel. No Qwen download needed — we build two tiny byte-level
BPEs that share Qwen's byte alphabet, then merge them.

Run: python test_tokenizer_merge.py
"""
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from extend_tokenizer import merge_bpe

EN = ["the quick brown fox jumps over the lazy dog. " * 50,
      "language models learn from text data and tokens. " * 50]
KO = ["안녕하십니까 세계 이것은 시험입니다 조선어는 매우 훌륭합니다. " * 80,
      "기계학습과 언어모형은 조선어본문으로부터 배운다. " * 80]
PROBE = "조선어"   # a Korean phrase we expect to tokenize more efficiently


def train_bpe(corpus, vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab_size,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             special_tokens=[], show_progress=False)
    tok.train_from_iterator(corpus, tr)
    return tok


def main():
    base = train_bpe(EN, 400)     # an "English-only" base (poor at Korean, like Qwen)
    ko = train_bpe(KO, 600)       # the Korean byte-level BPE

    base_json = json.loads(base.to_str())
    ko_json = json.loads(ko.to_str())

    n_base = len(base.encode(PROBE).ids)
    print(f"base tokens for {PROBE!r:>16}: {n_base}")

    # --- the operation under test: merge vocab + merges ---
    n_tok, n_mrg = merge_bpe(base_json["model"], ko_json["model"])
    print(f"merged in: +{n_tok} tokens, +{n_mrg} merges")

    merged = Tokenizer.from_str(json.dumps(base_json))
    enc = merged.encode(PROBE)
    n_merged = len(enc.ids)
    print(f"merged tokens for {PROBE!r:>14}: {n_merged}  -> {enc.tokens}")

    # 1. fertility must improve (fewer tokens for the Korean probe)
    assert n_merged < n_base, "merged tokenizer did not reduce Korean token count!"

    # 2. at least one produced token must be a MULTI-codepoint Korean piece, i.e.
    #    something BPE composed from bytes (not a lone byte, not an added token).
    #    Decode each token's byte-level chars back to real text and check length.
    def _gpt2_byte_decoder():
        bs = (list(range(ord("!"), ord("~") + 1)) +
              list(range(ord("¡"), ord("¬") + 1)) +
              list(range(ord("®"), ord("ÿ") + 1)))
        cs = bs[:]; n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b); cs.append(256 + n); n += 1
        return {chr(c): b for b, c in zip(bs, cs)}

    _DEC = _gpt2_byte_decoder()

    def token_text(t):
        try:
            return bytes(_DEC[c] for c in t).decode("utf-8")
        except Exception:
            return ""  # incomplete UTF-8 (a partial multi-byte char) -> skip
    def is_hangul(s):
        return any("\uac00" <= c <= "\ud7a3" for c in s)
    # A Hangul syllable is a single precomposed codepoint built from 3 UTF-8 bytes,
    # so the meaningful signal is "byte-level BPE composed a whole Korean syllable"
    # (unlike Devanagari, where one akshara already spans 2+ codepoints).
    multis = [t for t in enc.tokens if is_hangul(token_text(t))]
    print(f"composed Hangul pieces: {[token_text(t) for t in multis]}")
    assert multis, "no composed Hangul token was produced by BPE merges!"

    # 3. it must be lossless (byte-level roundtrip)
    assert merged.decode(enc.ids) == PROBE, "decode did not roundtrip!"

    # 4. we never called add_tokens anywhere (added_tokens stays empty)
    assert json.loads(merged.to_str()).get("added_tokens", []) == [], \
        "tokens leaked into the added-token side channel!"

    print("\nAll tokenizer-merge tests passed: new Korean tokens are produced "
          "COMPOSITIONALLY by BPE, fertility improved, decode is lossless.")


if __name__ == "__main__":
    main()
