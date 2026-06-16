#!/usr/bin/env python3
"""
Stage 1 — PROPERLY extend Qwen3's byte-level BPE with new Korean tokens.

The right way (vs the broken `add_tokens` side-channel): splice the new pieces into
the BPE MODEL itself — union the vocab AND append the new merge rules — so the new
tokens are produced NATIVELY by the BPE merge process, not by a literal added-token
matcher running around it.

Steps:
  1. read Qwen's tokenizer JSON  -> base vocab + base merges
  2. read the Korean byte-level BPE JSON (from train_korean_bpe.py)
  3. add Korean tokens missing from base (assign fresh ids)
  4. append Korean merge rules missing from base (preserve their relative order)
  5. rebuild a Tokenizer with the merged BPE model, keeping Qwen's
     normalizer / pre-tokenizer / decoder / post-processor untouched

Merge-priority caveat: appended Korean merges have LOWER priority than Qwen's
existing merges. Since Qwen has few Hangul merges this is fine in practice; if
you see odd Korean segmentation, interleave by frequency instead of appending.
"""
import argparse, json


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE  (operates on tokenizer-JSON dicts)
# ---------------------------------------------------------------------------
def _merge_to_pair(m):
    """tokenizers stores merges as either 'a b' (str) or ['a','b'] (list)."""
    return tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1))


def merge_bpe(base_model: dict, ko_model: dict, max_add=None):
    """Mutate & return base_model['vocab'] / ['merges'] with Korean pieces spliced in.
    Returns (n_tokens_added, n_merges_added)."""
    base_vocab = base_model["vocab"]
    base_merges = base_model["merges"]
    ko_vocab = ko_model["vocab"]
    ko_merges = ko_model["merges"]

    # 1. add missing tokens, ordered by Korean id so dependencies precede composites
    next_id = max(base_vocab.values()) + 1
    added = 0
    for tok in sorted(ko_vocab, key=lambda t: ko_vocab[t]):
        if tok not in base_vocab:
            base_vocab[tok] = next_id
            next_id += 1
            added += 1
            if max_add and added >= max_add:
                break

    # 2. append missing merges (only those whose pieces + result now exist in vocab)
    have = {_merge_to_pair(m) for m in base_merges}
    list_format = bool(base_merges) and isinstance(base_merges[0], list)
    merges_added = 0
    for m in ko_merges:
        pair = _merge_to_pair(m)
        if pair in have:
            continue
        if pair[0] in base_vocab and pair[1] in base_vocab and \
           (pair[0] + pair[1]) in base_vocab:
            base_merges.append(list(pair) if list_format else f"{pair[0]} {pair[1]}")
            have.add(pair)
            merges_added += 1

    return added, merges_added


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B", help="HF id/path of Qwen3 tokenizer")
    ap.add_argument("--korean-bpe", required=True, help="ko_bpe.json from train_korean_bpe.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-add", type=int, default=16000)
    args = ap.parse_args()

    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from tokenizers import Tokenizer

    base = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    assert base.is_fast, "need a fast (tokenizers-backed) tokenizer"
    base_json = json.loads(base.backend_tokenizer.to_str())
    ko_json = json.loads(Tokenizer.from_file(args.korean_bpe).to_str())
    assert base_json["model"]["type"] == "BPE", "base model is not BPE?"

    before_v = len(base_json["model"]["vocab"])
    before_m = len(base_json["model"]["merges"])
    n_tok, n_mrg = merge_bpe(base_json["model"], ko_json["model"], args.max_add)

    new_backend = Tokenizer.from_str(json.dumps(base_json))
    fast = PreTrainedTokenizerFast(tokenizer_object=new_backend)
    fast.add_special_tokens({k: v for k, v in base.special_tokens_map.items()
                             if isinstance(v, str)})
    if getattr(base, "chat_template", None):
        fast.chat_template = base.chat_template
    fast.save_pretrained(args.out)

    print(f"vocab : {before_v:,} -> {before_v + n_tok:,}  (+{n_tok:,})")
    print(f"merges: {before_m:,} -> {before_m + n_mrg:,}  (+{n_mrg:,})")
    print(f"saved properly-extended tokenizer -> {args.out}")


if __name__ == "__main__":
    main()
