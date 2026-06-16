#!/usr/bin/env python3
"""
Stage 1 (REPLACEMENT variant) — remove Korean from Qwen's tokenizer and replace
with the user's Korean tokens.

Unlike `extend_tokenizer.py` which UNIONS Qwen's vocab + the user's Korean
tokenizer (preserving Qwen's Korean tokens and merges), this script:

  1. Identifies Korean-bearing tokens in Qwen's BBPE vocabulary
  2. Removes them AND all merge rules that produce them
  3. Inserts the user's Korean tokens + merges
  4. Re-numbers the remaining Qwen tokens to fill the gap
  5. Preserves special tokens at the highest IDs (Qwen tradition)

The new tokenizer has the same byte-level base alphabet (the 256 base bytes are
shared across ALL languages and never removed), but a completely fresh Korean
sub-vocabulary owned by the user's training corpus.

USE THIS WHEN:
  * You want maximum control over Korean token distribution
  * You believe Qwen's preexisting Korean training was suboptimal
  * You can afford ~2× the continual-pretraining budget (random-init Korean
    embeddings need substantial training to converge)

DON'T USE WHEN:
  * Your goal is just better Korean fertility — use extend_tokenizer.py instead
  * Your continual-pretraining budget is < 50B Korean tokens
  * You're new to LLM adaptation — extension is more forgiving

See TOKENIZER_REPLACEMENT_GUIDE.md at repo root for the tradeoff details.
"""
import argparse, json, os, shutil, sys
from pathlib import Path


# ---------------------------------------------------------------------------
# GPT-2 / BBPE byte ↔ unicode mapping — the standard one transformers uses.
# Critical to get right; off-by-one in the alphabet here corrupts the tokenizer.
# ---------------------------------------------------------------------------
def _gpt2_bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_ENCODER = _gpt2_bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
def bbpe_token_to_bytes(token_str):
    """Convert a BBPE-encoded token string back to its raw bytes.
    Returns None if the string contains characters not in the BBPE alphabet."""
    try:
        return bytes(BYTE_DECODER[c] for c in token_str)
    except KeyError:
        return None


def has_korean(text):
    """Hangul syllables block U+AC00 .. U+D7A3 (the script used for Korean)."""
    return any('\uac00' <= c <= '\ud7a3' for c in text)


def classify_token(token_str):
    """Return 'korean' / 'other' / 'invalid' based on the token's decoded UTF-8."""
    bts = bbpe_token_to_bytes(token_str)
    if bts is None:
        return 'invalid'
    try:
        decoded = bts.decode('utf-8')
    except UnicodeDecodeError:
        # partial UTF-8 sequences — treat as 'other' (shared base bytes for many scripts)
        return 'other'
    return 'korean' if has_korean(decoded) else 'other'


def _normalize_merge(m):
    """Merges in tokenizer.json may be 'a b' strings or [a, b] arrays."""
    if isinstance(m, str):
        parts = m.split(" ", 1)
        return tuple(parts) if len(parts) == 2 else None
    if isinstance(m, list) and len(m) == 2:
        return tuple(m)
    return None


def filter_merges(merges, removed_tokens):
    """Drop merges whose result (left + right) is a removed token."""
    kept = []
    for m in merges:
        pair = _normalize_merge(m)
        if pair is None:
            continue
        left, right = pair
        if (left + right) in removed_tokens:
            continue
        kept.append(pair)
    return kept


def build_new_vocab(kept_tokens, new_korean_tokens, special_tokens):
    """Renumber tokens so kept come first, new Korean next, specials last."""
    new_vocab = {}
    next_id = 0
    for tok in kept_tokens:
        new_vocab[tok] = next_id; next_id += 1
    boundary_after_kept = next_id
    for tok in new_korean_tokens:
        new_vocab[tok] = next_id; next_id += 1
    boundary_after_korean = next_id
    for tok in special_tokens:
        new_vocab[tok] = next_id; next_id += 1
    return new_vocab, boundary_after_kept, boundary_after_korean


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def replace_korean_tokens(qwen_path, korean_path, output_path):
    """Full replacement pipeline. Returns a dict of stats."""
    qwen_dir, korean_dir, out_dir = Path(qwen_path), Path(korean_path), Path(output_path)

    with open(qwen_dir / "tokenizer.json", encoding="utf-8") as f:
        qwen_json = json.load(f)

    qwen_vocab = qwen_json["model"]["vocab"]            # {token_str: id}
    qwen_merges_raw = qwen_json["model"]["merges"]
    qwen_added = qwen_json.get("added_tokens", [])

    # Identify Korean tokens in Qwen
    korean_in_qwen = {tok for tok in qwen_vocab if classify_token(tok) == 'korean'}

    # Identify special tokens (preserve at high IDs)
    special_token_strs = []
    for t in sorted(qwen_added, key=lambda x: x.get("id", 0)):
        if t.get("special", False):
            special_token_strs.append(t["content"])
    special_set = set(special_token_strs)

    # Sort kept tokens by their ORIGINAL Qwen ID so we preserve relative ordering
    kept_tokens = []
    for tok, tok_id in sorted(qwen_vocab.items(), key=lambda x: x[1]):
        if tok in korean_in_qwen or tok in special_set:
            continue
        kept_tokens.append(tok)

    # Filter Qwen merges: drop any that produce a Korean token
    filtered_qwen_merges = filter_merges(qwen_merges_raw, korean_in_qwen)

    # Load user's Korean tokenizer
    with open(korean_dir / "tokenizer.json", encoding="utf-8") as f:
        korean_json = json.load(f)
    korean_vocab = korean_json["model"]["vocab"]
    korean_merges_raw = korean_json["model"]["merges"]

    # Add user's Korean tokens, excluding any that overlap with kept Qwen tokens
    # (e.g., base bytes 0x00..0xFF appear in both tokenizers — skip duplicates)
    kept_set = set(kept_tokens)
    new_korean_tokens = []
    for tok, _id in sorted(korean_vocab.items(), key=lambda x: x[1]):
        if tok in kept_set or tok in special_set:
            continue
        new_korean_tokens.append(tok)

    # Combine merges: filtered-Qwen-merges + user's-Korean-merges
    new_merges_pairs = list(filtered_qwen_merges)
    seen_merges = {(l, r) for l, r in filtered_qwen_merges}
    for m in korean_merges_raw:
        pair = _normalize_merge(m)
        if pair is None or pair in seen_merges:
            continue
        new_merges_pairs.append(pair); seen_merges.add(pair)

    # Renumber
    new_vocab, n_kept, n_after_korean = build_new_vocab(
        kept_tokens, new_korean_tokens, special_token_strs)
    final_vocab_size = n_after_korean + len(special_token_strs)

    # Rebuild added_tokens with new IDs (special tokens get the highest IDs)
    new_added_tokens = []
    id_by_tok = {t["content"]: t for t in qwen_added}
    for tok in special_token_strs:
        spec = dict(id_by_tok.get(tok, {"content": tok, "special": True}))
        spec["id"] = new_vocab[tok]
        new_added_tokens.append(spec)

    # Assemble new tokenizer.json (preserve pre_tokenizer, post_processor, normalizer, etc.)
    new_json = dict(qwen_json)
    new_json["model"] = dict(qwen_json["model"])
    new_json["model"]["vocab"] = new_vocab
    new_json["model"]["merges"] = [f"{l} {r}" for l, r in new_merges_pairs]
    new_json["added_tokens"] = new_added_tokens

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(new_json, f, ensure_ascii=False, indent=2)
    # Copy tokenizer_config.json (chat template etc. — same as Qwen's)
    cfg_src = qwen_dir / "tokenizer_config.json"
    if cfg_src.exists():
        shutil.copy(cfg_src, out_dir / "tokenizer_config.json")
    # Also copy special_tokens_map.json if present
    stm_src = qwen_dir / "special_tokens_map.json"
    if stm_src.exists():
        shutil.copy(stm_src, out_dir / "special_tokens_map.json")

    stats = {
        "qwen_vocab_size": len(qwen_vocab),
        "korean_removed_from_qwen": len(korean_in_qwen),
        "qwen_merges_dropped": len(qwen_merges_raw) - len(filtered_qwen_merges),
        "user_korean_added": len(new_korean_tokens),
        "user_korean_merges_added": len(new_merges_pairs) - len(filtered_qwen_merges),
        "kept_qwen_tokens": len(kept_tokens),
        "special_tokens": len(special_token_strs),
        "final_vocab_size": final_vocab_size,
        "boundary_kept_korean": n_kept,
        "boundary_korean_special": n_after_korean,
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", required=True, help="path to Qwen3-8B (containing tokenizer.json)")
    ap.add_argument("--korean", required=True, help="path to user's Korean BPE tokenizer dir")
    ap.add_argument("--out", required=True, help="output directory for new tokenizer")
    args = ap.parse_args()
    stats = replace_korean_tokens(args.qwen, args.korean, args.out)
    print("\n=== Replacement summary ===")
    for k, v in stats.items():
        print(f"  {k:35s} {v:>10,}")
    print(f"\nID layout: [0, {stats['boundary_kept_korean']:,}) Qwen non-Korean | "
          f"[{stats['boundary_kept_korean']:,}, {stats['boundary_korean_special']:,}) user Korean | "
          f"[{stats['boundary_korean_special']:,}, {stats['final_vocab_size']:,}) specials")
    print(f"Saved to: {args.out}")


# ---------------------------------------------------------------------------
def _selftest():
    # 1. Byte mapping roundtrip — fundamental correctness check
    assert len(BYTE_ENCODER) == 256, len(BYTE_ENCODER)
    for b in range(256):
        c = BYTE_ENCODER[b]
        assert BYTE_DECODER[c] == b, f"byte {b} roundtrip failed"

    # 2. Korean character detection on raw strings
    assert has_korean("안녕")              # "hello" in Korean
    assert has_korean("hello 안녕")        # mixed → still Korean
    assert not has_korean("hello")
    assert not has_korean("你好")          # Chinese, not Hangul

    # 3. classify_token on real BBPE-encoded token strings
    #    Encode '안녕' (UTF-8 bytes) into BBPE form, classify as 'korean'
    raw = "안녕".encode("utf-8")
    bbpe_korean = "".join(BYTE_ENCODER[b] for b in raw)
    assert classify_token(bbpe_korean) == 'korean', \
        f"Korean token mis-classified: got {classify_token(bbpe_korean)}"
    #    Encode plain ASCII 'hello' into BBPE
    bbpe_ascii = "".join(BYTE_ENCODER[b] for b in b"hello")
    assert classify_token(bbpe_ascii) == 'other'
    #    Partial UTF-8: take first 2 of 3 Hangul bytes — should be 'other' (no decode)
    partial = "".join(BYTE_ENCODER[b] for b in raw[:2])
    assert classify_token(partial) == 'other', "partial UTF-8 should be 'other'"
    #    Invalid character (not in BBPE alphabet)
    assert classify_token("\u0001") == 'invalid'

    # 4. Merge filtering — drop merges whose result is in removed_tokens
    merges = [["a", "b"], "c d", ["e", "f"], "g h"]
    removed = {"cd", "ef"}
    kept = filter_merges(merges, removed)
    assert kept == [("a", "b"), ("g", "h")], kept

    # 5. Vocab building: kept come first, then Korean, then specials at high IDs
    kept = ["a", "b", "c"]
    new_h = ["조", "선"]
    spec = ["<|im_start|>", "<|im_end|>"]
    vocab, b1, b2 = build_new_vocab(kept, new_h, spec)
    assert vocab == {"a": 0, "b": 1, "c": 2, "조": 3, "선": 4,
                     "<|im_start|>": 5, "<|im_end|>": 6}
    assert b1 == 3 and b2 == 5     # kept→korean boundary at 3, korean→specials at 5

    # 6. End-to-end on synthetic tokenizers
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        qwen_dir = Path(td) / "qwen"; korean_dir = Path(td) / "korean"; out_dir = Path(td) / "out"
        qwen_dir.mkdir(); korean_dir.mkdir()

        # Synthetic Qwen tokenizer with: 'a', 'b' (other), BBPE-Korean token, special
        bbpe_ko = "".join(BYTE_ENCODER[b] for b in "선".encode("utf-8"))
        bbpe_more = "".join(BYTE_ENCODER[b] for b in "조".encode("utf-8"))
        qwen_t = {
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, bbpe_ko: 2, bbpe_more: 3, "<|sp|>": 4},
                "merges": ["a b", f"{bbpe_ko[:len(bbpe_ko)//2]} {bbpe_ko[len(bbpe_ko)//2:]}"],
            },
            "added_tokens": [{"content": "<|sp|>", "id": 4, "special": True}],
        }
        with open(qwen_dir / "tokenizer.json", "w") as f:
            json.dump(qwen_t, f)

        # Synthetic user's Korean tokenizer: 2 new tokens (one duplicates Qwen's bbpe_ko)
        new_user_ko = "".join(BYTE_ENCODER[b] for b in "어".encode("utf-8"))
        korean_t = {
            "model": {
                "type": "BPE",
                "vocab": {bbpe_ko: 0, new_user_ko: 1},   # bbpe_ko already in Qwen as Korean → both removed and re-added by user
                "merges": [f"{new_user_ko[:len(new_user_ko)//2]} {new_user_ko[len(new_user_ko)//2:]}"],
            },
            "added_tokens": [],
        }
        with open(korean_dir / "tokenizer.json", "w") as f:
            json.dump(korean_t, f)

        stats = replace_korean_tokens(str(qwen_dir), str(korean_dir), str(out_dir))
        assert stats["korean_removed_from_qwen"] == 2, stats         # bbpe_ko + bbpe_more
        assert stats["user_korean_added"] == 2, stats                # both Korean tokens from user added
        assert stats["kept_qwen_tokens"] == 2, stats                # 'a', 'b'
        assert stats["special_tokens"] == 1, stats                  # <|sp|>

        # Verify saved tokenizer has correct ID layout
        with open(out_dir / "tokenizer.json") as f:
            new_t = json.load(f)
        v = new_t["model"]["vocab"]
        assert v["a"] == 0 and v["b"] == 1, "kept tokens should occupy low IDs"
        # User's Korean tokens at IDs 2, 3 (in some order)
        assert {v[bbpe_ko], v[new_user_ko]} == {2, 3}, f"Korean IDs wrong: {v}"
        # Special at highest
        assert v["<|sp|>"] == 4
        # added_tokens reflects new ID
        assert new_t["added_tokens"][0]["id"] == 4

    print("PASS all tokenizer-replacement tests (byte map + classify + filter + e2e)")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
