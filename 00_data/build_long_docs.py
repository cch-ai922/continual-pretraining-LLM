#!/usr/bin/env python3
"""
Stage 0 (long-context input) — build the LONG-DOCUMENT corpus for Stage 4c.

Stage 4c (04_pretrain/stage2c_longctx.py) trains at seq_length=32768 to keep
Qwen3-8B's native 32K context alive — but it needs GENUINELY long documents
(blend_long.yaml → ko_long_text_document / en_long_text_document). Most raw Korean
web text is short, so this tool assembles long documents two ways:

  1. PASS-THROUGH : keep any source doc already >= --min-tokens as-is.
  2. PACK         : concatenate shorter docs up to ~--target-tokens each.

⚠️  COHERENCE MATTERS. Concatenating UNRELATED docs creates artificial long
context and teaches the model little. Pass --group-key (e.g. "topic", "source",
"url_domain") so only thematically related docs are joined — the goal is real
long-range dependencies, not 32K of noise. Books / long articles / transcripts /
multi-section reports are ideal raw material.

Output: {"text": "<long document>"} JSONL, ready for 00_data/preprocess_pretrain.sh
(prefix ko_long → ko_long_text_document). Remember Stage 4c runs with
position/attention resets OFF so each long doc spans the whole 32K window.

  python build_long_docs.py --in ko_articles.jsonl --out ko_long.jsonl \
      --group-key topic --target-tokens 28000 --min-tokens 8000 --tokenizer ./qwen3-ko-base-hf
  python build_long_docs.py --selftest
"""
import argparse, json, sys
from collections import OrderedDict


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
def pack_documents(docs, target_tokens, count_tokens, separator="\n\n", min_tokens=None):
    """Greedily pack a list of doc strings into long docs of ~target_tokens.

    A doc already >= min_tokens (if given) is emitted on its own (pass-through).
    Pure function over an injected token counter.
    """
    out, cur, cur_tok = [], [], 0
    sep_tok = count_tokens(separator)
    for d in docs:
        t = count_tokens(d)
        if min_tokens is not None and t >= min_tokens:
            if cur:                                   # flush whatever is buffered
                out.append(separator.join(cur)); cur, cur_tok = [], 0
            out.append(d)                             # long enough alone
            continue
        add = t + (sep_tok if cur else 0)
        if cur and cur_tok + add > target_tokens:     # would overflow -> flush first
            out.append(separator.join(cur)); cur, cur_tok = [], 0
            add = t
        cur.append(d); cur_tok += add
        if cur_tok >= target_tokens:                  # reached target -> flush
            out.append(separator.join(cur)); cur, cur_tok = [], 0
    if cur:
        out.append(separator.join(cur))
    return out


def group_by_key(rows, key):
    """Group row dicts by row[key] (preserving first-seen order). key=None -> one group."""
    groups = OrderedDict()
    for r in rows:
        g = r.get(key, "_all") if key else "_all"
        groups.setdefault(g, []).append(r.get("text", ""))
    return groups


def build_long_corpus(rows, target_tokens, count_tokens, group_key=None,
                      min_tokens=None, separator="\n\n"):
    """Group (for coherence) then pack each group. Returns list[str] of long docs."""
    out = []
    for _, docs in group_by_key(rows, group_key).items():
        out.extend(pack_documents([d for d in docs if d.strip()],
                                  target_tokens, count_tokens, separator, min_tokens))
    return out


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
def _make_counter(tokenizer_path):
    if tokenizer_path:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        return lambda s: len(tok.encode(s, add_special_tokens=False))
    print("[warn] no --tokenizer: counting tokens by whitespace words "
          "(lengths approximate; pass your tokenizer for accurate packing).",
          file=sys.stderr)
    return lambda s: len(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="JSONL: {text, [group-key]}")
    ap.add_argument("--out", required=True, help="JSONL: {text} long documents")
    ap.add_argument("--target-tokens", type=int, default=28000,
                    help="pack up to this many tokens/doc (keep headroom under 32768)")
    ap.add_argument("--min-tokens", type=int, default=None,
                    help="docs already this long are kept as-is (pass-through)")
    ap.add_argument("--group-key", default=None,
                    help="join only docs sharing this field (e.g. topic/source) for coherence")
    ap.add_argument("--separator", default="\n\n")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer for accurate token counts")
    args = ap.parse_args()

    count_tokens = _make_counter(args.tokenizer)
    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    longs = build_long_corpus(rows, args.target_tokens, count_tokens,
                              group_key=args.group_key, min_tokens=args.min_tokens,
                              separator=args.separator)
    toks = [count_tokens(d) for d in longs]
    with open(args.out, "w", encoding="utf-8") as f:
        for d in longs:
            f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
    mean_t = sum(toks) / len(toks) if toks else 0
    print(f"{len(rows):,} source docs -> {len(longs):,} long docs "
          f"(mean {mean_t:,.0f} tokens, max {max(toks) if toks else 0:,})  -> {args.out}")
    print("Next: preprocess to ko_long_text_document, set --long-budget-b in build_data_mix.py, "
          "run 04_pretrain/stage2c_longctx.py.")


# ---------------------------------------------------------------------------
def _selftest():
    wc = lambda s: len(s.split())            # 1 token per word

    # pack to ~target; 'aa bb' = 2 words each, sep '\n\n' = 1 word
    docs = ["aa bb", "cc dd", "ee ff", "gg hh"]
    packed = pack_documents(docs, target_tokens=6, count_tokens=wc, separator="\n\n")
    assert all(wc(p) <= 8 for p in packed), [wc(p) for p in packed]   # ~target + slack
    assert sum("aa" in p or "cc" in p or "ee" in p or "gg" in p for p in packed) >= 1
    # all content preserved
    joined = " ".join(packed)
    for w in ["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh"]:
        assert w in joined

    # pass-through: a doc already >= min_tokens is emitted alone
    big = " ".join(["w"] * 50)
    packed2 = pack_documents(["aa bb", big, "cc dd"], target_tokens=20,
                             count_tokens=wc, min_tokens=40)
    assert big in packed2 and any(p == big for p in packed2), "long doc kept as-is"

    # grouping keeps topics separate (no cross-topic concatenation)
    rows = [{"text": "korea one two", "topic": "geo"},
            {"text": "seoul three four", "topic": "geo"},
            {"text": "atom five six", "topic": "sci"}]
    longs = build_long_corpus(rows, target_tokens=100, count_tokens=wc, group_key="topic")
    # the 'sci' doc must never be glued to 'geo' docs
    assert not any(("korea" in d and "atom" in d) for d in longs), longs
    assert any("korea" in d and "seoul" in d for d in longs), "same-topic docs join"

    # empty/whitespace docs are skipped
    longs2 = build_long_corpus([{"text": ""}, {"text": "  "}, {"text": "real doc here"}],
                               target_tokens=100, count_tokens=wc)
    assert longs2 == ["real doc here"], longs2

    print("PASS all long-doc builder tests (pack + pass-through + grouping + skip-empty)")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
