#!/usr/bin/env python3
"""
Stage 2 (REPLACEMENT variant) — initialize embeddings after tokenizer replacement.

Companion to `01_tokenizer/replace_korean_in_tokenizer.py`. After that script
removes Korean tokens from Qwen's tokenizer and inserts the user's new Korean
tokens, this script rebuilds the embedding matrix (and lm_head) for the new
token IDs:

  * Tokens present in BOTH old and new vocab (Qwen non-Korean + special tokens):
    COPY the existing embedding to its NEW ID position. Critical for retaining
    Qwen's English/Chinese/general-language knowledge.

  * Tokens present ONLY in new vocab (user's Korean tokens):
    RANDOM init using the same std as Qwen's input embedding distribution.
    These embeddings have no prior; they'll be learned from scratch during
    Stage 4 continual pretraining.

The lookup is by token STRING (not ID), so reordering is handled automatically
— a token that was at ID 50000 in old Qwen and is now at ID 47000 in the new
tokenizer gets its old embedding copied to row 47000. Same for lm_head.

WARNING: random-init Korean embeddings need substantially more continual-
pretraining tokens to converge than averaging-init would. Budget accordingly:
  * Stage 4a (embedding-only warmup): 5-10B tokens (vs 2B for averaging-init)
  * Stage 4b (full continual): 80-120B tokens (vs 60B for averaging-init)
"""
import argparse, json, os, sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
def plan_embedding_init(old_vocab: dict, new_vocab: dict):
    """Decide for each new-vocab token: copy-from-old or random-init.

    Returns:
        copy_pairs   : list of (new_id, old_id) — embedding rows to copy
        random_ids   : list of new_id — embedding rows to random-init
        stats        : dict with counts and a few useful diagnostics
    """
    copy_pairs, random_ids = [], []
    for tok, new_id in new_vocab.items():
        if tok in old_vocab:
            copy_pairs.append((new_id, old_vocab[tok]))
        else:
            random_ids.append(new_id)
    stats = {
        "new_vocab_size": len(new_vocab),
        "n_copied": len(copy_pairs),
        "n_random": len(random_ids),
        "copy_fraction": len(copy_pairs) / max(len(new_vocab), 1),
    }
    return copy_pairs, random_ids, stats


def fill_new_matrix(old_matrix: torch.Tensor,
                    copy_pairs, random_ids,
                    new_vocab_size: int,
                    embed_dim: int,
                    init_std: float = 0.02,
                    seed: int = 42):
    """Build a new embedding matrix; copy rows for `copy_pairs`, random-init
    rows in `random_ids`. Pure tensor function — no model side effects."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Allocate as zeros, then write copies, then random-fill the rest.
    new_matrix = torch.zeros(new_vocab_size, embed_dim, dtype=old_matrix.dtype)

    # 1. Copy preserved tokens
    for new_id, old_id in copy_pairs:
        new_matrix[new_id] = old_matrix[old_id]

    # 2. Random-init the new (Korean) rows.
    if random_ids:
        rand_block = torch.empty(len(random_ids), embed_dim, dtype=torch.float32)
        torch.nn.init.normal_(rand_block, mean=0.0, std=init_std, generator=g)
        rand_block = rand_block.to(old_matrix.dtype)
        idx = torch.tensor(random_ids, dtype=torch.long)
        new_matrix[idx] = rand_block
    return new_matrix


# ---------------------------------------------------------------------------
# I/O wrappers that actually touch the model on disk
# ---------------------------------------------------------------------------
def _load_vocab_from_tokenizer_json(path: Path) -> dict:
    with open(path / "tokenizer.json", encoding="utf-8") as f:
        t = json.load(f)
    return t["model"]["vocab"]


def reinitialize_model(qwen_path, new_tokenizer_path, output_path,
                       init_std: float = 0.02, seed: int = 42):
    """Load Qwen, build new embedding + lm_head matrices, save the new model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    qwen_dir = Path(qwen_path); new_tok_dir = Path(new_tokenizer_path)
    out_dir = Path(output_path); out_dir.mkdir(parents=True, exist_ok=True)

    old_vocab = _load_vocab_from_tokenizer_json(qwen_dir)
    new_vocab = _load_vocab_from_tokenizer_json(new_tok_dir)

    copy_pairs, random_ids, stats = plan_embedding_init(old_vocab, new_vocab)
    print(f"Plan: {stats['n_copied']:,} copied | {stats['n_random']:,} random-init "
          f"(={stats['copy_fraction']:.1%} carry-over)")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(qwen_dir, torch_dtype=torch.bfloat16)
    old_emb = model.get_input_embeddings().weight.data.clone()
    # Qwen3 has untied embeddings; lm_head is separate.
    has_separate_lm_head = (hasattr(model, "lm_head") and
                            id(model.lm_head.weight) != id(model.get_input_embeddings().weight))
    old_lm = model.lm_head.weight.data.clone() if has_separate_lm_head else None

    new_size = len(new_vocab)
    embed_dim = old_emb.shape[1]
    print(f"Old vocab: {old_emb.shape[0]:,} | New vocab: {new_size:,} | dim={embed_dim}")

    new_emb = fill_new_matrix(old_emb, copy_pairs, random_ids, new_size, embed_dim,
                              init_std=init_std, seed=seed)
    new_lm = (fill_new_matrix(old_lm, copy_pairs, random_ids, new_size, embed_dim,
                              init_std=init_std, seed=seed + 1)
              if has_separate_lm_head else None)

    print("Resizing model embeddings...")
    model.resize_token_embeddings(new_size)
    model.get_input_embeddings().weight.data.copy_(new_emb)
    if new_lm is not None:
        model.lm_head.weight.data.copy_(new_lm)

    print(f"Saving model to {out_dir}...")
    model.save_pretrained(out_dir)
    # Copy the new tokenizer files alongside the model
    import shutil
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = new_tok_dir / f
        if src.exists():
            shutil.copy(src, out_dir / f)
    print(f"Done.\n  Copied embeddings: {stats['n_copied']:,}\n  "
          f"Random Korean rows: {stats['n_random']:,}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", required=True, help="path to Qwen3-8B HF model")
    ap.add_argument("--new-tokenizer", required=True,
                    help="path to new tokenizer produced by replace_korean_in_tokenizer.py")
    ap.add_argument("--out", required=True, help="output dir for model with new embeddings")
    ap.add_argument("--init-std", type=float, default=0.02,
                    help="std for random init (Qwen uses ~0.02 for embeddings)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    reinitialize_model(args.qwen, args.new_tokenizer, args.out, args.init_std, args.seed)


# ---------------------------------------------------------------------------
def _selftest():
    # 1. plan_embedding_init: kept tokens → copy; Korean tokens → random
    old_vocab = {"a": 0, "b": 1, "c": 2, "<|sp|>": 3}                  # 4 tokens
    new_vocab = {"a": 0, "b": 1, "조": 2, "선": 3, "<|sp|>": 4}        # 'c' removed, 2 new Korean
    copy_pairs, random_ids, stats = plan_embedding_init(old_vocab, new_vocab)
    # 'a','b','<|sp|>' present in both → copy with potentially different IDs
    assert sorted(copy_pairs) == [(0, 0), (1, 1), (4, 3)]
    assert sorted(random_ids) == [2, 3]
    assert stats["n_copied"] == 3 and stats["n_random"] == 2

    # 2. fill_new_matrix: copied rows match old exactly; random rows are non-zero
    embed_dim = 4
    old = torch.tensor([[1., 2., 3., 4.],     # 'a' at ID 0
                        [5., 6., 7., 8.],     # 'b' at ID 1
                        [9., 0., 0., 0.],     # 'c' at ID 2 (will be DROPPED)
                        [-1., -2., -3., -4.]], # '<|sp|>' at ID 3
                       dtype=torch.float32)
    new_matrix = fill_new_matrix(old, copy_pairs, random_ids, new_vocab_size=5,
                                 embed_dim=embed_dim, init_std=0.02, seed=123)
    # Copied rows exact
    assert torch.equal(new_matrix[0], old[0])              # 'a'
    assert torch.equal(new_matrix[1], old[1])              # 'b'
    assert torch.equal(new_matrix[4], old[3])              # '<|sp|>' moved 3 -> 4
    # Random rows are not exactly zero (with overwhelming probability) and
    # are NOT equal to any old row
    for rid in random_ids:
        row = new_matrix[rid]
        assert row.abs().max() > 0
        for oid in range(old.shape[0]):
            assert not torch.equal(row, old[oid]), \
                f"random row {rid} accidentally equals old row {oid}"
    # Random init std should be close to requested (loose check; small sample)
    rand_rows = new_matrix[torch.tensor(random_ids)]
    assert rand_rows.std().item() < 0.1                    # Conservatively below 5×requested std
    # The "dropped" old row (ID 2 = 'c') should NOT appear anywhere in new_matrix
    for nid in range(new_matrix.shape[0]):
        assert not torch.equal(new_matrix[nid], old[2]), \
            f"dropped token 'c' leaked into new ID {nid}"

    # 3. reordering correctness: a token whose ID shifts gets its OLD embedding
    #    (not whatever happens to live at the new ID in the old matrix)
    #    '<|sp|>' moved 3 → 4. new_matrix[4] must equal old[3], not old[4] (doesn't exist).
    assert torch.equal(new_matrix[4], old[3])

    # 4. determinism: same seed → same random init
    new2 = fill_new_matrix(old, copy_pairs, random_ids, 5, embed_dim, init_std=0.02, seed=123)
    assert torch.equal(new_matrix, new2)
    new3 = fill_new_matrix(old, copy_pairs, random_ids, 5, embed_dim, init_std=0.02, seed=999)
    # Different seed → different random rows (copied rows still identical)
    for rid in random_ids:
        assert not torch.equal(new_matrix[rid], new3[rid]), \
            f"seed changed but random row {rid} identical"
    for cid, _ in copy_pairs:
        assert torch.equal(new_matrix[cid], new3[cid])

    # 5. dtype preservation
    old_bf16 = old.to(torch.bfloat16)
    new_bf16 = fill_new_matrix(old_bf16, copy_pairs, random_ids, 5, embed_dim, init_std=0.02, seed=0)
    assert new_bf16.dtype == torch.bfloat16

    print("PASS all replacement-init tests (plan + fill + reorder + determinism + dtype)")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
