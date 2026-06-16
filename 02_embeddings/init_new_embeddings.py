#!/usr/bin/env python3
"""
Stage 2 — initialize embeddings for the newly added Korean tokens (Qwen3-8B).

The right way (FOCUS / WECHSEL-lite): for each NEW token, tokenize its surface
string with the ORIGINAL tokenizer into old sub-pieces, and set the new embedding
to the mean of those old pieces' embeddings. Every new token then starts at a
semantically meaningful point, so training is faster and far less disruptive than
random or global-mean init.

Qwen3-8B specifics that make this simple:
  * Standard dense transformer — NO multi-token-prediction head, NO vision tower.
  * Embeddings are UNTIED (tie_word_embeddings=False), so we initialize BOTH the
    input embedding matrix AND the lm_head. (Input-only would let the model READ
    Korean tokens but not PRODUCE them.)

The core math lives in `average_rows()` and is unit-tested in test_init_logic.py
(no model download required).
"""
import argparse
import numpy as np


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
def average_rows(piece_id_lists, matrix, fallback_row):
    """For each new token, average the rows of `matrix` at its old piece ids.

    piece_id_lists : list[list[int]]  old sub-piece ids for each new token
    matrix         : np.ndarray [old_vocab, dim]
    fallback_row   : np.ndarray [dim]  used when a token has no pieces
    returns        : np.ndarray [n_new, dim]
    """
    dim = matrix.shape[1]
    out = np.empty((len(piece_id_lists), dim), dtype=matrix.dtype)
    for i, pieces in enumerate(piece_id_lists):
        out[i] = matrix[pieces].mean(axis=0) if pieces else fallback_row
    return out


# ---------------------------------------------------------------------------
# MODEL-SIDE DRIVER
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="base HF model")
    ap.add_argument("--new-tokenizer", required=True, help="extended tokenizer dir from Stage 1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise-std", type=float, default=1e-4,
                    help="tiny noise to break symmetry between identical inits")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    base_tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    new_tok = AutoTokenizer.from_pretrained(args.new_tokenizer, trust_remote_code=True)
    old_vocab, new_vocab = len(base_tok), len(new_tok)
    n_new = new_vocab - old_vocab
    print(f"old vocab {old_vocab:,} -> new vocab {new_vocab:,}  (+{n_new:,})")
    assert n_new > 0, "new tokenizer is not larger than base; nothing to init"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.resize_token_embeddings(new_vocab)  # HF default-inits new rows; we overwrite below

    new_ids = list(range(old_vocab, new_vocab))
    new_strs = [new_tok.convert_tokens_to_string([t])
                for t in new_tok.convert_ids_to_tokens(new_ids)]
    piece_id_lists = [base_tok.encode(s, add_special_tokens=False) for s in new_strs]
    empties = sum(1 for p in piece_id_lists if not p)
    print(f"new tokens with empty decomposition (use fallback): {empties}")

    def overwrite(weight, name):
        W = weight.detach().to(torch.float32).cpu().numpy()
        fallback = W[:old_vocab].mean(axis=0)
        rows = average_rows(piece_id_lists, W[:old_vocab], fallback)
        if args.noise_std:
            rows = rows + np.random.normal(0, args.noise_std, rows.shape).astype(rows.dtype)
        W[old_vocab:new_vocab] = rows
        weight.data.copy_(torch.from_numpy(W).to(weight.dtype).to(weight.device))
        print(f"  initialized {name}: {rows.shape[0]} x {rows.shape[1]}")

    inp, out = model.get_input_embeddings(), model.get_output_embeddings()
    tied = getattr(model.config, "tie_word_embeddings", False) or out is None

    print("initializing input embeddings ...")
    overwrite(inp.weight, "input_embeddings")
    if not tied:                       # Qwen3 is UNTIED -> must init lm_head too
        print("untied lm_head detected -> initializing lm_head ...")
        overwrite(out.weight, "lm_head")
    else:
        print("embeddings tied; lm_head shares the input matrix (done).")

    model.save_pretrained(args.out, safe_serialization=True)
    new_tok.save_pretrained(args.out)
    print(f"saved initialized model + tokenizer -> {args.out}")


if __name__ == "__main__":
    main()
