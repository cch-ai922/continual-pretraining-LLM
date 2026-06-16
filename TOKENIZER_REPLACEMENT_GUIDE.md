# Guide — Tokenizer EXTENSION vs REPLACEMENT for Korean Adaptation

Two paths exist for getting your Korean tokens into Qwen3-8B's tokenizer. They
have very different cost/benefit profiles. This guide explains when each is
appropriate.

## The two paths in one sentence each

**Extension (`01_tokenizer/extend_tokenizer.py` + `02_embeddings/init_new_embeddings.py`)**:
keep Qwen's existing Korean tokens, ADD your Korean tokens alongside, give new
tokens averaging-init embeddings derived from their constituent subwords. The
model retains its existing Korean capability and is enhanced by your tokens.

**Replacement (`01_tokenizer/replace_korean_in_tokenizer.py` + `02_embeddings/init_replaced_embeddings.py`)**:
REMOVE Qwen's existing Korean tokens and their merge rules, insert your Korean
tokens at the freed-up IDs, give all new Korean tokens random-init embeddings.
The model's Korean capability is forced to be rebuilt from scratch in the
embedding space.

## The cost/benefit comparison

|                                                    | Extension                  | Replacement                                      |
| -------------------------------------------------- | -------------------------- | ------------------------------------------------ |
| Tokenizer surgery complexity                       | Low (union vocab + merges) | High (remove + filter + renumber + add)          |
| Embedding init strategy                            | Averaging from subwords    | Random                                           |
| Initial loss on Korean text                        | Modest jump above baseline | Very high (essentially uniform)                  |
| Tokens needed for Stage 4a (embedding warmup)      | ~2B                        | ~5-10B                                           |
| Tokens needed for Stage 4b (full continual)        | ~60B                       | ~80-120B                                         |
| Total continual-pretraining budget                 | ~60B                       | ~100B+                                           |
| Carry-over of Qwen's Korean knowledge              | Full                       | Embedding layer wiped; transformer body retained |
| Risk of catastrophic forgetting of English/Chinese | Low                        | Low (other tokens preserved)                     |
| Engineering debuggability                          | Higher                     | Lower (more moving parts)                        |

## When EXTENSION is the right choice

- **Default case** for most adaptation projects
- You want better Korean fertility without significant engineering overhead
- Your continual-pretraining budget is limited (< 80B Korean tokens)
- You're early in the adaptation project and want a forgiving recipe
- Qwen's existing Korean capability is "good enough" as a starting point
- You're new to LLM adaptation work

## When REPLACEMENT is the right choice

- **Research / experimental** setting where you want to study how the
  transformer body's multilingual representations support a freshly-attached
  language-specific embedding surface
- You strongly believe Qwen's preexisting Korean training data was low quality
  (lots of machine-translated text, code-switched scraping, etc.) and want to
  avoid inheriting those distributional biases
- You have a high-quality curated Korean corpus and want maximum control over
  the token distribution that emerges
- Your continual-pretraining budget is generous (> 100B Korean tokens)
- You're prepared to debug a more complex pipeline

## Two things to be honest about

**1. Replacement does NOT remove Qwen's Korean knowledge.**

Qwen's Korean capability lives in the transformer body weights — attention
patterns, FFN representations, the LayerNorm/RMSNorm scales — not in the
embedding layer alone. The embedding layer is just a lookup table from token
IDs to vectors. By removing Korean tokens and random-initing their replacements,
you reset the lookup, but the transformer body still encodes everything it
learned from Qwen's Korean pretraining data. You explicitly stated you want to
preserve transformer-body knowledge — good, since you couldn't remove it
without re-pretraining from scratch.

What replacement DOES achieve: complete control over the input/output token
distribution. The model is forced to learn `embedding[your_korean_token]` from
scratch in the context of the existing transformer's representational space.

**2. Random init is genuinely costly.**

Averaging-init gives new tokens a meaningful starting point — the embedding
for "조선어" is initialized as the average of embeddings for its constituent
byte/subword tokens, which are already co-adapted with the transformer body.
Random init throws away this structure. Initial loss on Korean text will be
essentially uniform across the vocabulary (because the random Korean embeddings
produce roughly uniform logits in the LM head). Convergence requires
substantially more training tokens.

Don't choose replacement because it sounds cleaner — choose it because the
research/quality benefit specifically justifies the cost.

## ID layout produced by replacement

After running `replace_korean_in_tokenizer.py`, the new tokenizer has this ID
layout:

```
[ 0, K )            Qwen non-Korean tokens, renumbered preserving relative order
                    (English, Chinese, code, base bytes, etc.)

[ K, K+H )          Your new Korean tokens, in the order they appear in your
                    Korean BPE's vocab

[ K+H, K+H+S )      Qwen special tokens (<|im_start|>, <|im_end|>, <|endoftext|>,
                    etc.) preserving their relative order from Qwen
```

This puts special tokens at the highest IDs — a Qwen tradition that helps with
some downstream tooling (vLLM logit-bias handling, certain chat templates).

## Continual-pretraining budget recommendations

For the replacement path specifically:

**Stage 4a (embedding-only warmup)** — Freeze the transformer body; only train
the embedding matrix and lm_head. The point is to let the random-init Korean
embeddings find a sensible position in the model's representational space
before unfreezing the body. Budget: 5-10B Korean tokens. Learning rate: 1e-4
to 3e-4 (higher than extension's 1e-4 because random init is further from the
target). LR schedule: cosine to min_lr=1e-5.

**Stage 4b (full continual)** — Unfreeze everything. Train on the standard
Korean 45% / English replay 45% / Parallel EN-KO 10% blend. Budget: 80-120B
tokens. Learning rate: start at 1e-5 (cautious for the first ~3k iters since
the random-init embeddings are still adjusting), ramp to 2-3e-5 once loss
stabilizes.

**Don't skimp on Stage 4a.** If you under-train the embedding warmup, when you
unfreeze the body you'll see the body weights catastrophically over-adjust to
compensate for poor Korean embeddings. This degrades English/Chinese capability
unnecessarily.

## Sanity checks for the replacement output

Things to verify after running both scripts:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("./qwen3-korean-replaced")

# 1. Some Korean token gets a NEW ID (not Qwen's old ID for that string)
print(tok.encode("안녕하십니까"))   # should produce some IDs; not necessarily the same as Qwen's

# 2. Round-trip works
text = "안녕하십니까"
assert tok.decode(tok.encode(text)) == text

# 3. Special tokens are at high IDs
print(tok.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"]))   # should be high

# 4. English still tokenizes
print(tok.encode("Hello, World!"))   # should be efficient; few tokens

# 5. The model loads with the new vocab size
model = AutoModelForCausalLM.from_pretrained("./qwen3-korean-replaced", torch_dtype="bfloat16")
print(model.get_input_embeddings().weight.shape[0] == tok.vocab_size)
```

If any of these fail, something went wrong in the replacement; do NOT proceed
to continual pretraining until they all pass.

## What about extension after replacement?

You can't easily mix the two. Once you've replaced, the model has random-init
Korean embeddings. If you then run extension on TOP of that, the "extension"
step would average-init new tokens from random-init constituents — producing
random outputs. Pick one path and stick with it.

## A nuance about base bytes

The 256 byte-level base tokens (IDs 0-255 in a typical BBPE vocab) are NEVER
removed by replacement. They're shared infrastructure across ALL languages —
Korean text decomposes into multi-byte UTF-8 sequences whose individual bytes
appear in English, Chinese, code, everything. The replacement script correctly
preserves these.

What IS removed: the merged tokens that Qwen's BPE produced specifically for
Korean byte sequences (e.g., the token that represents `안녕하십니까` as a single ID
rather than 18 individual UTF-8 bytes).

## Honest final word

The replacement path is unusual. Most LLM adaptation work uses extension. The
reasoning for choosing replacement should be a specific concrete belief about
your data, your model, or your research question — not an aesthetic preference
for "starting fresh". If you can't articulate the concrete reason in one
sentence, you probably want extension.

If you can articulate it (e.g., "Qwen was trained on a Korean corpus dominated
by Wikipedia translations and I have a curated literary/news corpus that I
want to shape the token distribution"), then replacement is reasonable and
the scripts in this directory implement it correctly.
