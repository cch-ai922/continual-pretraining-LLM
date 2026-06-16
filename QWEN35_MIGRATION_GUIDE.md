# Guide — Migrating from Qwen3-8B to Qwen3.5-9B as Base

The deeper-base swap: use Qwen3.5-9B (the natively multimodal, hybrid GDN
architecture model) as the starting point instead of Qwen3-8B. The capability
gain is real; the engineering cost is real too. This guide lays out exactly
what changes and what stays the same.

## Verified architectural facts

From the official Qwen3.5-9B HuggingFace model card (Feb 2026 release):

| Property                     | Qwen3-8B                        | Qwen3.5-9B                                  |
| ---------------------------- | ------------------------------- | ------------------------------------------- |
| Total parameters             | 8B                              | 9B (text + vision encoder)                  |
| Layers                       | 36                              | **32**                                      |
| Hidden dimension             | 4096                            | 4096                                        |
| FFN intermediate             | 12288                           | 12288                                       |
| Vocabulary                   | 151,936                         | **248,320** (padded)                        |
| Layer composition            | uniform full attention          | **3:1 GDN to Gated Attention**              |
| Standard attention heads     | 32 Q / 8 KV (GQA), head_dim 128 | 16 Q / 4 KV (GQA), head_dim **256**         |
| Linear attention (GDN) heads | —                               | 32 V-heads / 16 QK-heads, head_dim 128      |
| RoPE dimension               | 128                             | **64** (only on attention layers)           |
| Multi-token prediction       | no                              | **yes (MTP head)**                          |
| Vision                       | none                            | **early-fusion, baked into shared weights** |
| Native context               | 32K                             | **262K** (extensible to ~1M)                |
| Languages in pretraining     | 119                             | **201**                                     |
| Required `transformers`      | ≥ 4.51                          | **≥ 5.2.0**                                 |
| Megatron-Bridge provider     | `qwen3`                         | `qwen3_5`                                   |
| License                      | Apache 2.0                      | Apache 2.0                                  |

The exact layer pattern is `8 × (3 × (GDN → FFN) → 1 × (Gated Attention → FFN))`
— every block of 4 layers has 3 linear-attention layers followed by 1 full
attention layer.

## Why someone would make this swap

The honest case for Qwen3.5-9B as base for a Korean adaptation:

1. **Stronger base capability transfers to stronger adapted model**. Qwen3.5-9B
   beats Qwen3-8B by a wide margin on MMLU-Pro (82.5 vs ~71), GPQA Diamond
   (81.7 vs much lower), and most reasoning benchmarks. The latent reasoning
   you'll extract via STaR-bootstrap and self-consistency is correspondingly
   higher.

2. **Better Korean coverage at start**. 201 languages in pretraining (vs 119)
   means more Korean data was seen. Measure fertility to confirm — you may
   need much less vocabulary extension than for Qwen3-8B.

3. **Long context out of the box**. 262K native context (vs Qwen3-8B's 32K) is
   useful for grounded reasoning over long passages, document Q&A, code at
   repo scale. You won't always use it, but it's there.

4. **Better cross-lingual transfer**. Larger pretraining + more languages means
   the model has stronger latent representations to draw on for Korean tasks.

## Why someone would NOT make this swap

The honest case against:

1. **Hybrid GDN architecture is less battle-tested for adaptation**. The
   Megatron-Bridge `qwen3_5` provider exists and works, but recipes for
   non-English adaptation specifically are scarce. You're partly in
   exploration territory.

2. **Vision parameters you'll never use cost disk/RAM**. About 1-2GB of the
   9B is the vision encoder. You load it, ignore it, but pay the load cost.

3. **Multi-Token Prediction adds complexity**. The MTP head needs to be
   handled deliberately during continual pretraining — typically you'll
   disable the auxiliary MTP loss for adaptation, but if you want to
   preserve the speculative-decoding advantage at inference time, you have
   to be careful.

4. **Framework version churn**. `transformers >= 5.2.0` is a major version
   jump. Ecosystem tools (vLLM, TGI, accelerate, peft) may lag the official
   transformers release. Pin everything carefully or expect surprises.

5. **The capability gain is partly knowledge depth, partly multimodal**.
   For pure-text Korean adaptation, you're paying for capability you won't
   exercise (multimodal reasoning, image understanding, video processing).
   The pure-text portion of Qwen3.5-9B's advantage over Qwen3-8B is smaller
   than the headline benchmark gap suggests.

## Honest recommendation

**If your goal is a working Korean LLM and your compute budget is finite**:
stick with Qwen3-8B. The pipeline is more predictable, the recipes are more
mature, the framework versions are stable, and you can iterate faster. The
capability gap to Qwen3.5-9B will be partly closed by your adaptation work
anyway.

**If your goal is the strongest possible Korean model with no other constraints**:
Qwen3.5-9B is a defensible choice. You'll have to do more engineering on the
early stages, but the higher latent capability does propagate through.

For a first pass through this pipeline, I'd actually recommend doing **both** —
run the Qwen3-8B pipeline first to validate the recipe end-to-end on your data,
then swap to Qwen3.5-9B once everything is working. The 4-week timeline saved
by using the simpler base for the first iteration is usually worth more than
the capability gain on attempt one.

## What "suppressing vision" actually means

Qwen3.5 is early-fusion VL, not modular. There is no clean way to extract a
"pure text" Qwen3.5-9B because the vision tokens were processed through the
_same_ transformer layers as text tokens during pretraining.

What you can do:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
# Standard load. The model has both text and vision processing pathways
# but routes automatically based on what inputs you pass.
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-9B",
    torch_dtype="bfloat16",
    trust_remote_code=True,
)

# Text-only forward pass. No `pixel_values`, no `image_grid_thw`.
# The vision encoder weights are loaded but compute nothing.
input_ids = tok("어떤 조선어질문", return_tensors="pt").input_ids
out = model(input_ids)
```

That's all "suppression" means in practice. You pass only text; the model's
forward pass routes through the text-only code path; the vision parameters sit
idle. Continued pretraining proceeds identically.

There is no `--no-vision` flag. The vision encoder weights occupy disk and RAM
but consume no FLOPs during text-only forward passes.

## What changes at each pipeline stage

### Stage 0 — Data audit

Minor change. Re-run `measure_fertility.py` with the Qwen3.5 tokenizer:

```bash
python 00_data/measure_fertility.py --tokenizer Qwen/Qwen3.5-9B --corpus ko_sample.txt
```

Expect noticeably better Korean fertility than Qwen3-8B because of the 248K vocab
and 201-language training. **This may flip your Stage 1 decision.**

### Stage 1 — Tokenizer extension

Probably skip. The decision rule from your existing guide ("extend if fertility

> 2.0 tokens/word on Korean") almost certainly comes out negative for Qwen3.5-9B.
> If fertility comes in at ~1.3-1.5 tokens/word, the cost of extending vocabulary
> exceeds the benefit. Verify with measurement.

### Stage 2 — Embedding init

Same logic, larger matrix. If you DO extend the vocabulary in Stage 1, the
existing `init_new_embeddings.py` works unchanged — it operates on the
embedding matrix regardless of size. If you skip Stage 1, you skip Stage 2 too.

### Stage 3 — HF → Megatron conversion

Different Megatron-Bridge provider:

```bash
python -m megatron.bridge.convert \
    --hf-path Qwen/Qwen3.5-9B \
    --output ./mcore_qwen35 \
    --provider qwen3_5         # not 'qwen3'
```

Confirm the exact provider string against your installed `megatron-bridge`
version — naming conventions sometimes shift between minor releases.

### Stage 4 — Continual pretraining

Most affected stage. See `04_pretrain/common_config_qwen35.py` for the
architecture-aware config. Key differences from the Qwen3-8B config:

- **GDN layers** have linear attention; their gradient flow and convergence
  properties differ from standard attention. Use slightly lower LR (1e-5 to
  2e-5 instead of 2e-5 to 4e-5) for the first few thousand steps to confirm
  stability, then ramp.

- **MTP head**: typically disabled for continual pretraining. Set
  `mtp_auxiliary_loss_weight: 0.0` in the config. You can re-enable it at the
  end for a brief MTP-aware fine-tune if you want speculative decoding at
  inference, but most adaptations skip this.

- **KV cache memory**: dramatically lower than for full-attention Qwen3-8B
  at long context because 3/4 of layers are GDN (which has fixed-size state).
  This is an advantage if you want to train at longer sequences than 8K, but
  also means batch sizes that worked for Qwen3-8B may underutilize memory.

- **Vision tokens in the vocab**: the special tokens for vision (`<image>`,
  `<vision_start>`, `<vision_end>`, etc.) are at known IDs in the vocabulary.
  Your Korean text corpus won't contain these, so they'll appear at zero
  frequency during continual pretraining. This is fine — their embeddings
  drift slightly but don't actively degrade. Don't bother filtering them
  from the loss.

### Stage 5 — SFT

Chat template is different. Use `tokenizer.apply_chat_template()` to get the
right format; everything else (translation, grounded generation, faithfulness
scoring) works as-is. The 9 grounded task types in `generate_grounded_ko.py`
apply identically — they're tokenizer-level operations.

### Stage 6 — DPO

Unchanged. Pure model-agnostic.

### Stage 7 — Teacher loop

Unchanged. All five scripts (self-instruct, AI-feedback judge, distill from
teacher, self-consistency, critique-revise, few-shot bootstrap) work
identically. The bootstrap-from-base will likely yield higher solve rates on
the first round than with Qwen3-8B, because the base model is stronger.

### Stage 8 — RLVR

Unchanged. All five verifiers, all nine converters, all GRPO training work
identically. The capability gain shows up here as faster convergence — the
base model's solve rate at K=8 sampling is higher, so the verified pool
fills faster.

## Updated Stage-4 architecture knobs

Replace your `04_pretrain/common_config.py` ARCH dict with the Qwen3.5-9B
values. See `04_pretrain/common_config_qwen35.py` for the full config; the key
deltas from Qwen3-8B:

```python
ARCH = {
    "num_layers": 32,              # was 36
    "hidden_size": 4096,           # same
    "ffn_hidden_size": 12288,      # same
    "num_attention_heads": 16,     # was 32 (in the attention layers)
    "num_query_groups": 4,         # was 8 (GQA group count in attn layers)
    "kv_channels": 256,            # was 128 (head_dim in attention layers)
    "vocab_size": 248320,          # was 151936

    # NEW: GDN-specific knobs (not present in Qwen3-8B)
    "use_gated_deltanet": True,
    "linear_num_value_heads": 32,
    "linear_num_qk_heads": 16,
    "linear_head_dim": 128,
    "layer_pattern": "8x(3*GDN+1*ATT)",   # explicit pattern marker

    # NEW: MTP head
    "use_mtp": True,
    "mtp_steps": 1,                # depth of MTP prediction
    "mtp_auxiliary_loss_weight": 0.0,  # disabled for adaptation

    # Tokenizer / RoPE
    "rope_theta": 1e6,
    "rope_dim": 64,                # was 128
    "max_position_embeddings": 262144,
}
```

Confirm the exact field names against the Megatron-Bridge `qwen3_5` provider
in your installed version — different releases use slightly different
naming. The values are stable; the field names occasionally shift.

## Practical compute budget impact

Memory at BF16 (rough estimates):

- Qwen3-8B: ~16GB for weights, modest KV cache → ~24GB total at batch=4 seq=8K
- Qwen3.5-9B: ~18GB for weights + 2GB vision encoder + lower KV cache (GDN) →
  ~22-26GB total at batch=4 seq=8K; **much lower at long context**

Throughput: Qwen3.5-9B's 32 layers (vs 36) gives ~10% less compute per token,
roughly offset by the larger vocab. Net wall-clock per token is similar.

Continual pretraining cost: same budget as Qwen3-8B (~60B tokens for stable
adaptation). Don't expect to need less because the base is stronger — the
distribution shift is roughly the same.

## What to actually do

1. **Don't blindly swap.** Run `measure_fertility.py` first. If Qwen3.5-9B's
   tokenizer already gives ≤ 1.5 tokens/word on your Korean corpus, this swap
   saves you the entire Stage 1 + Stage 2 effort.

2. **Validate the recipe end-to-end on Qwen3-8B first.** A working pipeline
   on the simpler base is worth more than a half-debugged pipeline on the
   stronger base. Swap once Stage 4-8 are verified.

3. **Plan for transformers 5.2.0 ecosystem catch-up time.** Pin all
   dependencies to versions known to work with transformers 5.2; expect to
   wait 2-3 weeks for vLLM/TGI/PEFT compatible releases after the official
   transformers cut.

4. **Skip the vision suppression scripting.** There's nothing to script — you
   just don't pass image inputs. Don't waste time trying to extract a
   "pure text" model; it doesn't exist as a clean object.

5. **Be honest about the capability gap.** The benchmarks comparing Qwen3.5-9B
   to Qwen3-8B include multimodal tasks where Qwen3-8B can't even compete.
   The pure-text gap is real but smaller than the headline numbers suggest.
   For Korean-only adaptation, you'll see maybe 5-15% relative improvement
   on downstream tasks, not 50%+.

## Bottom line

Qwen3.5-9B as base is the strongest choice if you can absorb the engineering
overhead — better latent capability, broader language coverage, longer
context. The penalty is a less mature pipeline, framework version churn, and
1-2GB of vision parameters you'll load and ignore. For a first-pass Korean
adaptation, Qwen3-8B is the safer choice; for a long-term production model,
Qwen3.5-9B has the upside.

If you proceed with the swap, the architecture-specific changes live in
Stages 0-4. Stages 5-8 work identically because the data generation pipeline
is model-agnostic.
