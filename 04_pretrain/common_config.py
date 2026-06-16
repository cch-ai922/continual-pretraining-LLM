#!/usr/bin/env python3
"""
Shared config for the continual-pretraining stages (Megatron-Bridge), Qwen3-8B.

Plain dicts capturing the README decisions. Wire into the Bridge ConfigContainer /
recipe for your version; field names mirror common Megatron-Core options — confirm
against your install.
"""

# Exact Qwen3-8B architecture (Bridge auto-detects from HF, listed here for clarity
# / for vanilla Megatron-LM users who must set these by hand).
ARCH = dict(
    num_layers=36,
    hidden_size=4096,
    ffn_hidden_size=12288,
    num_attention_heads=32,
    num_query_groups=8,          # GQA: 8 KV heads
    kv_channels=128,             # head_dim
    normalization="RMSNorm",
    qk_layernorm=True,           # Qwen3 QK-Norm
    gated_linear_unit=True,      # SwiGLU
    activation="silu",
    add_bias_linear=False,
    add_qkv_bias=False,          # Qwen3 uses QK-norm, not qkv bias
    position_embedding_type="rope",
    rotary_base=1_000_000.0,
    layernorm_epsilon=1e-6,
    vocab_size=151936,           # +N if you extended; the resized ckpt carries the real size
    share_embeddings_and_output_weights=False,   # UNTIED
    seq_length=4096,
    bf16=True,
)

PARALLEL = dict(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    sequence_parallel=True,
)

DATA = dict(
    blend_yaml="blend.yaml",     # produced by 00_data/build_data_mix.py
    split="998,1,1",
    seq_length=4096,
)

OPT_COMMON = dict(
    optimizer="adam",
    adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8,
    weight_decay=0.1,
    clip_grad=1.0,
)

# ---- Stage 4a: embedding-only warmup (only if you extended the vocab) ----
STAGE1 = dict(
    train_iters=2000,
    global_batch_size=256,
    micro_batch_size=2,
    lr=1e-4, min_lr=1e-5,                 # embeddings can move fast; body is frozen
    lr_warmup_fraction=0.02,
    lr_decay_style="cosine",
    freeze_transformer_body=True,         # train ONLY embeddings + lm_head
)

# ---- Stage 4b: continual pretrain (full or LoRA + embeddings) ----
STAGE2 = dict(
    train_iters=14000,                    # ~60B tokens at GBS=1024 x 4096
    global_batch_size=1024,
    micro_batch_size=1,
    lr=2e-5, min_lr=2e-6,                 # LOW + re-warmup -> the anti-forgetting knob
    lr_warmup_fraction=0.015,
    lr_decay_style="cosine",
    freeze_transformer_body=False,
    use_lora=False,                       # set True for the cheaper variant
    lora_rank=64, lora_alpha=128,
    lora_target="linear_qkv,linear_proj,linear_fc1,linear_fc2",  # attn + MLP
)

# ---- Stage 4c: LONG-CONTEXT RETENTION (seq 32768) ----
# Why: Stage 4b at seq 4096 never sends gradient to positions 4096..32768, so
# Qwen3-8B's native 32K ability erodes. This short phase exercises the long
# positions on GENUINELY long documents (blend_long.yaml from build_data_mix.py),
# starting from the Stage-4b checkpoint. Do NOT change rotary_base — θ=1e6 is
# already correct for 32K; we are RETAINING context, not extending it.
PARALLEL_LONG = dict(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=1,
    context_parallel_size=2,      # split the 32K sequence across GPUs (needed at 32K)
    sequence_parallel=True,
)
DATA_LONG = dict(
    blend_yaml="blend_long.yaml", # long-doc mix; seq_length carried in the yaml
    split="998,1,1",
    seq_length=32768,
    # CRITICAL: keep whole long documents intact across the 32K window. If your
    # packer resets positions/attention per document, each doc gets fresh RoPE
    # positions and cannot attend across boundaries -> you are NOT training
    # long-context even at seq 32768. Disable the resets for this bucket.
    reset_position_ids=False,
    reset_attention_mask=False,
    eod_mask_loss=False,
)
STAGE2C = dict(
    train_iters=2000,                     # ~4B tokens at GBS=64 seqs x 32768
    global_batch_size=64,                 # in SEQUENCES (64 x 32768 ~= 2.1M tokens)
    micro_batch_size=1,
    lr=1e-5, min_lr=1e-6,                 # gentle: we are retaining, not relearning
    lr_warmup_fraction=0.02,
    lr_decay_style="cosine",
    freeze_transformer_body=False,
    recompute_granularity="full",         # activation recompute: 32K activations are large
    recompute_method="uniform",
    use_lora=False,                       # LoRA+frozen-attn is the lowest-drift variant
    lora_rank=64, lora_alpha=128,
    lora_target="linear_qkv,linear_proj,linear_fc1,linear_fc2",
)
