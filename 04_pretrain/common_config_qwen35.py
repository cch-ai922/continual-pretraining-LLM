#!/usr/bin/env python3
"""
Stage 4 common config — Qwen3.5-9B variant.

Drop-in replacement for `common_config.py` when using Qwen3.5-9B (hybrid GDN +
Gated Attention) as the base model instead of Qwen3-8B. Reflects the verified
architectural facts from the official HuggingFace model card.

CRITICAL — verify before training:
  1. transformers >= 5.2.0 installed
  2. Megatron-Bridge with qwen3_5 provider support
  3. Field names in this dict match your installed Megatron-Bridge version
     (Bridge minor releases occasionally rename keys; values are stable)

The architecture is fundamentally different from Qwen3-8B's uniform dense
attention: every block of 4 layers contains 3 Gated DeltaNet (linear attention)
layers followed by 1 Gated Attention (full attention) layer. Pattern repeats
8 times for a total of 32 layers.

If you're new to GDN: linear attention maintains fixed-size state matrices
instead of growing KV cache. This means dramatically lower memory at long
context (most layers don't grow with sequence length), but different
optimization dynamics — be more conservative with the initial learning rate
until you confirm training stability.
"""

# ---------------------------------------------------------------------------
# ARCHITECTURE  (Qwen3.5-9B, from the HF model card)
# ---------------------------------------------------------------------------
ARCH = {
    # Core dimensions
    "num_layers": 32,
    "hidden_size": 4096,
    "ffn_hidden_size": 12288,
    "vocab_size": 248320,             # padded; actual ≈ 246K
    "max_position_embeddings": 262144,

    # Gated Attention layers (1 in every 4) — these use standard full attention
    "num_attention_heads": 16,        # Q heads
    "num_query_groups": 4,            # KV heads (GQA 4:1)
    "kv_channels": 256,               # head_dim — larger than Qwen3-8B's 128
    "rope_theta": 1e6,
    "rope_dim": 64,                   # only applied to attention layers

    # Gated DeltaNet layers (3 in every 4) — linear attention
    "use_gated_deltanet": True,
    "linear_num_value_heads": 32,
    "linear_num_qk_heads": 16,
    "linear_head_dim": 128,

    # Layer composition pattern: 8 × (3 × GDN → 1 × Attention)
    # If your Megatron-Bridge version requires explicit per-layer types:
    "layer_types": (["gdn"] * 3 + ["attention"]) * 8,

    # Multi-Token Prediction head (a Qwen3.5 feature; disable aux loss for adaptation)
    "use_mtp": True,
    "mtp_steps": 1,
    "mtp_auxiliary_loss_weight": 0.0,   # set > 0 only if you want to train MTP

    # Activation / norm
    "activation": "swiglu",
    "norm_type": "rmsnorm",
    "norm_epsilon": 1e-6,

    # QK-norm (carried over from Qwen3)
    "qk_layernorm": True,

    # Embeddings: untied (same as Qwen3-8B)
    "tied_embeddings_and_output_weights": False,

    # Vision encoder weights are loaded but unused during text-only training.
    # Tell Bridge to skip vision-side conversion paths when applicable.
    "skip_vision_encoder": True,
}


# ---------------------------------------------------------------------------
# PARALLEL  (similar to Qwen3-8B, may need adjustment for memory profile)
# ---------------------------------------------------------------------------
# GDN layers have lower memory pressure at long context, so TP=2 / PP=1 often
# suffices even when training at seq_length = 32K. For seq_length ≥ 65K consider
# context-parallel as well.
PARALLEL = {
    "tensor_model_parallel_size": 2,
    "pipeline_model_parallel_size": 1,
    "context_parallel_size": 1,        # bump to 2-4 if training at ≥ 64K context
    "sequence_parallel": True,
}


# ---------------------------------------------------------------------------
# STAGE 4a — EMBEDDING-ONLY WARMUP
# ---------------------------------------------------------------------------
# If you DID extend the tokenizer (Stage 1), warm up only the new embedding
# rows + lm_head before unfreezing the rest. Skip Stage 4a if you didn't extend.
STAGE1 = {
    "tokens_total": int(2e9),          # 2B tokens
    "global_batch_size": 1024,
    "seq_length": 4096,
    "learning_rate": 1e-4,             # higher than full continual — embeddings only
    "min_lr": 1e-5,
    "lr_warmup_iters": 200,
    "lr_decay_style": "cosine",
    "weight_decay": 0.0,                # no decay on embeddings
    "freeze_params": "all_except_embeddings_and_lm_head",
    "bf16": True,
    "grad_clip": 1.0,
    "save_interval": 1000,
    "log_interval": 10,
}


# ---------------------------------------------------------------------------
# STAGE 4b — CONTINUAL PRETRAIN (full model)
# ---------------------------------------------------------------------------
# Korean 45% / English replay 45% / Parallel EN-KO 10% — see 00_data/build_data_mix.py
#
# LOWER initial LR than Qwen3-8B (2e-5) for the first ~3000 iters to confirm GDN
# training stability. Ramp to 2e-5 after that if loss curves look healthy. R1's
# adaptation work suggests GDN is more LR-sensitive in the first few thousand
# steps than full attention.
STAGE2 = {
    "tokens_total": int(60e9),         # 60B tokens
    "global_batch_size": 1024,
    "seq_length": 8192,                # GDN allows pushing to 32K+ if you want
    "learning_rate": 1e-5,             # conservative start
    "lr_after_ramp": 2e-5,             # ramp target after ~3000 iters
    "min_lr": 2e-6,
    "lr_warmup_iters": 500,            # longer warmup than Qwen3-8B
    "lr_decay_style": "cosine",
    "weight_decay": 0.01,
    "bf16": True,
    "grad_clip": 1.0,
    "save_interval": 2000,
    "log_interval": 10,
    # Watch for: GDN loss can be noisier than full attention in early steps;
    # if you see > 5% relative loss spikes in the first 1k iters, drop LR by 2x.
}


# ---------------------------------------------------------------------------
# DATA  (unchanged from Qwen3-8B config — model-agnostic)
# ---------------------------------------------------------------------------
DATA = {
    "data_path": "./pretrain_data_mix",
    "blend_weights": [0.45, 0.45, 0.10],   # Korean / EN-replay / Parallel
    "blend_paths": [
        "ko_pretrain_text_document",
        "en_replay_text_document",
        "parallel_enko_text_document",
    ],
    "tokenizer_type": "HuggingFaceTokenizer",
    "tokenizer_model": "Qwen/Qwen3.5-9B",  # or path to your extended version
}


# ---------------------------------------------------------------------------
# CHECKPOINTING
# ---------------------------------------------------------------------------
CHECKPOINT = {
    "load_path": "./mcore_qwen35_initial",  # output of Stage 3 conversion
    "save_path": "./mcore_qwen35_korean",
    "save_interval": 2000,
    "no_save_optim": False,                  # save optimizer for resumption
}


# ---------------------------------------------------------------------------
# SANITY: knobs that commonly bite
# ---------------------------------------------------------------------------
# 1. transformers >= 5.2.0 — otherwise model loads but produces garbage.
# 2. mtp_auxiliary_loss_weight = 0.0 unless you specifically want MTP training.
# 3. Don't filter vision tokens from loss — they're at zero frequency anyway.
# 4. If using flash-attn: confirm it supports GDN; older versions don't.
# 5. Megatron-Bridge `qwen3_5` provider name may differ; check installed version.

if __name__ == "__main__":
    import json
    print("=== Qwen3.5-9B architecture config ===")
    print(json.dumps(ARCH, indent=2, default=str))
    print("\n=== Stage-2 training config ===")
    print(json.dumps(STAGE2, indent=2))
    print("\nReminder: see QWEN35_MIGRATION_GUIDE.md at repo root for context.")
