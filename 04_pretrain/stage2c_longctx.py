#!/usr/bin/env python3
"""
Stage 4c — LONG-CONTEXT RETENTION at seq_length=32768 (run AFTER Stage 4b).

Purpose: keep Qwen3-8B's native 32K context alive. Stage 4b trains at seq 4096,
so positions 4096..32768 get zero gradient and long-range attention drifts. This
short phase resumes from the Stage-4b checkpoint and trains a small budget (~3-5B
tokens) on GENUINELY long documents (blend_long.yaml), exercising the long
positions WITHOUT relearning the language (low LR, brief).

Two hard requirements (both encoded in common_config.DATA_LONG / STAGE2C):
  1. reset_position_ids / reset_attention_mask = False — so long docs span the
     whole 32K window. Otherwise you train short-context at a long seq_length.
  2. Activation recompute ON and context_parallel_size>=2 — 32K activations are
     large; on 8xH200 (141GB) TP=2 + CP=2 + recompute fits an 8B comfortably.

Do NOT change rotary_base (θ=1e6 is correct for 32K). We RETAIN context here; we
do not extend it. Skip this stage only if you genuinely never need >4K context.

Megatron-Bridge recipe scaffold — confirm entrypoint/field names against your version.
"""
import common_config as C


def build_cfg():
    from megatron.bridge.recipes import qwen as qwen_recipes          # name may vary
    from megatron.bridge.training import ConfigContainer, pretrain    # name may vary

    cfg = qwen_recipes.qwen3_8b(seq_length=C.DATA_LONG["seq_length"], **C.PARALLEL_LONG)
    cfg.data.update(C.DATA_LONG)
    cfg.optimizer.update(C.OPT_COMMON)
    cfg.train.update({k: C.STAGE2C[k] for k in
                      ("train_iters", "global_batch_size", "micro_batch_size")})
    cfg.scheduler.update({
        "lr": C.STAGE2C["lr"], "min_lr": C.STAGE2C["min_lr"],
        "lr_warmup_fraction": C.STAGE2C["lr_warmup_fraction"],
        "lr_decay_style": C.STAGE2C["lr_decay_style"],
    })
    # 32K activation memory: recompute + sequence/context parallel.
    cfg.model.update(dict(
        recompute_granularity=C.STAGE2C["recompute_granularity"],
        recompute_method=C.STAGE2C["recompute_method"],
    ))

    if C.STAGE2C["use_lora"]:
        cfg.peft = dict(scheme="lora", rank=C.STAGE2C["lora_rank"],
                        alpha=C.STAGE2C["lora_alpha"],
                        target_modules=C.STAGE2C["lora_target"],
                        train_embeddings=True, train_output_layer=True)

    cfg.checkpoint.update(dict(
        load="./mcore_ckpt_final",        # resume from the Stage-4b result
        save="./mcore_ckpt_longctx",
        save_interval=500,
    ))
    cfg.eval = dict(eval_interval=500, eval_iters=50)  # watch Korean AND English
    return cfg, ConfigContainer, pretrain


if __name__ == "__main__":
    cfg, ConfigContainer, pretrain = build_cfg()
    pretrain(cfg if isinstance(cfg, ConfigContainer) else ConfigContainer(**cfg))
