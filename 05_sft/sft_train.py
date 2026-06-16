#!/usr/bin/env python3
"""
Stage 5 — supervised fine-tuning of the Korean base into an instruct model (Qwen3-8B).

Megatron-Bridge finetune path with chat-formatted JSONL and loss masking on
assistant turns only. LoRA default; full FT if you have budget. Scaffold — confirm
entrypoint/field names against your Megatron-Bridge version.
"""

CFG = dict(
    provider="qwen3",
    load="./mcore_ckpt_final",                 # the continual-pretrained base
    data=dict(
        train="ko_sft_blend.jsonl",
        chat_template="qwen",                  # Qwen3 chat template (supports thinking tags)
        loss_mask="assistant_only",
        packing=True, seq_length=4096,
    ),
    train=dict(global_batch_size=128, micro_batch_size=1, epochs=3),
    scheduler=dict(lr=1e-5, min_lr=1e-6, lr_warmup_fraction=0.03, lr_decay_style="cosine"),
    optimizer=dict(optimizer="adam", weight_decay=0.0, clip_grad=1.0),
    parallel=dict(tensor_model_parallel_size=2, pipeline_model_parallel_size=1,
                  sequence_parallel=True),
    peft=dict(scheme="lora", rank=32, alpha=64,
              target_modules="linear_qkv,linear_proj,linear_fc1,linear_fc2"),
    checkpoint=dict(save="./mcore_sft", save_interval=500),
    eval=dict(eval_interval=500, eval_iters=50),
    bf16=True,
)


if __name__ == "__main__":
    from megatron.bridge.recipes import qwen as qwen_recipes        # name may vary
    from megatron.bridge.training import ConfigContainer, finetune  # name may vary
    cfg = qwen_recipes.qwen3_8b(**CFG["parallel"])
    for k, v in CFG.items():
        if k == "parallel":
            continue
        (getattr(cfg, k).update(v) if isinstance(v, dict) and hasattr(cfg, k)
         else cfg.update({k: v}))
    finetune(cfg)
