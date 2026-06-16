#!/usr/bin/env python3
"""
Stage 4b — continual pretraining on the Korean + English-replay + parallel blend.

The main run (~60B tokens by default). Full-parameter or LoRA+embeddings. The
anti-forgetting levers are all here: LOW lr (2e-5) with a short re-warmup, healthy
English replay (set in blend.yaml), and parallel data for transfer.

Megatron-Bridge recipe scaffold — confirm entrypoint names against your version.
"""
import common_config as C


def build_cfg():
    from megatron.bridge.recipes import qwen as qwen_recipes          # name may vary
    from megatron.bridge.training import ConfigContainer, pretrain    # name may vary

    cfg = qwen_recipes.qwen3_8b(seq_length=C.ARCH["seq_length"], **C.PARALLEL)
    cfg.data.update(C.DATA)
    cfg.optimizer.update(C.OPT_COMMON)
    cfg.train.update({k: C.STAGE2[k] for k in
                      ("train_iters", "global_batch_size", "micro_batch_size")})
    cfg.scheduler.update({
        "lr": C.STAGE2["lr"], "min_lr": C.STAGE2["min_lr"],
        "lr_warmup_fraction": C.STAGE2["lr_warmup_fraction"],   # re-warmup from cold
        "lr_decay_style": C.STAGE2["lr_decay_style"],
    })

    if C.STAGE2["use_lora"]:
        cfg.peft = dict(
            scheme="lora",
            rank=C.STAGE2["lora_rank"], alpha=C.STAGE2["lora_alpha"],
            target_modules=C.STAGE2["lora_target"],
            train_embeddings=True, train_output_layer=True,  # keep embeddings full-trained
        )

    cfg.checkpoint.update(dict(
        load="./mcore_ckpt_stage1",   # or "./mcore_ckpt" if you skipped Stage 4a
        save="./mcore_ckpt_final",
        save_interval=1000,
    ))
    cfg.eval = dict(eval_interval=1000, eval_iters=50)  # watch English AND Korean
    return cfg, ConfigContainer, pretrain


if __name__ == "__main__":
    cfg, ConfigContainer, pretrain = build_cfg()
    pretrain(cfg if isinstance(cfg, ConfigContainer) else ConfigContainer(**cfg))
