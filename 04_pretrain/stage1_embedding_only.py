#!/usr/bin/env python3
"""
Stage 4a — embedding-only warmup (run ONLY if you extended the vocab).

Freeze the entire transformer body and train just the embedding matrices so the
freshly-initialized Korean tokens settle into the model's representation space
WITHOUT disturbing its English/knowledge. Cheap and safe.

Megatron-Bridge recipe scaffold — confirm entrypoint/field names against your
installed version (the canonical pattern: build a ConfigContainer from provider +
optimizer + scheduler + data, then call the bridge pretrain()).
"""
from dataclasses import dataclass
from typing import Optional

import common_config as C


def build_cfg():
    import torch.nn as nn
    from megatron.bridge.recipes import qwen as qwen_recipes          # name may vary
    from megatron.bridge.training import ConfigContainer, pretrain    # name may vary
    from megatron.bridge.peft.base import PEFT

    # Bridge has no `cfg.model.freeze` for LLMs (see NVIDIA-NeMo/NeMo#14462 —
    # closed "not planned"). The only DDP-safe path is a PEFT subclass whose
    # freeze_model() flips requires_grad selectively; transform() is a no-op
    # because we are not inserting any adapter modules.
    @dataclass
    class EmbeddingOnlyFreeze(PEFT):
        def freeze_model(self, model, training: bool = True) -> None:
            chunks = model if isinstance(model, list) else [model]
            for chunk in chunks:
                for name, param in chunk.named_parameters():
                    train_this = ("embedding" in name) or ("output_layer" in name)
                    param.requires_grad = train_this
                    if train_this:
                        self.params_to_save.add(name)

        def transform(self, module: nn.Module, name: Optional[str] = None,
                      prefix: Optional[str] = None) -> nn.Module:
            return module

    cfg = qwen_recipes.qwen3_8b(seq_length=C.ARCH["seq_length"], **C.PARALLEL)
    cfg.data.update(C.DATA)
    cfg.optimizer.update(C.OPT_COMMON)
    cfg.train.update({k: C.STAGE1[k] for k in
                      ("train_iters", "global_batch_size", "micro_batch_size")})
    cfg.scheduler.update({
        "lr": C.STAGE1["lr"], "min_lr": C.STAGE1["min_lr"],
        "lr_warmup_fraction": C.STAGE1["lr_warmup_fraction"],
        "lr_decay_style": C.STAGE1["lr_decay_style"],
    })
    # Freeze everything except embeddings + output layer.
    cfg.peft = EmbeddingOnlyFreeze()
    # PEFT requires `pretrained_checkpoint` to load the frozen base; only the
    # unfrozen params (embeddings + output_layer) are written to `save`.
    cfg.checkpoint.update(dict(pretrained_checkpoint="./mcore_ckpt",
                              save="./mcore_ckpt_stage1",
                              save_interval=500))
    return cfg, ConfigContainer, pretrain


if __name__ == "__main__":
    cfg, ConfigContainer, pretrain = build_cfg()
    pretrain(cfg if isinstance(cfg, ConfigContainer) else ConfigContainer(**cfg))
