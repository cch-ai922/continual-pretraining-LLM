#!/usr/bin/env python3
"""
Stage 6 — DPO on the Korean preference pairs (Qwen3-8B).

DPO is small-N friendly (thousands, not millions), so it's cheap. LoRA on top of
the SFT model. Mix in any genuinely native-rated pairs you collect — they punch
far above their weight. Use the Megatron-Bridge RL/PEFT path, OR the simpler TRL
path on the exported HF checkpoint.
"""

CFG = dict(
    provider="qwen3",
    load="./mcore_sft",
    data=dict(train="ko_dpo_pairs.jsonl", seq_length=2048, chat_template="qwen"),
    dpo=dict(beta=0.1, loss="sigmoid"),
    train=dict(global_batch_size=64, micro_batch_size=1, epochs=1),
    scheduler=dict(lr=5e-6, min_lr=5e-7, lr_warmup_fraction=0.05, lr_decay_style="cosine"),
    peft=dict(scheme="lora", rank=16, alpha=32,
              target_modules="linear_qkv,linear_proj,linear_fc1,linear_fc2"),
    parallel=dict(tensor_model_parallel_size=2, pipeline_model_parallel_size=1),
    checkpoint=dict(save="./mcore_dpo", save_interval=200),
    bf16=True,
)

# --- Simpler alternative with TRL on the exported HF model -----------------------
TRL_ALTERNATIVE = r"""
# pip install trl>=0.12 transformers>=4.51 peft datasets
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

m = "./qwen3-ko-sft-hf"
tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(m, torch_dtype="bfloat16", trust_remote_code=True)
ds = load_dataset("json", data_files="ko_dpo_pairs.jsonl", split="train")
cfg = DPOConfig(beta=0.1, learning_rate=5e-6, per_device_train_batch_size=2,
                gradient_accumulation_steps=16, num_train_epochs=1, bf16=True,
                output_dir="./qwen3-ko-dpo")
DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok).train()
"""

if __name__ == "__main__":
    print("Use the Megatron-Bridge RL/PEFT path with CFG above, or the TRL "
          "alternative below for a simpler single-node run:\n")
    print(TRL_ALTERNATIVE)
