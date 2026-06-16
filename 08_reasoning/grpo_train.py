#!/usr/bin/env python3
"""
Stage 8 — GRPO training on verified math reward (the "R1-style" recipe).

Group Relative Policy Optimization: for each prompt, sample G completions, score
each with the VERIFIABLE reward (math correctness), and use within-group relative
advantage as the policy gradient signal. No reward model needed — the verifier IS
the reward, which is the whole point of RLVR.

Two practical paths:

  (A) TRL's GRPOTrainer (simplest, single-node, runs on the exported HF checkpoint).
      Recommended for a first run — that template is below.

  (B) Megatron-Bridge RL (scales to multi-node, integrates with your pretraining
      stack). The recipe shape is the same; check the Bridge RL docs for the exact
      entrypoint for your installed version.
"""

# ============================================================================
# (A) TRL GRPO  — copy and adapt
# ============================================================================
TRL_GRPO_TEMPLATE = r'''
# pip install "trl>=0.14" "transformers>=4.51" datasets accelerate vllm
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
from verify_math import extract_final_answer, numeric_equal
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "./qwen3-ko-sft-hf"          # Stage-5 instruct model (HF format)
DATA  = "ko_math_problems.jsonl"     # JSONL: {"problem": "...", "answer": "..."}
OUT   = "./qwen3-ko-rlvr"

# 1. dataset: each row needs a "prompt" field for TRL GRPO
ds = load_dataset("json", data_files=DATA, split="train")
SYSTEM_KO = ("당신은 수학교원입니다. 문제를 단계별로 풀고 "
             "마지막에 적으시오: '최종답: <수자>'.")
def to_prompt(r):
    return {"prompt": [{"role": "system", "content": SYSTEM_KO},
                       {"role": "user",   "content": r["problem"]}],
            "answer": str(r["answer"])}
ds = ds.map(to_prompt)

# 2. REWARD FUNCTIONS  -- this is the heart of RLVR
def correctness_reward(completions, answer, **_):
    """Primary signal: 1.0 if the extracted final answer matches gold, else 0.0."""
    out = []
    for comp, gold in zip(completions, answer):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        out.append(1.0 if numeric_equal(extract_final_answer(text), gold) else 0.0)
    return out

def format_reward(completions, **_):
    """Shaping: 0.1 bonus if the response contains the '최종 답:' marker."""
    out = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else comp
        out.append(0.1 if "최종답" in text else 0.0)
    return out

# 3. GRPO config
cfg = GRPOConfig(
    output_dir=OUT,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_generations=8,                 # G = group size; samples per prompt
    max_prompt_length=512,
    max_completion_length=1024,
    temperature=0.9,
    learning_rate=5e-6,
    num_train_epochs=1,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    # use vLLM for fast group sampling if available:
    use_vllm=True, vllm_mode="server",
)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="bfloat16",
                                             trust_remote_code=True)

trainer = GRPOTrainer(
    model=model,
    processing_class=tok,
    reward_funcs=[correctness_reward, format_reward],
    train_dataset=ds,
    args=cfg,
)
trainer.train()
trainer.save_model(OUT)
'''


# ============================================================================
# (B) Megatron-Bridge RL  — sketch
# ============================================================================
BRIDGE_TEMPLATE = r"""
# Pseudocode for the Bridge RL path. Confirm field names against your version.
#
# from megatron.bridge.recipes import qwen as qwen_recipes
# from megatron.bridge.training import ConfigContainer, rl_train
# from verify_math import extract_final_answer, numeric_equal
#
# def reward_fn(prompt, completion, meta):
#     return 1.0 if numeric_equal(extract_final_answer(completion), meta["answer"]) else 0.0
#
# cfg = qwen_recipes.qwen3_8b(tensor_model_parallel_size=2)
# cfg.rl.algorithm = "grpo"
# cfg.rl.num_generations = 8
# cfg.rl.reward_fn = reward_fn
# cfg.data.train = "ko_math_problems.jsonl"
# cfg.scheduler.lr = 5e-6
# cfg.checkpoint.load = "./mcore_sft"
# cfg.checkpoint.save = "./mcore_rlvr"
# rl_train(ConfigContainer(**cfg))
"""

if __name__ == "__main__":
    print("=== Path A: TRL GRPOTrainer (simplest, single-node) ===")
    print(TRL_GRPO_TEMPLATE)
    print("\n=== Path B: Megatron-Bridge RL (multi-node) ===")
    print(BRIDGE_TEMPLATE)
