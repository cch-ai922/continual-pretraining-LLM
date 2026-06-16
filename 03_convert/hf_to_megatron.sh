#!/usr/bin/env bash
# Stage 3 — convert an HF Qwen3-8B checkpoint to Megatron-Core format.
#
# Qwen3 is a standard dense decoder, so you have TWO clean options:
#   (A) Megatron-Bridge AutoBridge (recommended; auto-detects Qwen3 config)
#   (B) vanilla Megatron-LM's Qwen/HF converter (works because the arch is standard)
# Confirm flags against your installed version.
set -euo pipefail

HF_PATH=${1:-./qwen3-ko-init}        # or Qwen/Qwen3-8B if you did NOT extend the vocab
MCORE_OUT=${2:-./mcore_ckpt}
TP=${TP:-2}; PP=${PP:-1}

# --- Option A: Megatron-Bridge ---
python -m megatron.bridge.convert \
    --source-format hf \
    --target-format megatron \
    --model "$HF_PATH" \
    --provider qwen3 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --output "$MCORE_OUT"

# --- Option B (alternative): vanilla Megatron-LM ---
# python Megatron-LM/tools/checkpoint/convert.py \
#     --model-type GPT --loader llama_mistral --saver mcore \
#     --model-size qwen3-8B --checkpoint-type hf \
#     --load-dir "$HF_PATH" --save-dir "$MCORE_OUT" \
#     --tokenizer-model "$HF_PATH" --target-tensor-parallel-size "$TP"

echo "Megatron-Core checkpoint -> $MCORE_OUT"
