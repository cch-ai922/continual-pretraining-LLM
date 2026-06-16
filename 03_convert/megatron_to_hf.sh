#!/usr/bin/env bash
# Stage 3 (reverse) — export a trained Megatron-Core checkpoint back to HF for
# SFT/DPO data generation, eval, and serving.
set -euo pipefail

MCORE_IN=${1:-./mcore_ckpt_final}
HF_OUT=${2:-./qwen3-ko-base-hf}

python -m megatron.bridge.convert \
    --source-format megatron \
    --target-format hf \
    --model "$MCORE_IN" \
    --provider qwen3 \
    --output "$HF_OUT"

# Copy the (extended) tokenizer alongside the exported weights:
cp -f ./qwen3-ko-tok/tokenizer*.json ./qwen3-ko-tok/*.model "$HF_OUT" 2>/dev/null || true
echo "HF checkpoint -> $HF_OUT"
