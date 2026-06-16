#!/usr/bin/env bash
# Stage 0 — tokenize raw JSONL text into Megatron-Core .bin/.idx datasets.
# Each input is JSONL with a {"text": "..."} field per line.
# Use the SAME tokenizer you will train with (extended one if you extended).
set -euo pipefail

# --- edit these ---
TOKENIZER_PATH=${TOKENIZER_PATH:-./qwen3-ko-tok}     # or Qwen/Qwen3-8B if not extended
MBRIDGE=${MBRIDGE:-/opt/Megatron-Bridge}              # repo / install root
OUT=/data/mcore
WORKERS=${WORKERS:-64}
mkdir -p "$OUT"

# Megatron-Core ships tools/preprocess_data.py. Megatron-Bridge wraps the same
# IndexedDataset builder. Adjust the path to whichever you have installed.
PREPROCESS=${PREPROCESS:-"$MBRIDGE/Megatron-LM/tools/preprocess_data.py"}

preprocess () {
  local in_jsonl=$1 out_prefix=$2
  python "$PREPROCESS" \
    --input "$in_jsonl" \
    --json-keys text \
    --output-prefix "$out_prefix" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$TOKENIZER_PATH" \
    --append-eod \
    --workers "$WORKERS"
}

# NOTE: Megatron's preprocess_data.py names outputs "<output-prefix>_<json-key>_document.{bin,idx}".
# With --json-keys text, prefix "ko_mono" produces "ko_mono_text_document" — which is exactly
# what build_data_mix.py's SOURCES expect. Do NOT add a trailing "_text" to the prefix here
# (that would produce "ko_mono_text_text_document" and the blend paths would not be found).
preprocess /data/raw/ko_mono.jsonl        "$OUT/ko_mono"
preprocess /data/raw/en_mono.jsonl        "$OUT/en_mono"
# For parallel data, format each pair as one document. build_parallel_jsonl.py
# (below) emits {"text": "<en> ... \n<hi> ..."} or an instruction template.
preprocess /data/raw/parallel_enko.jsonl  "$OUT/parallel_enko"

echo "Done. Datasets in $OUT (use the *_text_document prefixes in blend.yaml)."
