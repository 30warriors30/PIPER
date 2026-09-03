#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/yanlu/BREW/models/facebook/opt-1.3b}"
RUN_ID="${RUN_ID:-smoke_c4_opt13b}"

cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python run_generation.py \
  --model-path "$MODEL_PATH" \
  --secret-key "smoke-test-key" \
  --dataset c4 \
  --max-samples 3 \
  --max-new-tokens 32 \
  --generation-batch-size 3 \
  --presence-mode soft \
  --delta-presence 0 \
  --delta-payload 2 \
  --run-id "$RUN_ID" \
  --resume

python run_detection.py \
  --input-type samples \
  --run-dir "outputs/$RUN_ID" \
  --presence-test exact_binomial \
  --counting-mode unique_context \
  --primary-decoding-policy tie_zero \
  --max-erasure-assignments 64 \
  --resume

python evaluate_results.py --run-dir "outputs/$RUN_ID" --input-mode blind_text --decoding-policy tie_zero
cat "outputs/$RUN_ID/summary.txt"
