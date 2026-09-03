#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/yanlu/BREW/models/facebook/opt-1.3b}"
RUN_ID="${RUN_ID:-c4_opt13b_piper_200}"
SECRET_KEY="${SECRET_KEY:-dual-layer-key-2026}"
BATCH_SIZE="${BATCH_SIZE:-16}"

cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python run_generation.py \
  --model-path "$MODEL_PATH" \
  --secret-key "$SECRET_KEY" \
  --dataset c4 \
  --dataset-name allenai/c4 \
  --dataset-config realnewslike \
  --dataset-split validation \
  --streaming \
  --max-samples 200 \
  --max-new-tokens 200 \
  --generation-batch-size "$BATCH_SIZE" \
  --presence-mode soft \
  --delta-presence 0 \
  --delta-payload 2 \
  --prf-mode paper_shared \
  --partition-mode exact_permutation \
  --allocation-mode hash_mod \
  --ecc-n 23 \
  --ecc-k 8 \
  --ecc-t 3 \
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
