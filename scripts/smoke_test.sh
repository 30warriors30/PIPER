#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to a local model directory}"
: "${C4_PATH:?Set C4_PATH to a processed C4 JSONL file}"
SECRET_KEY="${SECRET_KEY:-dual-layer-key-2026}"
RUN_ID="${RUN_ID:-piper_selfhash_smoke}"
BATCH_SIZE="${BATCH_SIZE:-3}"

cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
[[ -d "$MODEL_PATH" ]] || { echo "MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 1; }
[[ -f "$C4_PATH" ]] || { echo "C4_PATH does not exist: $C4_PATH" >&2; exit 1; }

mode_args=(--resume)
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  mode_args=(--overwrite)
fi

python run_generation.py \
  --model-path "$MODEL_PATH" --local-files-only \
  --secret-key "$SECRET_KEY" --dataset c4 --dataset-path "$C4_PATH" \
  --max-samples 3 --max-new-tokens 64 --generation-batch-size "$BATCH_SIZE" \
  --context-width 4 --candidate-top-k 50 \
  --seeding-scheme selfhash --partition-engine v2 \
  --temperature 1.0 --top-k 50 --top-p 0.95 \
  --presence-mode soft --delta-presence 0 --delta-payload 2 \
  --ecc-n 23 --ecc-k 8 --ecc-t 3 \
  --presence-test exact_binomial --target-fpr 0.01 \
  --counting-mode unique_context --primary-decoding-policy tie_zero \
  --run-id "$RUN_ID" "${mode_args[@]}"

python run_detection.py \
  --input-type samples --run-dir "outputs/$RUN_ID" \
  --presence-test exact_binomial --target-fpr 0.01 \
  --counting-mode unique_context --primary-decoding-policy tie_zero \
  "${mode_args[@]}"

python evaluate_results.py \
  --run-dir "outputs/$RUN_ID" --input-mode blind_text \
  --decoding-policy tie_zero

cat "outputs/$RUN_ID/summary.txt"
