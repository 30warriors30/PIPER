#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to a local OPT-1.3B checkpoint directory}"
: "${C4_PATH:?Set C4_PATH to the processed C4 RealNewsLike JSONL file}"
SECRET_KEY="${SECRET_KEY:-dual-layer-key-2026}"
EXPERIMENT_ID="${EXPERIMENT_ID:-piper_main_t300_b8_selfhash_top50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/experiments}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"

cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  [[ -d "$MODEL_PATH" ]] || { echo "MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 1; }
  [[ -f "$C4_PATH" ]] || { echo "C4_PATH does not exist: $C4_PATH" >&2; exit 1; }
fi

mode_args=(--resume)
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  mode_args=(--overwrite)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  mode_args=(--dry-run)
fi

python -m experiments.run_pareto_sweep \
  --model-path "$MODEL_PATH" --secret-key "$SECRET_KEY" \
  --experiment-id "$EXPERIMENT_ID" --output-root "$OUTPUT_ROOT" \
  --preset paper --only-point soft_p0_m2 --stage all \
  --calibration-samples 1000 --test-samples 5000 --limit-test 1000 \
  --exact-tokens 300 --context-width 4 \
  --ecc-n 23 --ecc-k 8 --ecc-t 3 \
  --prf-mode paper_shared --allocation-mode hash_mod \
  --candidate-top-k 50 --seeding-scheme selfhash --partition-engine v2 \
  --temperature 1.0 --top-k 50 --top-p 0.95 \
  --presence-test exact_binomial --target-fpr 0.01 \
  --counting-mode unique_context --dataset-path "$C4_PATH" \
  --global-seed 42 --message-seed 42 \
  --generation-batch-size "$BATCH_SIZE" --device "$DEVICE" --dtype "$DTYPE" \
  "${mode_args[@]}"
