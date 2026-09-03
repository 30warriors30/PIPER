#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/yanlu/BREW/models/facebook/opt-1.3b}"
C4_PATH="${C4_PATH:-/data/yanlu/BREW/dataset/c4/processed_c4.json}"
SECRET_KEY="${SECRET_KEY:-dual-layer-key-2026}"
EXPERIMENT_ID="${EXPERIMENT_ID:-piper_main_t200_b8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/experiments}"

PRESET="${PRESET:-paper}"
STAGE="${STAGE:-all}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-1000}"
TEST_SAMPLES="${TEST_SAMPLES:-1000}"
EXACT_TOKENS="${EXACT_TOKENS:-200}"
TARGET_FPR="${TARGET_FPR:-0.01}"
PRESENCE_TEST="${PRESENCE_TEST:-exact_binomial}"
COUNTING_MODE="${COUNTING_MODE:-unique_context}"
BATCH_SIZE="${BATCH_SIZE:-16}"

cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  if [[ ! -e "$MODEL_PATH" ]]; then
    echo "MODEL_PATH does not exist: $MODEL_PATH" >&2
    echo "Set MODEL_PATH=/path/to/local/model before running this script." >&2
    exit 1
  fi
  if [[ ! -f "$C4_PATH" ]]; then
    echo "C4_PATH does not exist: $C4_PATH" >&2
    echo "Set C4_PATH=/path/to/processed_c4.json before running this script." >&2
    exit 1
  fi
fi

mode_args=(--resume)
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  mode_args=(--overwrite)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  mode_args=(--dry-run)
fi

python -m experiments.run_pareto_sweep \
  --model-path "$MODEL_PATH" \
  --secret-key "$SECRET_KEY" \
  --experiment-id "$EXPERIMENT_ID" \
  --output-root "$OUTPUT_ROOT" \
  --preset "$PRESET" \
  --stage "$STAGE" \
  --calibration-samples "$CALIBRATION_SAMPLES" \
  --test-samples "$TEST_SAMPLES" \
  --exact-tokens "$EXACT_TOKENS" \
  --generation-batch-size "$BATCH_SIZE" \
  --target-fpr "$TARGET_FPR" \
  --presence-test "$PRESENCE_TEST" \
  --counting-mode "$COUNTING_MODE" \
  --dataset-path "$C4_PATH" \
  "${mode_args[@]}"
