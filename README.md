# Dual-Layer Multi-bit LLM Watermark

## Paper-aligned PIPER defaults

The main experiment path is now fixed to child-only embedding and a decoder-independent presence gate:

```text
presence_mode=soft, delta_presence=0, delta_payload=2
presence_test=exact_binomial, counting_mode=unique_context
primary_decoding_policy=tie_zero, headline_input_mode=blind_text
```

The exact-binomial gate runs before ECC. Z-score thresholds, `all_tokens`, parent bias, hard masking, and the legacy decoders remain available only for ablations. See [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md) for the complete experiment-purpose/command matrix.

A direct-Python research implementation of the dual-layer multi-bit watermark:

- first layer: zero-bit watermark presence detection;
- second layer: BCH-protected multi-bit payload recovery;
- exact seeded vocabulary permutation;
- paired watermarked/unwatermarked generation with exact continuation length;
- C4 natural-continuation negative samples with the same exact token length;
- known-boundary and blind-text detection;
- paper tie-to-zero BCH decoding alongside strict, hard-fill, and error-erasure ablations;
- TPR/FPR, ROC/AUC, codeword BER, message BER, and exact-message recovery.

The project has no YAML configuration, no `src/` layout, no editable installation, and no console-script command. Run the four root Python files directly.

## Project layout

```text
run_generation.py
run_detection.py
evaluate_results.py
calibrate_threshold.py
watermark/
evaluation/
utils/
tests/
scripts/
environment.yml
requirements.txt
```

## 1. Build the Blackwell environment

This environment targets an NVIDIA RTX PRO 6000 Blackwell GPU with compute capability 12.0.

```bash
cd /data/yanlu/BREW/dual_layer_watermark_refactor
bash scripts/create_environment.sh
conda activate dual-watermark-blackwell
```

Manual equivalent:

```bash
conda env create -f environment.yml
conda activate dual-watermark-blackwell
export PYTHONNOUSERSITE=1
python -m pip check
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("architectures:", torch.cuda.get_arch_list())
PY
```

For a Blackwell `sm_120` GPU, `sm_120` must appear in `torch.cuda.get_arch_list()`.

## 2. Run tests

```bash
pytest -q
```

Real local-model integration:

```bash
LOCAL_OPT_MODEL=/data/yanlu/BREW/models/facebook/opt-1.3b \
pytest -q tests/test_integration_local_model.py -s
```

## 3. Dry-run generation arguments

```bash
python run_generation.py \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --secret-key dual-layer-key-2026 \
  --run-id c4_opt13b_piper_200 \
  --dry-run
```

Dry-run parses and validates parameters and prints the experiment summary. It does not load the full model or create outputs.

## 4. Three-sample C4 smoke test

```bash
bash scripts/smoke_test.sh
```

Or run the four steps manually:

```bash
python run_generation.py \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --secret-key smoke-test-key \
  --dataset c4 \
  --max-samples 3 \
  --max-new-tokens 32 \
  --run-id smoke_c4_opt13b

python run_detection.py \
  --input-type samples \
  --run-dir outputs/smoke_c4_opt13b

python evaluate_results.py \
  --run-dir outputs/smoke_c4_opt13b

cat outputs/smoke_c4_opt13b/summary.txt
```

## 5. Generate 200 C4 samples with local OPT-1.3B

```bash
bash scripts/run_c4_opt13b_200.sh
```

The explicit command is:

```bash
python run_generation.py \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --device cuda \
  --dtype float16 \
  --local-files-only \
  --dataset c4 \
  --dataset-name allenai/c4 \
  --dataset-config realnewslike \
  --dataset-split validation \
  --streaming \
  --max-samples 200 \
  --max-new-tokens 200 \
  --secret-key dual-layer-key-2026 \
  --context-width 4 \
  --presence-mode soft \
  --delta-presence 0.0 \
  --delta-payload 2.0 \
  --prf-mode paper_shared \
  --partition-mode exact_permutation \
  --allocation-mode hash_mod \
  --ecc-n 23 \
  --ecc-k 8 \
  --ecc-t 3 \
  --temperature 1.0 \
  --top-p 0.95 \
  --global-seed 42 \
  --message-seed 42 \
  --run-id c4_opt13b_piper_200
```

`--max-new-tokens` is treated as an exact continuation length. The default is 200. Both model continuations receive equal `min_new_tokens` and `max_new_tokens`; C4 documents that cannot supply an exact 200-token natural continuation plus watermark context are filtered, and streaming continues until 200 valid triples are completed.

For each C4 sample the generator saves:

- exactly 200 watermarked token IDs;
- exactly 200 paired unwatermarked token IDs;
- exactly 200 original C4 natural-continuation token IDs;
- one deterministic independent 8-bit message;
- the BCH(23, 8, t=3) codeword;
- the watermarked continuation;
- the paired unwatermarked continuation generated from the same sample seed;
- the C4 natural continuation;
- prompt and continuation token IDs;
- generation and ECC timing.

## 6. Detect generated samples

```bash
python run_detection.py \
  --input-type samples \
  --run-dir outputs/c4_opt13b_piper_200 \
  --presence-test exact_binomial \
  --counting-mode unique_context \
  --primary-decoding-policy tie_zero \
  --max-erasure-assignments 64
```

The detector automatically reads generation-time watermark parameters from `experiment.json`. It does not use the ground-truth message to decode. It evaluates all three text classes in both modes:

- `known_boundary`: prompt provides context and only continuation tokens are scored;
- `blind_text`: the first `context_width` tokens are warm-up and the remainder is scored.

## 7. Evaluate

```bash
python evaluate_results.py \
  --run-dir outputs/c4_opt13b_piper_200 \
  --input-mode blind_text
```

Outputs:

```text
metrics.json
metrics.csv
roc_points.csv
summary.txt
```

Important metrics:

- `presence_by_input_mode.blind_text.tpr`
- `presence_by_input_mode.blind_text.model_fpr`
- `presence_by_input_mode.blind_text.natural_fpr`
- `payload_by_input_mode.blind_text.correct_attribution_rate`
- `payload_by_input_mode.blind_text.conditional_decoding_accuracy`
- `payload_by_input_mode.blind_text.tie_zero_exact_message_recovery`
- strict, hard-fill, and error-erasure decode success rates;
- policy-specific exact message recovery and conditional message BER;
- error-erasure status counts (`decoded`, `decode_failed`, `ambiguous`, `too_many_erasures`).

Known-boundary and blind-text metrics are reported separately.

### Conditional tie-to-zero BCH decoding

The default payload policy is `tie_zero`: a code bit is 1 only when its child-1 vote count is strictly larger; ties and uncovered positions are 0. ECC is invoked only after the exact-binomial presence gate accepts.

When all-policy evaluation is requested, each accepted detection record also contains `strict`, `hard_fill`, and `error_erasure` results for direct comparison.

## 8. Calibrate a Z-score ablation threshold

The paper main exact-binomial test does not use this calibration step. Use it only for the empirical Z-score ablation.

At a 1% target FPR, at least 100 independent negative calibration samples are required.

```bash
python calibrate_threshold.py \
  --run-dir outputs/c4_opt13b_piper_200 \
  --negative-source unwatermarked \
  --input-mode known_boundary \
  --target-fpr 0.01
```

Then apply the calibrated threshold to a separate detection file:

```bash
python run_detection.py \
  --input-type samples \
  --run-dir outputs/c4_opt13b_piper_200 \
  --presence-test z_score \
  --threshold-mode calibrated \
  --calibration-file outputs/c4_opt13b_hard_200/calibration.json \
  --output-file detections_calibrated.jsonl
```

## 9. Detect external text

Blind JSONL text detection:

```bash
python run_detection.py \
  --input-type text \
  --input-file data/external_texts.jsonl \
  --text-field text \
  --output-dir outputs/external_detection \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --secret-key dual-layer-key-2026 \
  --context-width 4 \
  --ecc-n 23 \
  --ecc-k 8 \
  --ecc-t 3
```

Known-boundary JSONL detection:

```bash
python run_detection.py \
  --input-type text \
  --input-file data/prompt_continuation.jsonl \
  --prompt-field prompt \
  --continuation-field continuation \
  --output-dir outputs/external_known \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --secret-key dual-layer-key-2026
```

## 10. Resume and overwrite

Generation:

```bash
python run_generation.py ... --run-id c4_opt13b_hard_200 --resume
```

Detection:

```bash
python run_detection.py --input-type samples --run-dir outputs/c4_opt13b_hard_200 --resume
```

`--resume` skips completed records and retries errors. Critical generation parameters must match the existing `experiment.json`. Use `--overwrite` only when you intentionally want to delete the existing run output.

## 11. Output layout

```text
outputs/<run_id>/
├── experiment.json
├── environment.json
├── samples.jsonl
├── detections.jsonl
├── detection_config.json
├── metrics.json
├── metrics.csv
├── roc_points.csv
├── summary.txt
├── calibration.json
├── logs/
└── errors/
```

## Current first-release limits

- only `exact_permutation` vocabulary partitioning;
- only `hash_mod` bit allocation;
- no insertion/deletion alignment search;
- no paraphrase attacks;
- no PPL/BERTScore evaluation;
- `exact_permutation` reconstructs a full CPU vocabulary permutation per token and is computationally expensive.
