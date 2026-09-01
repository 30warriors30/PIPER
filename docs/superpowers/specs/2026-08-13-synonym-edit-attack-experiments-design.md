# Synonym Edit Attack Experiments Design

## Goal

Implement attack experiments for the dual-layer fixed-length error-erasure watermark that match the attack flow used in "Block-wise Codeword Embedding for Reliable Multi-bit Text Watermarking" for synonym substitution attacks. The main experiment evaluates replacement, deletion-like, and insertion-like attacks under the existing Pareto run:

- Experiment directory: `outputs/experiments/opt13b_pareto_49x200`
- Operating point: `soft_p0_m2`
- Text length: `T=200`
- Message length and ECC: `b=8`, BCH `(23,8,3)`
- Dataset/model context: C4, OPT-1.3B generation outputs already present
- Attack rate: `10%` by default
- Detection threshold: the frozen calibrated threshold in `shared/calibration.json`

The output should support paper-facing plots and tables: TPR/FPR ROC curves plus fixed-threshold TPR, FPR, EMR, BER, AUC, and decode diagnostics for each attack class.

## Paper Alignment

The paper's Section 5.3 uses word-level synonym substitution. Although the text edit is always a synonym replacement at the word level, the tokenizer can produce three token-level effects:

- `replacement`: token-preserving synonym substitutions
- `deletion`: token-reducing, deletion-like synonym substitutions
- `insertion`: token-increasing, insertion-like synonym substitutions

The primary experiment should use this synonym-substitution definition instead of random token replacement/deletion/insertion. Random token edits may be added later as a separate ablation, but they are not the main paper-aligned result.

The paper's default attack plot uses 10% synonym substitutions, OPT-1.3B, and `T=200`. This design keeps the current detector unchanged and evaluates attacked continuations using the existing `known_boundary` and `blind_text` modes. It does not implement BREW's paper-specific window-shift detector because the current project has a different detector architecture and the request is to complete the corresponding attack experiment for the existing method.

## Inputs

The attack runner reads the completed Pareto experiment outputs:

- `runs/<point_id>/watermarked.jsonl`
- `shared/baseline.jsonl`
- `shared/calibration.json`
- `experiment.json`
- `runs/<point_id>/operating_point.json`

The runner uses only existing generated continuations. It does not regenerate text and does not require CUDA for the attack or detection pass. It loads the tokenizer locally from the model path in `experiment.json` so that synonym candidates can be categorized by token-length change.

## Attack Construction

The main attack module should operate at the text level:

1. Tokenize text into editable word spans and protected non-word spans.
2. For each editable word, generate synonym candidates.
3. Retokenize the original word and candidate synonym with the experiment tokenizer.
4. Keep candidates whose token-length delta matches the requested attack class:
   - `replacement`: delta `0`
   - `deletion`: delta `<0`
   - `insertion`: delta `>0`
5. Deterministically select up to `attack_rate * editable_word_count` replacements using a per-sample seed.
6. Reconstruct attacked text.
7. Tokenize the attacked continuation into token IDs for detector input.

The default synonym source should be WordNet through NLTK if it is installed and data is available. If WordNet data is unavailable, the runner should fail with a clear message explaining how to install/download it, rather than silently switching to random token edits. This keeps the main results aligned with the paper definition.

If a sample has too few valid synonym candidates for a requested class, the sample is still included with the achieved replacement count recorded. Summary metrics must report achieved attack rate and coverage so low-candidate cases are visible.

## Detection And Metrics

For each attack class, the runner attacks and detects:

- Watermarked test continuations from `runs/<point_id>/watermarked.jsonl`
- Unwatermarked test continuations from `shared/baseline.jsonl`
- Natural test continuations from `shared/baseline.jsonl`

Each attacked continuation is detected using:

- `known_boundary`: `detector.detect_continuation(prompt_ids, attacked_token_ids)`
- `blind_text`: `detector.detect_token_ids(attacked_token_ids)`, matching the project's existing continuation-only blind detection convention

Metrics are computed with the existing `evaluation.metrics.evaluate_records`, using the frozen calibration threshold. The headline mode is `known_boundary`.

The attack metrics should include:

- Presence: TPR, model FPR, natural FPR, AUC, ROC points
- Payload: exact message recovery, message BER, codeword BER
- Policy diagnostics: strict, hard-fill, error-erasure decode rates
- Attack diagnostics: achieved edit count, achieved attack rate, token-length delta, candidate coverage

## Output Layout

The default output root is:

```text
outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_synonym_rate10/
```

Within it:

```text
metadata.json
summary.json
summary.csv
replacement/
  attacked_records.jsonl
  detections.jsonl
  metrics.json
  roc_points.csv
deletion/
  attacked_records.jsonl
  detections.jsonl
  metrics.json
  roc_points.csv
insertion/
  attacked_records.jsonl
  detections.jsonl
  metrics.json
  roc_points.csv
figures/
  roc_synonym_attacks.png
  roc_synonym_attacks.pdf
```

`attacked_records.jsonl` stores the attacked text, attacked token IDs, original token count, attacked token count, edit count, target attack rate, achieved attack rate, and token-length delta. `detections.jsonl` follows the existing detection record schema with extra attack metadata fields.

## CLI

Add a standalone script under `experiments/attack/`:

```bash
python -m experiments.attack.run_synonym_attacks \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --point-id soft_p0_m2 \
  --attack-rate 0.10 \
  --attacks replacement deletion insertion \
  --overwrite
```

Useful options:

- `--max-samples` for smoke tests
- `--seed` for deterministic attack selection
- `--input-mode` or `--headline-input-mode`, default `known_boundary`
- `--resume` and `--overwrite`, mutually exclusive
- `--output-dir` to override the default output root
- `--require-full-rate` for strict experiments that should error if a sample cannot reach the requested attack rate

## Testing

Unit tests should cover:

- Word-span reconstruction preserves untouched text.
- Synonym candidate filtering correctly maps token-length deltas to `replacement`, `deletion`, and `insertion`.
- Attack selection is deterministic for the same sample seed.
- Metrics assembly uses the frozen threshold and passes records through `evaluate_records`.
- CLI smoke run works with a fake tokenizer and monkeypatched synonym provider without loading a real model.

Verification commands:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
pytest tests/experiments/test_synonym_attacks.py
python -m experiments.attack.run_synonym_attacks --help
```

For a real run, use:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
export NUMBA_CACHE_DIR=/tmp/numba-cache
python -m experiments.attack.run_synonym_attacks \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --point-id soft_p0_m2 \
  --attack-rate 0.10 \
  --attacks replacement deletion insertion \
  --overwrite
```

## Non-Goals

- Do not regenerate watermarked or unwatermarked text.
- Do not change the watermark embedding algorithm.
- Do not add the paper's window-shift detector in this task.
- Do not use random token edits as the main attack result.
- Do not require CUDA for the attack detection pass.
