# Reproducing the OPT-1.3B/C4 run

This page documents the `PIPER_selfhash` source snapshot's main OPT-1.3B/C4 configuration. The repository contains the experiment pipeline, but not the checkpoint, C4 snapshot, generated JSONL files, or separate baseline code.

## What this run computes

The command below creates a 6,000-record manifest: 1,000 calibration rows and 5,000 test rows. It generates watermarked continuations for the first 1,000 test rows. The model-generated and C4 natural null sets use all 5,000 test rows. The exact-binomial PIPER gate does not use the calibration rows to set its threshold.

The operating point is `soft_p0_m2`. It uses a 300-token continuation, an 8-bit BCH(23, 8, 3) payload, a payload-child bias of 2, no separate parent bias, and alpha 0.01. Detection counts unique contexts in blind-text mode and decodes with tie-to-zero only after presence acceptance.

For this selfhash implementation, the candidate's partition seed uses the last `h-1` history tokens followed by the candidate token. Encoded-bit allocation uses the last `h` history tokens. `unique_context` suppresses repeated candidate-conditioned contexts in the detector. The generator biases only its raw-logit top-50 candidates; top-k 50 and top-p 0.95 sampling occur after the watermark adjustment. Detection needs the observed token IDs and key, not model logits or the generation-time candidate list.

## Inputs

Follow [README.md](README.md) to create the Conda environment and a processed C4 RealNewsLike JSONL file with at least 6,000 valid records. Use the same OPT-1.3B tokenizer for preprocessing, generation, and detection. Set these paths before running:

```bash
export MODEL_PATH=/path/to/opt-1.3b
export C4_PATH=/path/to/c4_realnewslike_validation_t300_6000.jsonl
export SECRET_KEY=dual-layer-key-2026
```

`dual-layer-key-2026` is the example research key recorded in the server run notes. A different key changes the generated text and measured values. Do not use a research key from this repository as a deployment secret.

## Main command

```bash
python -m experiments.run_pareto_sweep \
  --model-path "$MODEL_PATH" --secret-key "$SECRET_KEY" \
  --experiment-id piper_main_t300_b8_selfhash_top50 \
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
  --generation-batch-size 16 --device cuda --dtype float16
```

The equivalent wrapper is `bash scripts/run_pareto_sweep.sh`. Add `--dry-run` to the direct command, or set `DRY_RUN=1` for the wrapper, to inspect the configuration without writing files. Add `--resume` after an interrupted run. `--overwrite` replaces the existing experiment; use it deliberately.

## Results

The run writes `outputs/experiments/piper_main_t300_b8_selfhash_top50/`. Check:

- `experiment.json` for model, dataset, seed, tokenizer-related, and watermark settings.
- `manifest.jsonl` for selected record IDs and generation seeds.
- `runs/soft_p0_m2/metrics.json` for presence TPR, correct attribution rate (CAR), conditional decoding accuracy (CDA), and quality.
- `shared/negative_detections.jsonl` for model and natural null detections.
- `sweep_results.csv` and `aggregate_summary.json` for compact summaries.

CAR requires both presence acceptance and the correct decoded payload. CDA is payload accuracy conditional on presence acceptance. Report model-null and natural-null false-positive rates separately. These empirical rates for a fixed key are distinct from the random-key null-model guarantee described in the manuscript.

For independent continuation perplexity under OPT-6.7B, run after the main experiment:

```bash
python -m experiments.evaluate_external_ppl \
  --experiment-dir outputs/experiments/piper_main_t300_b8_selfhash_top50 \
  --evaluator-model /path/to/opt-6.7b --evaluator-name opt-6.7b \
  --only-point soft_p0_m2 --batch-size 8 \
  --device cuda --dtype float16
```

This command scores saved continuations; it does not regenerate them. Its evaluator checkpoint should be pinned to an immutable revision or hash when reporting exact numbers.

## Scope of this source release

This directory also contains tools for sweeps, attacks, and diagnostics. Their presence does not mean every table or comparison in the paper can be reproduced from this source release alone. Baseline repositories and experiment outputs maintained outside `PIPER_selfhash` are not included.
