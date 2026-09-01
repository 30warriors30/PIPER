# Quality–Detection Pareto Experiments

The `paper` preset is the headline PIPER operating point: `soft_p0_m2` (`delta_presence=0`, `delta_payload=2`) with `unique_context`, an exact-binomial presence gate, blind-text reporting, and tie-to-zero conditional decoding. The `smoke`, `pilot`, and `full` grids are ablations; calibrated Z-score language below applies only when `--presence-test z_score` is selected.

This directory implements the no-attack experiment for the dual-layer watermark. It fixes prompts, natural continuations, payloads, BCH codewords, and random seeds once, generates the unwatermarked baseline once, calibrates one frozen presence threshold on a disjoint calibration split, and scans watermark operating points on the test split.

## What the experiment measures

Each operating point changes only:

- `presence_mode`: `soft` or `hard`;
- `delta_presence` for soft presence;
- `delta_payload` for payload embedding.

Everything else is shared: model, prompts, 200-token continuations, payloads, sampling seeds, key, BCH parameters, detector, and negative samples.

The pipeline reports:

- TPR at a frozen calibration threshold;
- model-generated and natural-text FPR on the test split;
- strict, hard-fill, and bounded error-erasure exact recovery;
- wrong-message and abstention rates;
- raw decided-bit BER and erasure rate;
- continuation-only corpus PPL;
- paired delta NLL, PPL ratio, and relative PPL increase;
- quality-versus-TPR and quality-versus-recovery Pareto fronts.

## Output layout

```text
outputs/experiments/<experiment_id>/
├── experiment.json
├── manifest.jsonl
├── errors/
│   └── filtered_manifest_samples.jsonl
├── shared/
│   ├── baseline.jsonl
│   ├── negative_detections.jsonl
│   ├── calibration.json
│   └── quality.jsonl
├── runs/
│   └── <point_id>/
│       ├── operating_point.json
│       ├── watermarked.jsonl
│       ├── watermarked_detections.jsonl
│       ├── detection_config.json
│       ├── quality.jsonl
│       ├── quality_summary.json
│       └── metrics.json
├── sweep_results.csv
├── sweep_results.json
├── pareto_presence.csv
├── pareto_payload.csv
├── aggregate_summary.json
└── figures/
    ├── quality_vs_tpr.png
    └── quality_vs_recovery.png
```

## 1. Dry run

```bash
python experiments/run_pareto_sweep.py \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --secret-key dual-layer-key-2026 \
  --experiment-id opt13b_pareto_smoke \
  --preset smoke \
  --dry-run
```

## 2. Four-point smoke experiment

The smoke preset uses 20 calibration and 20 test samples by default.

```bash
python experiments/run_pareto_sweep.py \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --secret-key dual-layer-key-2026 \
  --experiment-id opt13b_pareto_smoke \
  --preset smoke \
  --stage all \
  --overwrite
```

The four points are:

```text
soft_p0p5_m0p5
soft_p1_m2
soft_p2_m3
hard_m2
```

## 3. Full 49-point pilot grid

```bash
python experiments/run_pareto_sweep.py \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --secret-key dual-layer-key-2026 \
  --experiment-id opt13b_pareto_49x200 \
  --preset full \
  --calibration-samples 200 \
  --test-samples 200 \
  --exact-tokens 200 \
  --stage all \
  --overwrite
```

The full preset contains 42 soft points:

```text
delta_presence ∈ {0, 0.5, 1, 1.5, 2, 3}
delta_payload  ∈ {0, 0.5, 1, 1.5, 2, 3, 4}
```

and seven hard-presence points with the same payload values.

## 4. Run by stage

Long experiments can be split into stages:

```bash
# Build the fixed manifest.
python experiments/run_pareto_sweep.py ... --stage manifest --overwrite

# Generate shared unwatermarked baselines, detect negatives, calibrate the threshold,
# and evaluate shared quality.
python experiments/run_pareto_sweep.py ... --stage baseline --resume

# Generate/detect/score every selected operating point.
python experiments/run_pareto_sweep.py ... --stage sweep --resume

# Rebuild CSV files and figures without loading the model.
python experiments/aggregate_results.py ... --preset full
```

Use exactly the same arguments across stages because `experiment.json` defines the experimental contract.

## 5. Run selected points

```bash
python experiments/run_pareto_sweep.py \
  ... \
  --preset full \
  --stage sweep \
  --only-point soft_p1_m2 \
  --only-point hard_m2 \
  --resume
```

## Local C4 JSONL

For offline testing, each line must contain an `id` and `text` field:

```json
{"id":"0","text":"A sufficiently long C4 document ..."}
```

Run with:

```bash
python experiments/run_pareto_sweep.py ... --dataset-path /path/to/c4.jsonl
```

Documents that cannot provide prompt context plus exactly 200 natural tokens are recorded in `errors/filtered_manifest_samples.jsonl` and do not count toward the requested sample totals.

## Interpreting the main CSV

The most important columns in `sweep_results.csv` are:

- `relative_ppl_increase`: quality cost; lower is better;
- `tpr`: presence detection at the frozen threshold; higher is better;
- `model_fpr` and `natural_fpr`: test-set false-positive rates;
- `correct_attribution_rate`: paper headline CAR;
- `conditional_decoding_accuracy`: payload accuracy conditional on presence acceptance;
- `tie_zero_exact_recovery`: paper tie-to-zero recovery;
- `error_erasure_exact_recovery`: decoder ablation only;
- `wrong_message_rate`: decoded but incorrect messages;
- `abstention_rate`: samples for which error-erasure did not output a message;
- `mean_erasures`, `raw_decided_bit_ber`, `raw_erasure_rate`: pre-ECC evidence quality.

`hard` points are performance upper bounds. The paper's primary operating points should normally come from the soft Pareto front under an explicit quality budget.

## Synonym Substitution Attack Experiments

The paper-aligned attack runner evaluates 10% word-level synonym substitution and groups results by tokenizer-level effect:

- `replacement`: token-preserving substitutions
- `deletion`: token-reducing substitutions
- `insertion`: token-increasing substitutions

Real runs require NLTK WordNet data:

```bash
conda run -n BREW python -m nltk.downloader wordnet omw-1.4
```

Run the default `soft_p0_m2` attack suite:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
export NUMBA_CACHE_DIR=/tmp/numba-cache
conda run -n BREW python -m experiments.attack.run_synonym_attacks \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --point-id soft_p0_m2 \
  --attack-rate 0.10 \
  --attacks replacement deletion insertion \
  --input-modes known_boundary \
  --overwrite
```

By default the attack runner decodes payloads only for watermarked examples. Negative
examples still contribute to FPR through z-scores, but skip BCH payload decoding because
EMR is undefined for them. Add `--evaluate-all-policies` only when you need strict and
hard-fill policy breakdowns in addition to the primary error-erasure payload result.

For a CUDA-free smoke run over three samples, add `--max-samples 3` and use one attack:

```bash
conda run -n BREW python -m experiments.attack.run_synonym_attacks \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --point-id soft_p0_m2 \
  --attack-rate 0.10 \
  --attacks replacement \
  --max-samples 3 \
  --input-modes known_boundary \
  --overwrite
```

The runner reads the frozen threshold from `shared/calibration.json` and never regenerates text. Outputs are written to:

```text
outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_synonym_rate10/
```

## Re-score the completed sweep with OPT-6.7B / OPT-13B

The original Pareto sweep computes continuation PPL with the generation model. You can re-score the **already generated** watermarked and unwatermarked continuations with a larger independent OPT checkpoint without generating any text again.

The external evaluator:

- reads `shared/baseline.jsonl` and every `runs/<point_id>/watermarked.jsonl`;
- verifies that the generation and evaluator tokenizers have the same token-to-ID mapping before reusing saved token IDs;
- conditions on the full prompt but excludes prompt tokens from NLL;
- supports padded batched scoring for variable prompt lengths;
- preserves the original `quality.jsonl`, `metrics.json`, and `sweep_results.*` files;
- writes evaluator-specific results below `external_quality/<evaluator>/`;
- supports `--resume` so OPT-13B evaluation can safely continue after interruption.

### OPT-6.7B

```bash
python experiments/evaluate_external_ppl.py \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --evaluator-model /data/yanlu/BREW/models/facebook/opt-6.7b \
  --evaluator-name opt-6.7b \
  --batch-size 4 \
  --resume
```

Start with `--batch-size 1` or `2` if GPU memory is uncertain, then increase it after checking `nvidia-smi`.

### OPT-13B

```bash
python experiments/evaluate_external_ppl.py \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --evaluator-model /data/yanlu/BREW/models/facebook/opt-13b \
  --evaluator-name opt-13b \
  --batch-size 2 \
  --resume
```

For a quick one-point validation before scoring all 49 points:

```bash
python experiments/evaluate_external_ppl.py \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --evaluator-model /data/yanlu/BREW/models/facebook/opt-6.7b \
  --evaluator-name opt-6.7b \
  --only-point soft_p0_m2 \
  --batch-size 2 \
  --resume
```

Evaluator-specific outputs are written to, for example:

```text
outputs/experiments/opt13b_pareto_49x200/
└── external_quality/
    ├── opt-6_7b/
    │   ├── metadata.json
    │   ├── summary.json
    │   ├── shared/quality.jsonl
    │   ├── runs/<point_id>/quality.jsonl
    │   ├── runs/<point_id>/quality_summary.json
    │   ├── sweep_results.csv
    │   ├── sweep_results.json
    │   ├── pareto_presence.csv
    │   ├── pareto_payload.csv
    │   └── figures/
    │       ├── quality_vs_tpr.png
    │       └── quality_vs_recovery.png
    └── opt-13b/
        └── ...
```

The main new columns are:

- `external_paired_delta_nll`
- `external_ppl_ratio`
- `external_relative_ppl_increase`
- `external_evaluator_model`
- `external_evaluator_name`

For the Pareto plots, the external relative PPL increase is

\[
100\left(\exp(\Delta\mathrm{NLL}_{\mathrm{external}})-1\right)\%,
\]

where the paired delta is computed between the same sample's watermarked and unwatermarked 200-token continuation under the external evaluator.
