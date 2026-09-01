# Paraphrase Attack Experiments Design

## Goal

Add a reusable paraphrase attack experiment for the fixed-length error-erasure dual-layer watermark. The experiment evaluates the current default method, `soft_p0_m2`, on the existing Pareto output:

```text
outputs/experiments/opt13b_pareto_49x200
```

The runner should use the same generated texts, calibration threshold, detector configuration, ROC metrics, and payload metrics as the existing synonym attack runner.

## Attack Definition

The attack rewrites complete continuations with a sequence-to-sequence paraphraser. For the first local run, the paraphraser is the cached Hugging Face model:

```text
/data/yanlu/.cache/huggingface/hub/models--tuner007--pegasus_paraphrase/snapshots/0159e2949ca73657a2f1329898f51b7bb53b9ab2
```

The runner records the exact paraphraser path and generation parameters in `metadata.json`. This makes the result reproducible and allows a later T5-based paraphraser run if an exact paper model is selected.

## Data Flow

1. Load test split examples with `load_attack_examples(experiment_dir, point_id, max_samples)`.
2. Include `watermarked`, `unwatermarked`, and `natural` texts for the selected sample ids.
3. Rewrite each continuation independently.
4. Tokenize rewritten text with the experiment tokenizer.
5. Detect rewritten continuations with `known_boundary` and `blind_text` input modes.
6. Freeze detection decisions at the experiment's existing calibrated threshold.
7. Report TPR, model FPR, natural FPR, AUC, exact message recovery, BER, runtime, and paraphrase diagnostics.

## Output Layout

```text
outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_paraphrase_pegasus/
  metadata.json
  summary.json
  summary.csv
  paraphrase/
    attacked_records.jsonl
    detections.jsonl
    metrics.json
    roc_points.csv
  figures/
    roc_paraphrase_attack.png
    roc_paraphrase_attack.pdf
```

## Diagnostics

Each attacked record stores the original text, rewritten text, original and attacked token ids, token counts, token length delta, compression ratio, paraphrase seconds, and generation parameters. Payload decoding is attempted only for watermarked texts, matching the existing attack runner behavior.

## Validation

The implementation must add focused tests for:

- default output directory naming;
- paraphrase result diagnostics;
- detection record fields and payload decode policy;
- summary/evaluation behavior with frozen thresholds;
- module `--help` execution.

The real run should first execute a `--max-samples 2` smoke test before the full 200-sample run.
