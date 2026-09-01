# MPAC T200 B8 Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the MPAC baseline from `bangawayoo/mb-lm-watermarking` on the same C4/OPT-1.3B prompts and messages as the existing BREW `soft_p0_m2` experiment with `T=200`, `b=8`, and `delta=2`. The main comparison uses original MPAC with raw 8-bit messages and no ECC; a supplementary ablation runs MPAC with BCH(23,8,3) wrapping the same 8-bit payload.

**Architecture:** Add a local experiment adapter that imports the MPAC implementation from an external checkout, reuses `opt13b_pareto_49x200/manifest.jsonl` and `shared/baseline.jsonl`, generates only MPAC positives for the test split, scores MPAC positives and negatives, calibrates the MPAC threshold from calibration negatives, and writes comparable JSON/CSV summaries. Keep reusable parsing and aggregation logic testable without loading models.

**Tech Stack:** Python, PyTorch, Transformers, the existing project JSONL helpers, pytest, and the cloned MPAC repository under `/tmp/mb-lm-watermarking`.

## Global Constraints

- Existing BREW data source: `outputs/experiments/opt13b_pareto_49x200`.
- Existing BREW point: `soft_p0_m2`, equivalent to `T=200`, `b=8`, `delta_payload=2`.
- MPAC baseline parameters: `message_length=8`, `base=4`, `gamma=0.25`, `delta=2.0`, `seeding_scheme=lefthash`, `mpac_ecc=none`.
- MPAC+BCH ablation parameters: original `message_length=8`, embedded BCH codeword length `23`, BCH correction radius `t=3`, and the same `base=4`, `gamma=0.25`, `delta=2.0`, `seeding_scheme=lefthash`.
- Use the manifest's `prompt_token_ids`, `message_bits`, and `generation_seed` for MPAC positives.
- Use existing shared unwatermarked and natural completions for MPAC negative scoring.
- Do not regenerate BREW results.
- Calibrate detector threshold on calibration negatives and evaluate on test positives/negatives.

---

### Task 1: Testable MPAC Summary Utilities

**Files:**
- Create: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_mpac_comparison.py`
- Create: `dual_layer_watermark_fixed_length_error_erasure/experiments/mpac_comparison.py`

**Interfaces:**
- Produces: `load_manifest_rows(path: Path, split: str, limit: int | None) -> list[dict[str, Any]]`
- Produces: `empirical_threshold(scores: Sequence[float], target_fpr: float) -> float`
- Produces: `summarize_records(records: Iterable[dict[str, Any]], threshold: float, target_fpr: float) -> dict[str, Any]`

- [x] **Step 1: Write failing tests**

```python
def test_load_manifest_rows_filters_split_and_limit(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"sample_id":"0","split":"calibration"}\n'
        '{"sample_id":"1","split":"test"}\n'
        '{"sample_id":"2","split":"test"}\n'
    )
    rows = load_manifest_rows(manifest, split="test", limit=1)
    assert [row["sample_id"] for row in rows] == ["1"]
```

```python
def test_empirical_threshold_respects_target_fpr_with_ties():
    threshold = empirical_threshold([0.2, 1.0, 1.0, 3.0], target_fpr=0.25)
    assert threshold == 3.0
```

```python
def test_summarize_records_reports_fpr_tpr_and_payload_metrics():
    summary = summarize_records(
        [
            {"text_class": "watermarked", "z_score": 3.0, "bit_match": True, "bit_acc": 1.0},
            {"text_class": "watermarked", "z_score": 1.0, "bit_match": True, "bit_acc": 1.0},
            {"text_class": "watermarked", "z_score": 3.0, "bit_match": False, "bit_acc": 0.75},
            {"text_class": "unwatermarked", "z_score": 2.5},
            {"text_class": "natural", "z_score": 0.0},
        ],
        threshold=2.0,
        target_fpr=0.01,
    )
    assert summary["tpr"] == 2 / 3
    assert summary["combined_fpr"] == 0.5
    assert summary["exact_message_recovery"] == 2 / 3
    assert summary["end_to_end_exact_recovery"] == 1 / 3
    assert summary["mean_bit_accuracy"] == pytest.approx(11 / 12)
```

- [x] **Step 2: Run tests to verify failure**

Run: `/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_mpac_comparison.py -q`

Expected: fails because `experiments.mpac_comparison` does not exist.

- [x] **Step 3: Implement utilities**

Implement JSONL loading, target-FPR threshold selection with tie handling, and metric aggregation.

- [x] **Step 4: Run tests to verify pass**

Run: `/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_mpac_comparison.py -q`

Expected: all tests pass.

### Task 2: MPAC Generation And Scoring CLI

**Files:**
- Modify: `dual_layer_watermark_fixed_length_error_erasure/experiments/mpac_comparison.py`

**Interfaces:**
- Produces CLI: `python -m experiments.mpac_comparison --experiment-dir ... --mb-repo ... --output-dir ...`
- Writes: `records.jsonl`, `metrics.json`, `metadata.json`

- [x] **Step 1: Add parser tests for defaults**

Assert defaults match `T=200`, `message_length=8`, `base=4`, `gamma=0.25`, `delta=2.0`, and `seeding_scheme=lefthash`.

- [x] **Step 2: Implement CLI defaults and output metadata**

Add `build_parser()` and metadata writing before model loading.

- [x] **Step 3: Implement MPAC class loading**

Add `import_mpac(mb_repo: Path)` that inserts `watermark_reliability_release` into `sys.path` and returns `WatermarkLogitsProcessor` and `WatermarkDetector`.

- [x] **Step 4: Implement GPU-dependent run path**

Load OPT-1.3B locally, generate test watermarked completions with `min_new_tokens=max_new_tokens=200`, score watermarked/unwatermarked/natural records, calibrate on calibration negatives, and summarize test records.

- [x] **Step 5: Verify parser and utility tests**

Run: `/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_mpac_comparison.py -q`

Expected: all tests pass.

### Task 2b: MPAC BCH(23,8,3) Ablation Support

**Files:**
- Modify: `dual_layer_watermark_fixed_length_error_erasure/experiments/mpac_comparison.py`
- Modify: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_mpac_comparison.py`
- Modify: `dual_layer_watermark_fixed_length_error_erasure/watermark/ecc.py`

**Interfaces:**
- Adds CLI option: `--mpac-ecc none|bch23`
- Keeps default behavior as original MPAC: `--mpac-ecc none`
- In `bch23` mode, embeds a BCH(23,8,3) codeword while reporting recovery on the original 8-bit message.

- [x] **Step 1: Add failing tests for BCH mode**

Assert that `--mpac-ecc bch23` embeds a 23-bit BCH codeword and decodes predicted MPAC digits before computing exact message recovery.

- [x] **Step 2: Implement BCH embedding and recovery metrics**

Add `embedded_message_length()`, `embedded_payload_bits()`, `mpac_digits_from_bits()`, and `evaluate_payload_prediction()`. Use `message_recovered` and `original_bit_acc` for headline payload metrics, and add `mean_embedded_bit_accuracy` for the encoded codeword.

- [x] **Step 3: Make optional BCH backend robust**

Allow `BCHCodec` to fall back to its pure-Python implementation when the optional `galois` backend is installed but unusable in the current environment.

- [x] **Step 4: Verify tests**

Run: `/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_mpac_comparison.py -q`

Expected: all tests pass.

### Task 3: Run The Comparison

**Files:**
- Read: `dual_layer_watermark_fixed_length_error_erasure/outputs/experiments/opt13b_pareto_49x200/manifest.jsonl`
- Read: `dual_layer_watermark_fixed_length_error_erasure/outputs/experiments/opt13b_pareto_49x200/shared/baseline.jsonl`
- Read: `dual_layer_watermark_fixed_length_error_erasure/outputs/experiments/opt13b_pareto_49x200/runs/soft_p0_m2/metrics.json`
- Write: `dual_layer_watermark_fixed_length_error_erasure/outputs/experiments/opt13b_pareto_49x200/comparisons/mpac_t200_b8_m2/`

- [ ] **Step 1: Run when CUDA is available**

```bash
cd /data/yanlu/BREW/dual_layer_watermark_fixed_length_error_erasure
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m experiments.mpac_comparison \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --mb-repo /tmp/mb-lm-watermarking \
  --output-dir outputs/experiments/opt13b_pareto_49x200/comparisons/mpac_t200_b8_m2 \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --device cuda
```

- [ ] **Step 2: Run BCH(23,8,3) ablation when CUDA is available**

```bash
cd /data/yanlu/BREW/dual_layer_watermark_fixed_length_error_erasure
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m experiments.mpac_comparison \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --mb-repo /tmp/mb-lm-watermarking \
  --output-dir outputs/experiments/opt13b_pareto_49x200/comparisons/mpac_t200_b8_m2_bch23 \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --mpac-ecc bch23 \
  --device cuda
```

- [ ] **Step 3: Compare summary metrics**

Compare original MPAC `metrics.json`, MPAC+BCH `metrics.json`, and BREW `runs/soft_p0_m2/metrics.json` for calibrated FPR, TPR, exact recovery, bit accuracy, encoded-codeword bit accuracy, and end-to-end exact recovery.
