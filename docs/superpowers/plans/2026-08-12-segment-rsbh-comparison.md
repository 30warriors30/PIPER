# Segment RSBH Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible comparison against randomizedtree/segment-watermark using the existing BREW/MPAC T=200, b=8 experiment data.

**Architecture:** Keep the external repository read-only and add a local adapter under `dual_layer_watermark_fixed_length_error_erasure/experiments/`. The adapter imports Segment RSBH classes from a checkout path, reuses the existing manifest and shared baseline negatives, calibrates detection at 1% FPR, and writes comparison records/metadata/metrics in the same shape as MPAC comparisons.

**Tech Stack:** Python, pytest, PyTorch, Transformers, segment-watermark `RSBHGenerator`/`RSBHDecoder`, existing BREW quality helpers.

## Global Constraints

- Source baseline repository: `https://github.com/randomizedtree/segment-watermark`.
- Primary comparison: `T=200`, `b=8`, `gamma=0.5`, `delta=2.0`, `target_fpr=0.01`.
- 8bit Segment-RSBH adapter RS scheme: `(n, k, m) = (3, 1, 8)`.
- Do not modify the external repository checkout.
- Use the existing experiment directory `outputs/experiments/opt13b_pareto_49x200`.
- Use calibration negatives only to choose the threshold; report metrics on test records.
- Avoid touching unrelated dirty worktree files.

---

### Task 1: Segment Metric Helpers

**Files:**
- Create: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_segment_rsbh_comparison.py`
- Create: `dual_layer_watermark_fixed_length_error_erasure/experiments/segment_rsbh_comparison.py`

**Interfaces:**
- Produces: `segment_payload_to_bits(payload: int, bit_width: int) -> str`
- Produces: `evaluate_segment_payload_prediction(pred_payload: int, expected_bits: str) -> dict[str, object]`
- Produces: `summarize_records(records, threshold: float, target_fpr: float) -> dict[str, object]`

- [ ] **Step 1: Write the failing tests**

```python
def test_segment_payload_prediction_reports_exact_and_bit_accuracy():
    result = evaluate_segment_payload_prediction(pred_payload=0b00110001, expected_bits="00110001")
    assert result["decoded_message_bits"] == "00110001"
    assert result["message_recovered"] is True
    assert result["original_bit_acc"] == 1.0


def test_segment_summary_uses_z_score_threshold_and_payload_metrics():
    summary = summarize_records(
        [
            {"split": "test", "text_class": "watermarked", "z_score": 3.0, "message_recovered": True, "original_bit_acc": 1.0},
            {"split": "test", "text_class": "watermarked", "z_score": 1.0, "message_recovered": False, "original_bit_acc": 0.5},
            {"split": "test", "text_class": "unwatermarked", "z_score": 2.0},
            {"split": "test", "text_class": "natural", "z_score": 0.5},
        ],
        threshold=2.5,
        target_fpr=0.01,
    )
    assert summary["tpr"] == 0.5
    assert summary["model_fpr"] == 0.0
    assert summary["natural_fpr"] == 0.0
    assert summary["exact_message_recovery"] == 0.5
    assert summary["end_to_end_exact_recovery"] == 0.5
    assert summary["mean_bit_accuracy"] == 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_segment_rsbh_comparison.py -q`
Expected: FAIL because `experiments.segment_rsbh_comparison` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create the module with pure helper functions only.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_segment_rsbh_comparison.py -q`
Expected: PASS.

### Task 2: External Repo Import And Mapping

**Files:**
- Modify: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_segment_rsbh_comparison.py`
- Modify: `dual_layer_watermark_fixed_length_error_erasure/experiments/segment_rsbh_comparison.py`

**Interfaces:**
- Produces: `segment_rs_scheme(message_length: int) -> tuple[int, int, int]`
- Produces: `build_frequency_mapping(rows, vocab_size: int, gf_segments_num: int) -> dict[int, int]`
- Produces: `import_segment_rsbh(segment_repo: Path) -> tuple[type, type, Any]`

- [ ] **Step 1: Write the failing tests**

```python
def test_segment_rs_scheme_uses_8bit_adapter():
    assert segment_rs_scheme(8) == (3, 1, 8)


def test_frequency_mapping_assigns_every_vocab_id_to_a_segment():
    rows = [{"prompt_token_ids": [0, 1, 1, 2]}, {"natural_token_ids": [2, 3]}]
    mapping = build_frequency_mapping(rows, vocab_size=6, gf_segments_num=3)
    assert set(mapping) == set(range(6))
    assert set(mapping.values()) <= {0, 1, 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_segment_rsbh_comparison.py -q`
Expected: FAIL for missing functions.

- [ ] **Step 3: Write minimal implementation**

Add deterministic round-robin frequency-balanced mapping and guarded external import.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_segment_rsbh_comparison.py -q`
Expected: PASS.

### Task 3: Comparison Runner

**Files:**
- Modify: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_segment_rsbh_comparison.py`
- Modify: `dual_layer_watermark_fixed_length_error_erasure/experiments/segment_rsbh_comparison.py`
- Create: `experiments/segment_rsbh_comparison.py`

**Interfaces:**
- Produces: `build_parser() -> argparse.ArgumentParser`
- Produces: `run(args: argparse.Namespace) -> dict[str, object]`
- Produces root wrapper module importable as `python -m experiments.segment_rsbh_comparison`.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_parser_defaults_match_confirmed_segment_baseline():
    args = build_parser().parse_args([
        "--experiment-dir", "outputs/experiments/opt13b_pareto_49x200",
        "--segment-repo", "/tmp/segment-watermark",
        "--output-dir", "outputs/experiments/opt13b_pareto_49x200/comparisons/segment_rsbh_t200_b8_m2",
        "--model-path", "/data/yanlu/BREW/models/facebook/opt-1.3b",
    ])
    assert args.exact_tokens == 200
    assert args.message_length == 8
    assert args.gamma == 0.5
    assert args.delta == 2.0
    assert args.target_fpr == 0.01
    assert segment_rs_scheme(args.message_length) == (3, 1, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_segment_rsbh_comparison.py -q`
Expected: FAIL for missing parser.

- [ ] **Step 3: Write minimal implementation**

Implement runtime validation, model/tokenizer loading, negative scoring, positive generation, threshold calibration, metadata/records/metrics writing.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_segment_rsbh_comparison.py -q`
Expected: PASS.

### Task 4: Segment PPL Evaluator

**Files:**
- Create: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_segment_ppl.py`
- Create: `dual_layer_watermark_fixed_length_error_erasure/experiments/evaluate_segment_ppl.py`
- Create: `experiments/evaluate_segment_ppl.py`

**Interfaces:**
- Produces: `load_segment_quality_examples(comparison_dir: Path, include_shared: bool = True) -> list[dict[str, object]]`
- Produces: `score_segment_ppl(...) -> dict[str, object]`

- [ ] **Step 1: Write the failing tests**

```python
def test_segment_quality_loader_reads_test_watermarked_records(tmp_path):
    comparison = tmp_path / "segment"
    comparison.mkdir()
    (comparison / "metadata.json").write_text('{"baseline":"Segment-RSBH","experiment_dir":"."}\\n')
    (comparison / "records.jsonl").write_text(
        '{"split":"test","text_class":"watermarked","sample_id":"1","prompt_token_ids":[1],"token_ids":[2]}\\n'
    )
    examples = load_segment_quality_examples(comparison, include_shared=False)
    assert examples == [{
        "sample_id": "1",
        "split": "test",
        "text_class": "watermarked",
        "prompt_token_ids": [1],
        "continuation_token_ids": [2],
    }]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_segment_ppl.py -q`
Expected: FAIL because module does not exist.

- [ ] **Step 3: Write minimal implementation**

Mirror `evaluate_mpac_ppl.py` with `baseline == "Segment-RSBH"` and `token_ids` as continuation ids.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_segment_ppl.py -q`
Expected: PASS.

### Task 5: Verification And Run Commands

**Files:**
- No new production files.

**Interfaces:**
- Consumes: segment comparison CLI and segment PPL CLI.

- [ ] **Step 1: Run focused tests**

Run: `pytest dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_segment_rsbh_comparison.py dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_segment_ppl.py -q`
Expected: PASS.

- [ ] **Step 2: Run import wrappers**

Run: `python -m experiments.segment_rsbh_comparison --help`
Expected: exit 0 and include `--segment-repo`.

Run: `python -m experiments.evaluate_segment_ppl --help`
Expected: exit 0 and include `--comparison-dir`.

- [ ] **Step 3: Record real experiment commands**

Use:

```bash
python -m experiments.segment_rsbh_comparison \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --segment-repo /tmp/segment-watermark \
  --output-dir outputs/experiments/opt13b_pareto_49x200/comparisons/segment_rsbh_t200_b8_m2 \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --overwrite
```

Then:

```bash
python -m experiments.evaluate_segment_ppl \
  --comparison-dir outputs/experiments/opt13b_pareto_49x200/comparisons/segment_rsbh_t200_b8_m2 \
  --evaluator-model-path /data1/yanlu/models/facebook/opt-6.7b \
  --evaluator-name opt-6.7b \
  --batch-size 1 \
  --overwrite
```
