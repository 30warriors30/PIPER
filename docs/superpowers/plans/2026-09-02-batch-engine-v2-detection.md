# Batch Engine v2 Detection and Experiment Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add engine-aware constant-time batched detection with ordered process-pool execution and migrate every PIPER detection path.

**Architecture:** `DualLayerDetector` selects legacy PCG64 or stateless v2 partition semantics per detector instance. A shared `BatchDetectionExecutor` groups requests, routes mixed hybrid batches to long-lived v1/v2 worker detectors, and restores source order. Generic, Pareto, length, capacity, null, attack, and rescoring paths build requests instead of invoking `detect_token_ids()` in row loops.

**Tech Stack:** Python 3.10, `concurrent.futures.ProcessPoolExecutor`, SciPy statistics, NumPy, BCH codec, JSONL, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-batch-engine-v2-design.md`

## Global Constraints

- Complete the core and generation/persistence plans first.
- Defaults are detection batch `64` and workers `8`.
- Detection decisions must be identical between scalar and batch v2 paths; timing fields may differ.
- Hybrid records without `engine_version`, and frozen migration records, use partition engine v1.
- Records marked `engine_version=2` use partition engine v2.
- Mixed-engine tasks restore original source order and write `source_engine_version`.
- Known-boundary and blind-text context semantics do not change.
- Only requested counting modes and decoding policies run; output schema marks unrequested policies `not_evaluated`.
- Worker processes never write output files.
- No automatic worker or batch fallback is allowed.

---

### Task 1: Engine-Aware Scalar Detector

**Files:**
- Modify: `watermark/detector.py`
- Modify: `watermark/result_types.py`
- Modify: `utils/detection.py`
- Test: `tests/test_detector.py`
- Test: `tests/test_detection_workflow.py`

**Interfaces:**
- Consumes: legacy `ExactPermutationPartitioner`, v2 `StatelessExactPartitioner`, `partition_round_keys`.
- Produces: a `partition_engine: Literal["v1", "v2"] = "v1"` keyword in the existing `DualLayerDetector` constructor.
- Produces: `requested_counting_modes` and existing `evaluate_all_policies` controls.
- Preserves: `detect_continuation`, `detect_with_boundary`, and `detect_token_ids` signatures.

- [ ] **Step 1: Write failing v2 scalar and selective-evaluation tests**

```python
def test_v2_detector_regions_match_v2_partitioner():
    detector = make_detector(partition_engine="v2", evaluate_all_policies=False)
    context = (2, 3, 4, 5)
    partition, bit_index = detector.partition_for_context(context)
    result = detector.detect_continuation(context, [partition.bit1[0]], decode_payload=False)
    assert result.counting["all_tokens"].n1[bit_index] == 1


def test_primary_only_detection_marks_other_modes_not_evaluated():
    detector = make_detector(
        partition_engine="v2",
        primary_counting_mode="unique_context",
        requested_counting_modes=("unique_context",),
        evaluate_all_policies=False,
    )
    result = detector.detect_token_ids([1, 2, 3, 4, 5])
    assert set(result.counting) == {"unique_context"}
    assert result.strict_decode.status == "not_evaluated"
```

- [ ] **Step 2: Run detector tests and verify they fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_detector.py tests/test_detection_workflow.py -q
```

Expected: FAIL because engine selection and counting-mode selection do not exist.

- [ ] **Step 3: Implement partition-engine selection**

Store `partition_engine`. For v1, preserve current full-permutation behavior exactly. For v2, resolve partition/position base seeds through `_seeds`, expand partition round keys, and call `region_for_token` for each observed token. `partition_for_context` may materialize diagnostic partitions through the selected engine.

- [ ] **Step 4: Implement requested counting modes**

Normalize `requested_counting_modes=None` to all three legacy modes for backward compatibility. Require the primary mode to be included. `_detect_events` computes only normalized modes. Keep all decode result fields; use `_not_evaluated_decode` for policies not requested.

- [ ] **Step 5: Add v1 regression vectors**

Pin one existing v1 token sequence's `upper_hits`, `n0`, `n1`, p-value, and gated message before refactoring. Assert the same vector after engine selection is introduced.

- [ ] **Step 6: Run and commit scalar detector changes**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_detector.py tests/test_detection_workflow.py tests/test_paper_conformance.py -q
```

Expected: PASS.

Commit:

```bash
git add watermark/detector.py watermark/result_types.py utils/detection.py tests/test_detector.py tests/test_detection_workflow.py tests/test_paper_conformance.py
git commit -m "Add engine-aware v2 scalar detection"
```

---

### Task 2: Ordered Batch Detection Executor

**Files:**
- Create: `utils/batch_detection.py`
- Test: `tests/test_batch_detection.py`

**Interfaces:**
- Produces: `DetectionRequest`
- Produces: `DetectionOutcome`
- Produces: `DetectorFactoryConfig`
- Produces: `BatchDetectionExecutor.map(requests) -> Iterator[DetectionOutcome]`

- [ ] **Step 1: Write failing in-process ordering and routing tests**

```python
def test_batch_executor_routes_mixed_engines_and_preserves_order():
    requests = [
        DetectionRequest(0, "a", "natural", "blind_text", (), (1, 2, 3), 1, False),
        DetectionRequest(1, "b", "watermarked", "blind_text", (), (4, 5, 6), 2, True),
        DetectionRequest(2, "c", "natural", "known_boundary", (7, 8), (9,), 1, False),
    ]
    outcomes = list(make_executor(workers=1, batch_size=2).map(requests))
    assert [row.source_index for row in outcomes] == [0, 1, 2]
    assert [row.source_engine_version for row in outcomes] == [1, 2, 1]
    assert detector_calls() == [("v1", "a"), ("v2", "b"), ("v1", "c")]
```

- [ ] **Step 2: Run executor tests and verify they fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_batch_detection.py -q
```

Expected: FAIL because the executor does not exist.

- [ ] **Step 3: Implement serializable request/config/result shapes**

```python
@dataclass(frozen=True)
class DetectionRequest:
    source_index: int
    sample_id: str
    text_class: str
    input_mode: Literal["known_boundary", "blind_text"]
    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    source_engine_version: int
    decode_payload: bool


@dataclass(frozen=True)
class DetectionOutcome:
    request: DetectionRequest
    result: DetectionResult | None
    elapsed_seconds: float
    error_type: str | None = None
    error_message: str | None = None
    traceback_tail: str | None = None
```

`DetectorFactoryConfig` contains only JSON/pickle-safe detector constructor values and builds both v1 and v2 detector instances.

- [ ] **Step 4: Implement long-lived worker batches**

The process initializer creates global v1/v2 detectors once. A task receives a tuple of requests, partitions them by `source_engine_version`, detects each with the matching detector, then sorts outcomes by `source_index`. Version values other than 1 or 2 return an indexed error outcome.

With workers `1`, call the same worker function in-process. With workers above 1, use `ProcessPoolExecutor(max_workers=workers, initializer=_init_detection_worker, initargs=(factory_config,))`, submit one future per immutable request batch, consume futures in batch-ID order rather than completion order, and yield source-ordered outcomes.

- [ ] **Step 5: Add process-pool equality and failure-isolation tests**

```python
def test_workers_one_and_two_have_equal_decisions():
    serial = strip_timing(make_executor(workers=1, batch_size=3).map(real_requests()))
    parallel = strip_timing(make_executor(workers=2, batch_size=3).map(real_requests()))
    assert parallel == serial


def test_one_bad_request_does_not_discard_siblings():
    outcomes = list(make_executor(workers=1, batch_size=3).map(requests_with_one_bad_id()))
    assert [row.error_type is None for row in outcomes] == [True, False, True]
```

- [ ] **Step 6: Run and commit the executor**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_batch_detection.py tests/test_detector.py -q
```

Expected: PASS.

Commit:

```bash
git add utils/batch_detection.py tests/test_batch_detection.py
git commit -m "Add ordered process-pool batch detection"
```

---

### Task 3: Generic and Pareto Detection Migration

**Files:**
- Modify: `utils/detection.py`
- Modify: `run_detection.py`
- Modify: `experiments/detection.py`
- Modify: `experiments/run_pareto_sweep.py`
- Test: `tests/test_detection_workflow.py`
- Test: `tests/experiments/test_z_calibration_routing.py`
- Create: `tests/experiments/test_detection_batches.py`

**Interfaces:**
- Consumes: `BatchDetectionExecutor`, execution config, migration engine routing.
- Produces: batched `run_samples_detection`, `run_text_detection`, Pareto negative/positive detection.

- [ ] **Step 1: Write failing generic hybrid detection test**

Seed a sample file with one v1 record lacking `engine_version` and one v2 record. Invoke sample detection with workers 1 and batch size 8. Assert six text-class/mode outputs per sample, source-engine values 1 and 2, and stable source order.

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_detection_workflow.py tests/experiments/test_detection_batches.py -q
```

Expected: FAIL because workflows call scalar detectors directly.

- [ ] **Step 3: Refactor generic request construction and writes**

Build all missing `(sample_id, text_class, input_mode)` requests before execution. Infer source version from record provenance or migration frozen keys. Parent code converts outcomes through existing `detection_record`, adds `source_engine_version`, and writes one completed executor batch with `append_jsonl_batch`. Error outcomes go to both normal and error files without suppressing successful siblings.

Pass CLI detection batch/workers into both sample and text modes. Text-mode inputs default to engine v2 unless an explicit `engine_version` field is present.

- [ ] **Step 4: Refactor Pareto detection**

Replace `_detect_modes` row loops in shared-negative and operating-point detection with executor requests. Negative baseline records inherit their generation engine. Watermarked records inherit their own engine. Keep calibration split and exact-binomial routing unchanged.

- [ ] **Step 5: Verify completed-key resume**

Prewrite one detection key from a three-request fixture, run with resume, and assert the executor receives only two requests. Assert output has three unique keys and one fsync batch for the two new records.

- [ ] **Step 6: Run and commit workflow migration**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_detection_workflow.py tests/experiments/test_detection_batches.py tests/experiments/test_z_calibration_routing.py -q
```

Expected: PASS.

Commit:

```bash
git add run_detection.py utils/detection.py experiments/detection.py experiments/run_pareto_sweep.py tests/test_detection_workflow.py tests/experiments/test_detection_batches.py tests/experiments/test_z_calibration_routing.py
git commit -m "Batch generic and Pareto detection"
```

---

### Task 4: Length and Capacity Detection Migration

**Files:**
- Modify: `experiments/run_length_sweep.py`
- Modify: `experiments/run_capacity_fpr.py`
- Test: `tests/experiments/test_length_sweep.py`
- Test: `tests/experiments/test_capacity_fpr.py`

**Interfaces:**
- Consumes: shared executor and migration provenance.
- Produces: batched shared-negative and watermarked detection for all T/b points.

- [ ] **Step 1: Extend length hybrid fixture with detection assertions**

For the stopped-run fixture, assert T100 schedules no detection, existing T200 negatives schedule none, T200 schedules 3 legacy-v1 and 2 new-v2 watermarked detection requests after generation, and T300 schedules five v2 requests.

- [ ] **Step 2: Run focused tests and verify they fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_length_sweep.py tests/experiments/test_capacity_fpr.py -q
```

Expected: FAIL because local scalar detection loops remain.

- [ ] **Step 3: Replace local detection loops with request adapters**

Build one request per negative text class and one per watermarked row. Preserve current output filenames, record shapes, progress totals, metric inputs, exact-token/capacity fields, and BCH truth values. Add `source_engine_version` to every new detection.

- [ ] **Step 4: Add engine-composition metrics**

Each point's metrics JSON receives:

```json
{
  "engine_composition": {
    "mode": "pure-v2",
    "generation": {"v1": 0, "v2": 1000},
    "detection": {"v1": 0, "v2": 1000}
  }
}
```

Use `mode="legacy-v1"` when the point contains only v1 records,
`mode="pure-v2"` when it contains only v2 records, and
`mode="hybrid-v1-v2"` only when both versions are present. The experiment-root
metadata remains `hybrid-v1-v2` after any legacy import. Compute all counts
from records and the migration sidecar, not assumptions.

- [ ] **Step 5: Run and commit length/capacity detection**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_length_sweep.py tests/experiments/test_capacity_fpr.py -q
```

Expected: PASS.

Commit:

```bash
git add experiments/run_length_sweep.py experiments/run_capacity_fpr.py tests/experiments/test_length_sweep.py tests/experiments/test_capacity_fpr.py
git commit -m "Batch length and capacity detection"
```

---

### Task 5: Null and Shared-Negative FPR Detection Migration

**Files:**
- Modify: `experiments/run_paper_null.py`
- Modify: `experiments/score_shared_negative_fpr.py`
- Test: `tests/experiments/test_paper_null.py`
- Test: `tests/experiments/test_shared_negative_fpr.py`

**Interfaces:**
- Consumes: `BatchDetectionExecutor` and common batch CLI helper.
- Produces: batched multi-key null requests and ordered shared-negative results.

- [ ] **Step 1: Write failing multi-key batch test**

Use three texts and two keys. Assert six requests are grouped by detection batch size, results remain text-major/key-minor, and per-key/combined summaries receive the same six p-values.

- [ ] **Step 2: Run focused tests and verify they fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_paper_null.py tests/experiments/test_shared_negative_fpr.py -q
```

Expected: FAIL because paper-null loops through detector objects.

- [ ] **Step 3: Add common batch CLI controls**

Add `--detection-batch-size 64` and `--detection-workers 8`. Remove the separate shared-negative `--workers 32` option after adding a deprecated alias that maps to `detection_workers` for one release and rejects use of both names.

- [ ] **Step 4: Batch null work by key and text**

Extend the detector factory key with a secret-key ID so workers cache one detector per key/engine combination. Requests carry key ID in metadata. Write ordered records in batches and preserve exact Clopper-Pearson summary calculations.

- [ ] **Step 5: Refactor shared-negative scoring futures**

Submit request batches instead of one future per row. Preserve MPAC's separate detector initializer through a generic algorithm factory and write results in source order without a post-hoc sample-ID sort.

- [ ] **Step 6: Run and commit null/FPR migration**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_paper_null.py tests/experiments/test_shared_negative_fpr.py -q
```

Expected: PASS.

Commit:

```bash
git add experiments/run_paper_null.py experiments/score_shared_negative_fpr.py tests/experiments/test_paper_null.py tests/experiments/test_shared_negative_fpr.py
git commit -m "Batch null and shared-negative detection"
```

---

### Task 6: Attack and Rescoring Detection Migration

**Files:**
- Modify: `experiments/attack/run_synonym_attacks.py`
- Modify: `experiments/attack/run_paraphrase_attacks.py`
- Modify: `experiments/mpac_rescore_positive_tpr.py`
- Modify: `experiments/mpac_positive_tpr.py`
- Test: `tests/experiments/test_synonym_attacks.py`
- Test: `tests/experiments/test_paraphrase_attacks.py`
- Test: `tests/experiments/test_mpac_rescore_positive_tpr.py`
- Test: `tests/experiments/test_mpac_positive_tpr.py`

**Interfaces:**
- Consumes: common batch arguments and executor.
- Produces: batched detection after attack transformation/paraphrasing and MPAC rescoring.

- [ ] **Step 1: Write failing attack detection batching tests**

Generate five attacked examples, request both input modes, and use batch size 4. Assert ten ordered detection outcomes arrive in task sizes 4/4/2, while attack/paraphrase record order remains unchanged.

- [ ] **Step 2: Run focused tests and verify they fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_synonym_attacks.py tests/experiments/test_paraphrase_attacks.py tests/experiments/test_mpac_rescore_positive_tpr.py tests/experiments/test_mpac_positive_tpr.py -q
```

Expected: FAIL because each transformed example is detected immediately.

- [ ] **Step 3: Separate transformation from detection**

For synonym and paraphrase runners, complete ordered attacked-record construction first. Convert all requested input modes to `DetectionRequest` values, invoke one executor, and attach attack timing/seed metadata when serializing outcomes. Keep paraphraser `--paraphraser-batch-size` independent from detection batch size.

- [ ] **Step 4: Refactor MPAC rescore paths into request batches**

Replace one-future-per-record code with ordered task batches using the same `detection_batch_size` and `detection_workers` names. Preserve MPAC-specific result fields and source ordering.

- [ ] **Step 5: Run and commit attack/rescore migration**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_synonym_attacks.py tests/experiments/test_paraphrase_attacks.py tests/experiments/test_mpac_rescore_positive_tpr.py tests/experiments/test_mpac_positive_tpr.py -q
```

Expected: PASS.

Commit:

```bash
git add experiments/attack/run_synonym_attacks.py experiments/attack/run_paraphrase_attacks.py experiments/mpac_rescore_positive_tpr.py experiments/mpac_positive_tpr.py tests/experiments/test_synonym_attacks.py tests/experiments/test_paraphrase_attacks.py tests/experiments/test_mpac_rescore_positive_tpr.py tests/experiments/test_mpac_positive_tpr.py
git commit -m "Batch attack and MPAC rescoring detection"
```
