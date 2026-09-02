# Batch Engine v2 Comparison Adapters and Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish batch support in comparison generators, audit every production CLI, document v2/hybrid operation, and resume the stopped length experiment without recomputing completed records.

**Architecture:** External comparison methods use the shared model batch path with one upstream processor adapter per row. A CLI audit enforces the same execution defaults everywhere. Rollout first verifies a pure-v2 smoke run, then creates an explicit migration sidecar for the stopped length directory and resumes only its recorded missing-key workloads.

**Tech Stack:** Python 3.10, PyTorch, Transformers, external MPAC and Segment adapters, argparse, pytest, shell/NVIDIA tooling.

**Spec:** `docs/superpowers/specs/2026-09-02-batch-engine-v2-design.md`

## Global Constraints

- Complete the core, generation, and detection implementation plans first.
- Do not compare v1/v2 output text, metrics, or performance.
- MPAC and Segment/RS-BH model forward passes must accept generation batches; their row-specific processors remain isolated through the adapter.
- Every production generation/detection command exposes defaults `16/64/8` and runtime overrides.
- Existing quality/PPL-specific batch options remain available and are not conflated with detection workers.
- The current Pareto PID must not be terminated by rollout commands.
- The stopped length output is hybrid-imported in place only after complete JSONL and digest validation.
- `--import-v1-completed` is used once; later resumes omit it.

---

### Task 1: MPAC Batched Generation and Detection Adapter

**Files:**
- Modify: `experiments/mpac_comparison.py`
- Modify: `experiments/mpac_positive_tpr.py`
- Test: `tests/experiments/test_mpac_comparison.py`
- Test: `tests/experiments/test_mpac_positive_tpr.py`

**Interfaces:**
- Consumes: `BatchGenerator`, `GenerationRequest`, `PerRowLogitsProcessorAdapter`.
- Produces: MPAC batched baseline/positive generation and ordered batched scoring with v2 provenance.

- [ ] **Step 1: Write failing MPAC row-isolation batch test**

```python
def test_mpac_batch_builds_one_processor_per_row():
    rows = [mpac_row("a", "01010101"), mpac_row("b", "10101010")]
    generator = make_fake_mpac_generator(batch_size=2)
    results = generator.generate_positive_batch(rows)
    assert fake_model.batch_size == 2
    assert adapter_processor_messages() == ["01010101", "10101010"]
    assert [row.sample_id for row in results] == ["a", "b"]


def test_mpac_positive_scoring_runs_after_generation_in_batches():
    generated = [mpac_generated_row(str(index)) for index in range(5)]
    scored = score_mpac_generated(generated, detection_batch_size=4, detection_workers=1)
    assert mpac_score_task_sizes() == [4, 1]
    assert [row["sample_id"] for row in scored] == [str(index) for index in range(5)]
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_mpac_comparison.py tests/experiments/test_mpac_positive_tpr.py -q
```

Expected: FAIL because MPAC calls `model.generate()` for one row.

- [ ] **Step 3: Add generation batch arguments and build request batches**

Use the common CLI helper. Preserve MPAC sample/message/position seeds in manifest order. Create one upstream logits processor per row, wrap the processors in `PerRowLogitsProcessorAdapter`, and pass one padded batch to the model.

- [ ] **Step 4: Batch MPAC positive and negative scoring**

Finish ordered generation output before scoring. Group positive and negative MPAC detector inputs by `detection_batch_size`, initialize detector state once per worker, and return ordered score records. Do not invoke an external detector inside the model-generation row loop.

- [ ] **Step 5: Serialize provenance without changing MPAC metrics fields**

Add `engine_version`, batch ID, actual batch size, batch seconds, and amortized seconds. Keep algorithm name, message, encoded bits, token IDs, and existing MPAC score fields unchanged.

- [ ] **Step 6: Run and commit MPAC batching**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_mpac_comparison.py tests/experiments/test_mpac_positive_tpr.py -q
```

Expected: PASS.

Commit:

```bash
git add experiments/mpac_comparison.py experiments/mpac_positive_tpr.py tests/experiments/test_mpac_comparison.py tests/experiments/test_mpac_positive_tpr.py
git commit -m "Batch MPAC comparison generation and detection"
```

---

### Task 2: Segment/RS-BH Batched Generation and Detection Adapter

**Files:**
- Modify: `experiments/segment_rsbh_comparison.py`
- Test: `tests/experiments/test_segment_rsbh_comparison.py`

**Interfaces:**
- Consumes: shared batch generator and per-row processor adapter.
- Produces: Segment/RS-BH batched positive generation and ordered batched score records.

- [ ] **Step 1: Write failing Segment batch test**

Use three rows with distinct messages and batch size 2. Assert model calls have actual sizes 2 and 1, each upstream processor sees only its row, and outputs preserve source order and exact length. Then score with detection batch size 2 and assert task sizes 2 and 1 in the same source order.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_segment_rsbh_comparison.py -q
```

Expected: FAIL at the existing `batch_size=1` guard.

- [ ] **Step 3: Remove the local batch-one guard and use shared batches**

Replace local input tensor/model generation with `GenerationRequest` batches. Create one Segment processor/state object per row and wrap them. Preserve Segment repository setup, message encoding, score calls, and record fields.

- [ ] **Step 4: Batch Segment positive and negative scoring**

Separate generation from scoring. Initialize the external detector once per worker, send ordered score batches, and serialize results only in the parent. Preserve existing Segment/RS-BH score and payload fields.

- [ ] **Step 5: Run and commit Segment batching**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_segment_rsbh_comparison.py -q
```

Expected: PASS.

Commit:

```bash
git add experiments/segment_rsbh_comparison.py tests/experiments/test_segment_rsbh_comparison.py
git commit -m "Batch Segment RSBH generation and detection"
```

---

### Task 3: Production CLI Coverage Audit

**Files:**
- Modify: `experiments/generate_shared_negative_baseline.py`
- Modify: `experiments/evaluate_external_ppl.py`
- Modify: `experiments/evaluate_mpac_ppl.py`
- Modify: `experiments/evaluate_segment_ppl.py`
- Modify: `experiments/run_length_sweep.py`
- Modify: `experiments/run_capacity_fpr.py`
- Modify: `experiments/run_paper_null.py`
- Modify: `experiments/mpac_comparison.py`
- Modify: `experiments/segment_rsbh_comparison.py`
- Modify: `experiments/attack/run_synonym_attacks.py`
- Modify: `experiments/attack/run_paraphrase_attacks.py`
- Create: `tests/experiments/test_batch_cli_coverage.py`
- Modify: `tests/experiments/test_external_quality_cli.py`

**Interfaces:**
- Consumes: common batch argument helper.
- Produces: a parameterized test over every production parser.

- [ ] **Step 1: Write a failing parser coverage matrix**

```python
@pytest.mark.parametrize(
    "parser_factory,argv,expected",
    [
        (generation_parser, GENERATION_REQUIRED, (16, 64, 8)),
        (detection_parser, ["--input-type", "samples"], (None, 64, 8)),
        (pareto_parser, PARETO_REQUIRED, (16, 64, 8)),
        (length_parser, LENGTH_REQUIRED, (16, 64, 8)),
        (capacity_parser, CAPACITY_REQUIRED, (16, 64, 8)),
        (paper_null_parser, PAPER_NULL_REQUIRED, (None, 64, 8)),
        (mpac_parser, MPAC_REQUIRED, (16, 64, 8)),
        (segment_parser, SEGMENT_REQUIRED, (16, 64, 8)),
        (synonym_parser, SYNONYM_REQUIRED, (None, 64, 8)),
        (paraphrase_parser, PARAPHRASE_REQUIRED, (None, 64, 8)),
    ],
)
def test_all_production_parsers_expose_batch_defaults(parser_factory, argv, expected):
    args = parser_factory().parse_args(argv)
    actual = (
        getattr(args, "generation_batch_size", None),
        getattr(args, "detection_batch_size", None),
        getattr(args, "detection_workers", None),
    )
    assert actual == expected
```

- [ ] **Step 2: Run coverage tests and verify missing parsers fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_batch_cli_coverage.py tests/experiments/test_external_quality_cli.py -q
```

Expected: FAIL listing every unthreaded parser.

- [ ] **Step 3: Wire common arguments and printed plans**

Add the helper to every listed parser, pass settings to called generation/detection functions, and print values in dry-run plans. Commands that only score PPL retain their existing `--batch-size`; change its default from 1 to 16 only where it represents generation-model/PPL forward batching and tests prove the model fixture supports it.

- [ ] **Step 4: Add override and invalid-value coverage**

Assert `--generation-batch-size 3 --detection-batch-size 5 --detection-workers 2` reaches runtime config, and each zero value fails before model loading.

Add `--migration-dry-run` to length and capacity parsers. It requires
`--resume --import-v1-completed`, validates and prints the proposed frozen and
missing artifact counts, and returns before writing a sidecar or loading a
model. Test the stopped-run fixture's printed T100/T200 plan and assert the
fixture tree is byte-for-byte unchanged.

- [ ] **Step 5: Run and commit CLI audit**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_arguments.py tests/experiments/test_batch_cli_coverage.py tests/experiments/test_external_quality_cli.py -q
```

Expected: PASS.

Commit:

```bash
git add experiments/generate_shared_negative_baseline.py experiments/evaluate_external_ppl.py experiments/evaluate_mpac_ppl.py experiments/evaluate_segment_ppl.py experiments/run_length_sweep.py experiments/run_capacity_fpr.py experiments/run_paper_null.py experiments/mpac_comparison.py experiments/segment_rsbh_comparison.py experiments/attack/run_synonym_attacks.py experiments/attack/run_paraphrase_attacks.py tests/test_arguments.py tests/experiments/test_batch_cli_coverage.py tests/experiments/test_external_quality_cli.py
git commit -m "Expose batch controls across experiment CLIs"
```

---

### Task 4: Documentation and Hybrid Commands

**Files:**
- Modify: `README.md`
- Modify: `PAPER_EXPERIMENTS.md`
- Modify: `experiments/README.md`
- Modify: `scripts/run_pareto_sweep.sh`

**Interfaces:**
- Consumes: final CLI names and migration behavior.
- Produces: exact pure-v2 and one-time hybrid commands.

- [ ] **Step 1: Update common defaults and execution semantics**

Document:

```text
--generation-batch-size 16
--detection-batch-size 64
--detection-workers 8
```

State that batch size is part of the sampling configuration, v2 does not reproduce v1 token IDs, OOM requires a new output directory or unstarted pure-v2 run with a smaller explicit batch, and worker count controls CPU detection only.

- [ ] **Step 2: Add pure-v2 formal command examples**

Every generation-capable paper command includes all three values. Detection-only and attack commands include detection batch/workers. Keep `CUDA_VISIBLE_DEVICES` examples explicit and explain physical-to-logical GPU remapping.

- [ ] **Step 3: Add one-time and subsequent hybrid commands**

Document first import:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/home/yanlu/miniconda3/envs/dual-watermark-blackwell/lib \
python -m experiments.run_length_sweep \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --secret-key dual-layer-key-2026 \
  --dataset-path /data/yanlu/BREW/dataset/c4/c4_realnewslike_validation_t1000.jsonl \
  --test-samples 1000 --t-values 100 200 300 500 1000 \
  --output-dir outputs/experiments/piper_length_b8_m2 \
  --generation-batch-size 16 --detection-batch-size 64 --detection-workers 8 \
  --resume --import-v1-completed
```

Document subsequent resume with the same command but without `--import-v1-completed`.

- [ ] **Step 4: Validate documented commands through parsers**

Extract the argument lists into parser tests or run each command with `--dry-run`. Expected: every command exits 0 without model loading or output mutation.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md PAPER_EXPERIMENTS.md experiments/README.md scripts/run_pareto_sweep.sh tests/experiments/test_batch_cli_coverage.py
git commit -m "Document batch engine v2 execution"
```

---

### Task 5: Full Automated Verification

**Files:**
- No file changes. A failure returns execution to the task that owns the failed behavior before this verification task is rerun.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: release evidence for v2 and hybrid rollout.

- [ ] **Step 1: Run formatting and static syntax checks**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m compileall -q watermark utils experiments run_generation.py run_detection.py
git diff --check
```

Expected: both exit 0.

- [ ] **Step 2: Run the complete unit/integration suite without local GPU model**

Run:

```bash
/usr/bin/env LD_LIBRARY_PATH=/home/yanlu/miniconda3/envs/dual-watermark-blackwell/lib /home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest -q
```

Expected: PASS with zero failures.

- [ ] **Step 3: Run local OPT batch smoke**

Run:

```bash
/usr/bin/env CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/yanlu/miniconda3/envs/dual-watermark-blackwell/lib LOCAL_OPT_MODEL=/data/yanlu/BREW/models/facebook/opt-1.3b /home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_integration_local_model.py tests/test_batch_generation.py tests/test_batch_detection.py -q -s
```

Expected: PASS for batch sizes 1, 2, and 16 smoke fixtures without OOM.

- [ ] **Step 4: Run a new pure-v2 eight-sample smoke experiment**

Use a new output directory `outputs/paper_batch_v2_smoke`, exact 64 tokens, generation batch 4, detection batch 8, workers 2. Run generic generation, generic detection, and result evaluation. Assert 8 completed three-class generation records and 48 detection keys.

- [ ] **Step 5: Inspect smoke provenance**

Assert every generated record has engine version 2, actual batch sizes `4`, and exact 64-token continuations. Assert every detection has source engine 2. Do not compare against any v1 smoke output.

---

### Task 6: Import and Resume the Stopped Length Experiment

**Files:**
- Runtime output only: `outputs/experiments/piper_length_b8_m2/engine_migration.json`
- Runtime output only: missing JSONL records and later metrics/figures under the same root.

**Interfaces:**
- Consumes: completed Batch Engine v2 implementation and verified current output.
- Produces: a running hybrid length sweep that skips all frozen completion keys.

- [ ] **Step 1: Confirm no length process is running and Pareto remains untouched**

Run:

```bash
pgrep -af 'experiments.run_length_sweep'
pgrep -af 'experiments.run_pareto_sweep'
```

Expected: no length process; the existing Pareto process may remain present and must not be signaled.

- [ ] **Step 2: Revalidate all current JSONL artifacts**

Parse every JSONL line, reject duplicate completion keys, and assert at least:

```text
manifest=1000
T100 baseline=1000, negatives=2000, watermarked=1000, detections=1000
T200 baseline=1000, negatives=2000, watermarked=730
```

Use actual validated counts if another complete record was flushed before the old process stopped; never truncate a valid record to force the expected count.

- [ ] **Step 3: Run migration preflight without model loading**

Add and invoke a `--migration-dry-run` mode with the full hybrid command. It prints frozen and pending counts by artifact but writes nothing. Expected plan:

```text
T100: no generation or detection work
T200 baseline/negative: no work
T200 watermarked: 1000 - validated legacy count pending
T200 detection: all generated rows lacking detection pending, routed by source engine
T300/T500/T1000: full v2 work
```

- [ ] **Step 4: Execute the one-time migration import**

Run from `/data/yanlu/BREW/PIPER`:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/home/yanlu/miniconda3/envs/dual-watermark-blackwell/lib \
python -m experiments.run_length_sweep \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --secret-key dual-layer-key-2026 \
  --dataset-path /data/yanlu/BREW/dataset/c4/c4_realnewslike_validation_t1000.jsonl \
  --test-samples 1000 --t-values 100 200 300 500 1000 \
  --output-dir outputs/experiments/piper_length_b8_m2 \
  --generation-batch-size 16 --detection-batch-size 64 --detection-workers 8 \
  --resume --import-v1-completed
```

Expected: migration sidecar is created before model loading; progress skips all T100 work and frozen T200 artifacts.

- [ ] **Step 5: Verify the live process and first v2 batch**

Confirm the process environment contains `CUDA_VISIBLE_DEVICES=0`, execution settings are recorded, and the first appended v2 generation records have one immutable batch ID, actual batch size at most 16, engine version 2, and exact T tokens. Verify the legacy file prefix digest still matches the sidecar.

- [ ] **Step 6: Record the subsequent resume command**

If interrupted later, run the same command with `--resume` and without `--import-v1-completed`. Repeated import must fail before model loading.

- [ ] **Step 7: Final result verification after completion**

Assert every T point has 1000 baseline, 2000 negative detections, 1000 watermarked, 1000 watermarked detections, metrics JSON, aggregate tables, and figures. Verify each metrics file reports engine composition: T100 is `legacy-v1`, T200 is `hybrid-v1-v2`, and T300/T500/T1000 are `pure-v2`.
