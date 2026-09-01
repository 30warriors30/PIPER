# Length Sweep B8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent experiment script that measures how detection and payload recovery change as exact generated text length `T` increases.

**Architecture:** Create `experiments/run_length_sweep.py` as a sibling to `run_capacity_fpr.py`, but make `exact_tokens` a per-run sweep value instead of a module constant. Keep raw payload `b` configurable in the CLI while currently allowing only `b=8`, mapped to BCH(23,8,3). The script writes one run directory per `T`, aggregates CSV/JSON, and generates plots for `TPR@1%FPR`, FPR, and EMR versus `T`.

**Tech Stack:** Python, PyTorch, Transformers, matplotlib, scipy, existing dual-layer watermark generator/detector utilities, pytest.

## Global Constraints

- Main sweep values: `T = 100, 200, 300, 500, 1000`.
- Default and currently supported raw payload length: `b = 8`.
- ECC for `b=8`: BCH `(23,8,3)`.
- Fixed deltas: `delta_presence = 0.0`, `delta_payload = 2.0`.
- Fixed target false-positive rate: `target_fpr = 0.01`.
- Presence mode: `soft`.
- Default model/data settings follow existing C4 realnewslike OPT experiment conventions.
- Output must include per-T metrics, aggregate JSON/CSV, and plots with `T` on the x-axis.

---

### Task 1: Test Length Sweep Parser And Aggregation

**Files:**
- Create: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_length_sweep.py`
- Create: `dual_layer_watermark_fixed_length_error_erasure/experiments/run_length_sweep.py`

**Interfaces:**
- Produces: `parse_t_values(values: Sequence[str] | None) -> tuple[int, ...]`
- Produces: `run_id(exact_tokens: int, capacity_bits: int) -> str`
- Produces: `aggregate_length_results(root: Path, t_values: Iterable[int], capacity_bits: int) -> list[dict[str, Any]]`

- [x] **Step 1: Write failing parser and aggregation tests**

Assert default `T` values are `(100, 200, 300, 500, 1000)`, parser defaults use `--b 8`, and aggregation sorts rows by increasing `T` while writing `length_results.json` and `length_results.csv`.

- [x] **Step 2: Run tests to verify failure**

Run: `/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_length_sweep.py -q`

Expected: fails because `experiments.run_length_sweep` does not exist.

- [ ] **Step 3: Implement minimal parser and aggregation**

Add default T values, b validation, run id formatting, and aggregate output writers.

- [ ] **Step 4: Run tests to verify pass**

Run: `/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_length_sweep.py -q`

Expected: all tests pass.

### Task 2: Implement Variable-T Generation And Detection

**Files:**
- Modify: `dual_layer_watermark_fixed_length_error_erasure/experiments/run_length_sweep.py`

**Interfaces:**
- Produces CLI: `python -m experiments.run_length_sweep --model-path ... --secret-key ...`
- Writes per-T run files under `runs/T{T}_b8/`

- [ ] **Step 1: Implement dataset manifest for max T**

Prepare samples with `max_new_tokens=max(t_values)` and store natural continuations for the maximum length. Shorter T runs use prefixes of the same natural token IDs.

- [ ] **Step 2: Generate shared baselines per T**

For each T, generate exact-length unwatermarked continuations and pair them with natural prefixes of length T.

- [ ] **Step 3: Generate watermarked continuations per T**

Use BCH(23,8,3), raw `b=8`, `DualLayerLogitsProcessor`, `presence_mode=soft`, `delta_presence=0.0`, `delta_payload=2.0`.

- [ ] **Step 4: Detect negatives and watermarked outputs per T**

Use theoretical one-sided Z threshold for `target_fpr=0.01`; write model/natural FPR, TPR, error-erasure EMR, and end-to-end EMR.

- [ ] **Step 5: Aggregate and plot**

Write `length_results.json`, `length_results.csv`, `figures/length_vs_tpr.png`, `figures/length_vs_fpr.png`, and `figures/length_vs_emr.png`.

### Task 3: Verify Script Entry Points

**Files:**
- Modify: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_length_sweep.py`

**Interfaces:**
- Consumes: `build_parser()`

- [ ] **Step 1: Add CLI help smoke test**

Run `python -m experiments.run_length_sweep --help` from the project root and assert it exposes `--t-values` and `--b`.

- [ ] **Step 2: Add dry-run test**

Assert `--dry-run` prints the plan without loading a model or writing run outputs.

- [ ] **Step 3: Run targeted tests**

Run: `/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_length_sweep.py -q`

Expected: all tests pass.
