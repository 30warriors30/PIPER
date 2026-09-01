# Direct-Python Dual-Layer Watermark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the user-approved direct-Python dual-layer watermark project with separate generation, detection, evaluation, and calibration scripts.

**Architecture:** Core watermark logic lives under `watermark/`, metrics under `evaluation/`, and model/data/storage workflows under `utils/`. Four root scripts use argparse, construct validated dataclass configurations, print summaries, and call workflows. Generation writes `experiment.json` and `samples.jsonl`; detection reads those files and writes `detections.jsonl`; evaluation and calibration consume persisted results.

**Tech Stack:** Python 3.10, PyTorch 2.8.0+cu128, Transformers 4.x, Hugging Face Datasets, galois, NumPy, SciPy, scikit-learn, pytest.

## Global Constraints

- No YAML configuration.
- No `src/`, `pyproject.toml`, editable installation, or console-script command.
- Direct entry files: `run_generation.py`, `run_detection.py`, `evaluate_results.py`, `calibrate_threshold.py`.
- Project directories: `watermark/`, `evaluation/`, `utils/`.
- All samples in a run share watermark system parameters; each sample has a deterministic independent random payload.
- Generation and detection are separate processes.
- Preserve watermarked, unwatermarked, and natural text for each dataset sample.
- Blackwell environment uses Python 3.10 and PyTorch 2.8.0 CUDA 12.8.
- First release implements exact permutation and hash-mod allocation only.
- All production changes follow test-first development.

---

### Task 1: Project skeleton, configuration types, and CLI parsing

**Files:**
- Create: `watermark/config.py`
- Create: `utils/arguments.py`
- Create: `utils/validation.py`
- Create: `tests/test_arguments.py`
- Create: `tests/test_validation.py`

- [ ] Write failing parser and validation tests.
- [ ] Run tests and confirm expected failures.
- [ ] Implement dataclasses, argparse parsers, summary serialization, and validation.
- [ ] Run targeted tests.
- [ ] Commit.

### Task 2: Watermark primitives

**Files:**
- Create: `watermark/prf.py`
- Create: `watermark/partition.py`
- Create: `watermark/allocator.py`
- Create: `watermark/ecc.py`
- Create: `watermark/result_types.py`
- Create tests: `tests/test_prf.py`, `tests/test_partition.py`, `tests/test_allocator.py`, `tests/test_ecc.py`

- [ ] Write failing deterministic PRF, partition, allocator, and BCH tests.
- [ ] Run tests and confirm failures.
- [ ] Implement primitives.
- [ ] Run targeted tests.
- [ ] Commit.

### Task 3: Logits processor and detector

**Files:**
- Create: `watermark/logits_processor.py`
- Create: `watermark/detector.py`
- Create: `tests/test_logits_processor.py`
- Create: `tests/test_detector.py`

- [ ] Write failing embedding and detection tests.
- [ ] Run tests and confirm failures.
- [ ] Implement hard/soft embedding, counting modes, thresholding, strict/hard-fill decoding, known-boundary and blind detection.
- [ ] Run targeted tests.
- [ ] Commit.

### Task 4: Storage, seeds, environment, model, and dataset utilities

**Files:**
- Create: `utils/io.py`
- Create: `utils/seeds.py`
- Create: `utils/environment.py`
- Create: `utils/model.py`
- Create: `utils/datasets.py`
- Create: `utils/logging.py`
- Create: `tests/test_io_resume.py`
- Create: `tests/test_datasets.py`

- [ ] Write failing persistence/resume and dataset preparation tests.
- [ ] Run tests and confirm failures.
- [ ] Implement append-only JSONL, experiment compatibility checks, C4/OpenGen loading and prompt/natural splitting, model and Blackwell checks.
- [ ] Run targeted tests.
- [ ] Commit.

### Task 5: Generation workflow and entry script

**Files:**
- Create: `utils/generation.py`
- Create: `run_generation.py`
- Create: `tests/test_generation.py`

- [ ] Write failing paired-generation and CLI dry-run tests.
- [ ] Run tests and confirm failures.
- [ ] Implement deterministic per-sample payloads, common-seed paired generation, experiment/sample output, resume, and error handling.
- [ ] Run targeted tests.
- [ ] Commit.

### Task 6: Detection workflow and entry script

**Files:**
- Create: `utils/detection.py`
- Create: `run_detection.py`
- Create: `tests/test_detection_workflow.py`

- [ ] Write failing samples-mode and external-text detection tests.
- [ ] Run tests and confirm failures.
- [ ] Implement detection workflows, experiment parameter loading, known/blind records, resume, and external text mode.
- [ ] Run targeted tests.
- [ ] Commit.

### Task 7: Evaluation and calibration

**Files:**
- Create: `evaluation/metrics.py`
- Create: `evaluation/roc.py`
- Create: `evaluation/calibration.py`
- Create: `evaluation/report.py`
- Create: `evaluate_results.py`
- Create: `calibrate_threshold.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_calibration.py`

- [ ] Write failing metric/calibration tests.
- [ ] Run tests and confirm failures.
- [ ] Implement separate known/blind metrics, BER, ROC/AUC, calibration JSON, CSV and summary reports.
- [ ] Run targeted tests.
- [ ] Commit.

### Task 8: Environment, scripts, README, and integration verification

**Files:**
- Create: `environment.yml`
- Create: `requirements.txt`
- Create: `scripts/create_environment.sh`
- Create: `scripts/smoke_test.sh`
- Create: `scripts/run_c4_opt13b_200.sh`
- Create: `README.md`
- Create: `.gitignore`
- Create: `tests/test_integration_local_model.py`

- [ ] Add environment and documentation files.
- [ ] Run full offline test suite.
- [ ] Run compile checks.
- [ ] Package project ZIP.
- [ ] Commit and record verification limits for unavailable GPU/network integration.
