# Fixed-Length Generation and Error-Erasure BCH Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every completed watermarked, unwatermarked, and natural continuation contain exactly the configured 200 tokens and add a bounded BCH error-erasure decoding policy.

**Architecture:** Dataset preparation will preserve exact prompt and natural token IDs and stream candidates until the requested number of valid triples is completed. BCH error-erasure decoding will enumerate erased positions up to a fixed assignment bound, reuse the ordinary BCH decoder, re-encode candidates, and enforce `2*v + e < d`. Detection and evaluation will expose strict, hard-fill, and error-erasure results separately.

**Tech Stack:** Python 3.10, PyTorch/Transformers generation, shortened binary BCH codec, pytest.

## Global Constraints

- Preserve the four direct Python entry points and JSON/JSONL output protocol.
- `--max-new-tokens` remains accepted but means exact continuation length.
- Every completed sample has identical `watermarked`, `unwatermarked`, and `natural` token counts.
- Default primary decoding policy is `error_erasure`.
- Error-erasure enumeration is bounded to 64 assignments by default.
- Existing strict and hard-fill behavior remains available.

---

### Task 1: Exact token-preserving dataset preparation

**Files:**
- Modify: `utils/datasets.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Produces: `SampleRecord.prompt_token_ids`, `SampleRecord.natural_token_ids`, `prepare_sample(..., exact_new_tokens, ...)`.
- Filtering raises `SampleFilterError` carrying reason and token-count diagnostics.

- [ ] Write failing tests for exact C4 tail extraction, short-document filtering, and tokenizer-added prompt special tokens.
- [ ] Run `pytest tests/test_datasets.py -v` and verify failures.
- [ ] Implement token-preserving preparation and unbounded candidate streaming.
- [ ] Run `pytest tests/test_datasets.py -v` and verify passes.
- [ ] Commit dataset preparation.

### Task 2: Exact-length paired generation and completion-count loop

**Files:**
- Modify: `utils/generation.py`
- Modify: `watermark/config.py`
- Modify: `utils/arguments.py`
- Test: `tests/test_generation.py`
- Test: `tests/test_arguments.py`

**Interfaces:**
- `PairedGenerator._generation_kwargs()` supplies equal `min_new_tokens` and `max_new_tokens`.
- `run_generation()` continues until `max_samples` unique completed triples or finite-dataset exhaustion.

- [ ] Write failing tests for generation kwargs, three exact token counts, candidate filtering, and target completed count.
- [ ] Run focused tests and verify failures.
- [ ] Implement exact generation, direct prompt IDs, direct natural IDs, filtering log, and length invariants.
- [ ] Run focused tests and verify passes.
- [ ] Commit exact-length generation.

### Task 3: Bounded error-erasure BCH codec

**Files:**
- Modify: `watermark/result_types.py`
- Modify: `watermark/ecc.py`
- Test: `tests/test_ecc.py`

**Interfaces:**
- Produces: `BCHCodec.decode_errors_and_erasures(decisions, max_assignments=64) -> DecodeResult`.
- `decisions` contains `0`, `1`, or `None` for each code position.

- [ ] Write failing tests for `(v,e)=(2,2)`, `(1,4)`, `(0,6)`, too many erasures, and diagnostics.
- [ ] Run `pytest tests/test_ecc.py -v` and verify failures.
- [ ] Implement bounded assignment enumeration, ordinary decode reuse, canonical re-encoding, radius filtering, and deduplication.
- [ ] Run `pytest tests/test_ecc.py -v` and verify passes.
- [ ] Commit error-erasure codec.

### Task 4: Detection policy and JSON result wiring

**Files:**
- Modify: `watermark/config.py`
- Modify: `watermark/detector.py`
- Modify: `watermark/result_types.py`
- Modify: `utils/detection.py`
- Modify: `utils/arguments.py`
- Modify: `run_detection.py`
- Test: `tests/test_detector.py`
- Test: `tests/test_detection_workflow.py`

**Interfaces:**
- `DetectionResult` contains `strict_decode`, `hard_fill_decode`, and `error_erasure_decode`.
- Supported primary policies are `strict`, `hard_fill`, and `error_erasure`.

- [ ] Write failing tests for the new policy, default selection, gating, and serialized diagnostics.
- [ ] Run focused tests and verify failures.
- [ ] Implement detector and CLI wiring.
- [ ] Run focused tests and verify passes.
- [ ] Commit detection integration.

### Task 5: Policy-specific metrics and report output

**Files:**
- Modify: `evaluation/metrics.py`
- Modify: `evaluation/report.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Each policy reports decode success, exact recovery, and conditional message BER.
- Existing headline metrics use the selected primary policy fields in each detection record.

- [ ] Write failing tests for strict, hard-fill, and error-erasure correctness/coverage separation.
- [ ] Run `pytest tests/test_metrics.py -v` and verify failures.
- [ ] Implement policy metrics and summary lines.
- [ ] Run focused tests and verify passes.
- [ ] Commit metrics changes.

### Task 6: Full verification and documentation

**Files:**
- Modify: `README.md`
- Modify: `scripts/smoke_test.sh`

- [ ] Update commands and output descriptions for exact 200-token triples and `error_erasure` default.
- [ ] Run `pytest -q`.
- [ ] Run `python run_generation.py --help` and `python run_detection.py --help`.
- [ ] Run a dependency-light fake-model generation/detection workflow.
- [ ] Review `git diff`, commit documentation, and record final verification output.
