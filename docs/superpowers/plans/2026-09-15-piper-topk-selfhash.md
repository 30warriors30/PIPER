# PIPER Top-50 Selfhash Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an MPAC-style selfhash encoder/detector mode that biases only the raw model top-50 candidates and samples with top-k followed by top-p.

**Architecture:** Configuration records generation top-k separately from the watermark candidate limit. The logits processor uses PIPER's history-only bit allocator but candidate-conditioned partition seeds and the v2 stateless partitioner; the detector reconstructs the same candidate-conditioned region from observed text without model logits. Legacy history-only v1 behavior remains the fallback for old artifacts.

**Tech Stack:** Python 3.10, PyTorch 2.8, pytest, dataclasses, existing keyed PRF and exact partition modules.

**Spec:** `docs/superpowers/specs/2026-09-15-piper-topk-selfhash-design.md`

## Global Constraints

- Candidate limit is exactly 50 for the paper experiment.
- Selfhash partition context is the previous `context_width - 1` tokens plus the candidate token.
- BCH bit allocation remains based on the previous `context_width` history tokens.
- Sampling order is watermark, temperature, EOS mask, top-k, top-p, sample.
- Existing artifacts missing new fields retain history-only v1 behavior.
- Detector remains model-free.

---

### Task 1: Add exact candidate-region lookup

**Files:**
- Modify: `watermark/partition_v2.py`
- Test: `tests/test_partition_v2.py`

**Interfaces:**
- Consumes: `partition_round_keys(seed)` and token IDs.
- Produces: `StatelessExactPartitioner.regions_for_tokens(*, round_keys_by_token, token_ids) -> torch.Tensor` with shape `[batch, candidates]` and `-1` for excluded tokens.

- [x] **Step 1: Write the failing test**

```python
def test_regions_for_tokens_matches_scalar_region_lookup():
    partitioner = StatelessExactPartitioner(12, {0})
    token_ids = torch.tensor([[1, 4, 9], [2, 5, 10]])
    keys = tuple(
        tuple(partition_round_keys(seed + token)) for token in row)
        for seed, row in zip((10, 20), token_ids.tolist(), strict=True)
    )
    got = partitioner.regions_for_tokens(
        round_keys_by_token=keys,
        token_ids=token_ids,
    )
    want = torch.tensor([
        [partitioner.region_for_token(round_keys=keys[r][c], token_id=int(token_ids[r, c])) for c in range(3)]
        for r in range(2)
    ])
    assert torch.equal(got.cpu(), want)
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_partition_v2.py::test_regions_for_tokens_matches_scalar_region_lookup -q`

Expected: FAIL because `regions_for_tokens` does not exist.

- [x] **Step 3: Implement vectorized token lookup**

Flatten `[batch, candidates]` token IDs and matching six-round keys, map eligible
token IDs to dense indices, apply the existing tensor Feistel permutation and
cycle walking, then reshape exact region IDs to the input shape. Preserve `-1`
for excluded or out-of-range IDs.

- [x] **Step 4: Run partition tests**

Run: `python -m pytest tests/test_partition_v2.py -q`

Expected: PASS.

### Task 2: Add raw-top-k selfhash watermark processing

**Files:**
- Modify: `watermark/logits_processor.py`
- Test: `tests/test_logits_processor.py`

**Interfaces:**
- Consumes: `seeding_scheme: Literal["history", "selfhash"]`, `candidate_top_k: int | None`, and raw score rows.
- Produces: candidate-only PIPER bias while retaining the existing `__call__(input_ids, scores)` API.

- [x] **Step 1: Write failing behavior tests**

Add tests demonstrating with literal logits that only the raw top-k tokens can
change, and that two candidates in the same row are seeded with distinct
`history[-(context_width-1):] + [candidate]` contexts. Add validation tests that
selfhash requires v2 and a positive candidate limit.

- [x] **Step 2: Run the focused tests and observe expected failures**

Run: `python -m pytest tests/test_logits_processor.py -q`

Expected: FAIL on the new constructor arguments/behavior.

- [x] **Step 3: Implement minimal selfhash processing**

Compute raw candidate IDs once with `torch.topk(scores, candidate_top_k)`. Keep
the BCH bit index history-only. Derive one candidate-conditioned partition seed
per raw candidate, obtain exact v2 region IDs using Task 1, and scatter presence
and payload bias only to qualifying candidates. In hard mode, mask candidates
outside the allowed upper support so lower candidates cannot be replaced by
tokens outside the raw candidate set.

- [x] **Step 4: Run logits-processor tests**

Run: `python -m pytest tests/test_logits_processor.py -q`

Expected: PASS.

### Task 3: Make detector reconstruct selfhash regions

**Files:**
- Modify: `watermark/detector.py`
- Modify: `utils/detection.py`
- Test: `tests/test_detector.py`
- Test: `tests/test_detection_workflow.py`

**Interfaces:**
- Consumes: saved `seeding_scheme` and `partition_engine` metadata.
- Produces: identical region and code-bit assignments for an observed generated token, without model logits.

- [x] **Step 1: Write failing detector parity tests**

Construct a selfhash processor and detector with the same key, history, and
codeword. For one observed candidate, assert that the detector event region and
code-bit index equal the processor trace. Assert unique-context suppression uses
the candidate-conditioned four-token context.

- [x] **Step 2: Run focused detector tests and observe failures**

Run: `python -m pytest tests/test_detector.py tests/test_detection_workflow.py -q`

Expected: FAIL because the detector lacks selfhash configuration.

- [x] **Step 3: Implement detector selfhash routing**

Select v1 or v2 partitioner from metadata. For selfhash, compute the partition
context from the previous `context_width - 1` tokens plus the observed token,
while allocating the BCH index from the previous `context_width` tokens. Treat
only regions 0 and 1 as upper hits.

- [x] **Step 4: Run detector tests**

Run: `python -m pytest tests/test_detector.py tests/test_detection_workflow.py -q`

Expected: PASS.

### Task 4: Add sampling top-k and experiment metadata

**Files:**
- Modify: `utils/batched_generation.py`
- Modify: `watermark/config.py`
- Modify: `utils/arguments.py`
- Modify: `utils/validation.py`
- Modify: `utils/generation.py`
- Modify: `experiments/config.py`
- Modify: `experiments/arguments.py`
- Modify: `experiments/generation.py`
- Test: `tests/test_batched_generation.py`
- Test: `tests/test_arguments.py`
- Test: `tests/experiments/test_generation_batch_cli.py`

**Interfaces:**
- Consumes: `top_k`, `candidate_top_k`, `seeding_scheme`, and `partition_engine`.
- Produces: reproducible JSON metadata and CLI flags; legacy missing fields retain old behavior.

- [x] **Step 1: Write failing sampling/config tests**

Add a literal-logit sampling test proving top-k is applied before top-p. Add CLI
tests for `--top-k 50 --candidate-top-k 50 --seeding-scheme selfhash
--partition-engine v2`. Add an old-dictionary loading test that expects
`top_k=None`, `candidate_top_k=None`, `seeding_scheme="history"`, and
`partition_engine="v1"`.

- [x] **Step 2: Run focused tests and observe failures**

Run: `python -m pytest tests/test_batched_generation.py tests/test_arguments.py tests/experiments/test_generation_batch_cli.py -q`

Expected: FAIL because fields and filters are absent.

- [x] **Step 3: Implement and thread configuration**

Add `_top_k_filter`, apply it before `_top_p_filter`, validate positive values,
and pass all new fields through both generation entry paths. CLI defaults for
new runs are selfhash/v2/50/50; dataclass defaults used by old JSON remain
history/v1/None/None.

- [x] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_batched_generation.py tests/test_arguments.py tests/experiments/test_generation_batch_cli.py -q`

Expected: PASS.

### Task 5: Regression verification and paper command

**Files:**
- Modify: `PAPER_EXPERIMENTS.md`

**Interfaces:**
- Consumes: completed implementation and CLI.
- Produces: exact 300-token BCH23/8 selfhash command for the server.

- [x] **Step 1: Run the complete unit test suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [x] **Step 2: Run static compilation**

Run: `python -m compileall watermark utils experiments tests`

Expected: exit code 0.

- [x] **Step 3: Document the exact experiment command**

Document an invocation containing `--exact-tokens 300`, `--context-width 4`,
`--top-k 50`, `--top-p 0.95`, `--candidate-top-k 50`,
`--seeding-scheme selfhash`, `--partition-engine v2`, BCH `(23,8,3)`, and the
existing batch settings.

- [x] **Step 4: Review the diff**

Run: `git diff --check` and `git diff --stat`.

Expected: no whitespace errors and only planned files changed.
