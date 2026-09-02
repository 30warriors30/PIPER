# Batch Engine v2 Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared execution configuration, exact stateless v2 partition, and row-isolated batched PIPER logits processor.

**Architecture:** Runtime batch settings live in one immutable configuration used by generic and experiment-specific runners. A cycle-walked six-round Feistel PRP provides scalar and tensor region lookup without materializing NumPy permutations. The logits processor consumes one codeword per row and applies masks from the tensor partition path.

**Tech Stack:** Python 3.10, dataclasses, argparse, PyTorch, NumPy, Transformers `LogitsProcessor`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-batch-engine-v2-design.md`

## Global Constraints

- Engine version is exactly `2`.
- Defaults are generation batch `16`, detection batch `64`, and detection workers `8`.
- Batch and worker values must be positive and must be serialized in experiment metadata.
- `paper_shared` continues to share the partition/allocator base seed; `domain_separated` keeps separate base seeds.
- The v2 partition must be an exact bijection over eligible vocabulary indices.
- Existing `ExactPermutationPartitioner` remains available for legacy v1 detection.
- No Batch Engine v1/v2 text or speed comparison is added.
- Run tests with `/usr/bin/env LD_LIBRARY_PATH=/home/yanlu/miniconda3/envs/dual-watermark-blackwell/lib /home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest` until the Conda environment variable is repaired.

---

### Task 1: Shared Execution Configuration and CLI Arguments

**Files:**
- Create: `watermark/execution.py`
- Modify: `watermark/config.py`
- Modify: `utils/arguments.py`
- Modify: `experiments/config.py`
- Modify: `experiments/arguments.py`
- Modify: `utils/validation.py`
- Test: `tests/test_arguments.py`
- Test: `tests/test_validation.py`
- Test: `tests/experiments/test_z_calibration_routing.py`

**Interfaces:**
- Produces: `ENGINE_VERSION: int = 2`
- Produces: `BatchExecutionConfig(generation_batch_size=16, detection_batch_size=64, detection_workers=8)`
- Produces: `add_batch_execution_arguments(parser, *, include_generation, include_detection, include_import)`
- Produces: `batch_execution_config_from_args(args) -> BatchExecutionConfig`
- Consumed by: all later core, generation, detection, and rollout tasks.

- [ ] **Step 1: Write failing configuration and parser tests**

```python
def test_batch_execution_defaults_and_validation():
    value = BatchExecutionConfig()
    assert value.engine_version == 2
    assert value.generation_batch_size == 16
    assert value.detection_batch_size == 64
    assert value.detection_workers == 8
    with pytest.raises(ValueError, match="generation_batch_size must be positive"):
        BatchExecutionConfig(generation_batch_size=0)


def test_generation_parser_exposes_batch_defaults():
    args = generation_parser().parse_args(
        ["--model-path", "model", "--secret-key", "key", "--run-id", "run"]
    )
    assert args.generation_batch_size == 16
    assert args.detection_batch_size == 64
    assert args.detection_workers == 8
    assert args.import_v1_completed is False
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
/usr/bin/env LD_LIBRARY_PATH=/home/yanlu/miniconda3/envs/dual-watermark-blackwell/lib /home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_arguments.py tests/test_validation.py -q
```

Expected: FAIL because `BatchExecutionConfig` and the parser fields do not exist.

- [ ] **Step 3: Implement the immutable configuration**

Create `watermark/execution.py` with this public shape:

```python
from dataclasses import dataclass

ENGINE_VERSION = 2


@dataclass(frozen=True)
class BatchExecutionConfig:
    generation_batch_size: int = 16
    detection_batch_size: int = 64
    detection_workers: int = 8
    engine_version: int = ENGINE_VERSION

    def __post_init__(self) -> None:
        for name in ("generation_batch_size", "detection_batch_size", "detection_workers"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.engine_version) != ENGINE_VERSION:
            raise ValueError(f"engine_version must be {ENGINE_VERSION}")
```

Add an `execution` field with `default_factory=BatchExecutionConfig` to `ExperimentConfig`, read absent legacy values as v1 only in migration code, and serialize v2 values through `asdict`. Add the three numeric fields to `ParetoExperimentConfig` and validate them by constructing `BatchExecutionConfig`.

- [ ] **Step 4: Add shared argparse helpers and wire both parser families**

Implement:

```python
def add_batch_execution_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_generation: bool = True,
    include_detection: bool = True,
    include_import: bool = True,
) -> None:
    if include_generation:
        parser.add_argument("--generation-batch-size", type=int, default=16)
    if include_detection:
        parser.add_argument("--detection-batch-size", type=int, default=64)
        parser.add_argument("--detection-workers", type=int, default=8)
    if include_import:
        parser.add_argument("--import-v1-completed", action="store_true")


def batch_execution_config_from_args(args: argparse.Namespace) -> BatchExecutionConfig:
    return BatchExecutionConfig(
        generation_batch_size=int(getattr(args, "generation_batch_size", 16)),
        detection_batch_size=int(getattr(args, "detection_batch_size", 64)),
        detection_workers=int(getattr(args, "detection_workers", 8)),
    )
```

Call the helper from generic generation/detection parsers and Pareto parser. Pass the resulting values into both configuration types. Reject `--import-v1-completed` unless `--resume` is also true.

- [ ] **Step 5: Run parser/configuration tests**

Run:

```bash
/usr/bin/env LD_LIBRARY_PATH=/home/yanlu/miniconda3/envs/dual-watermark-blackwell/lib /home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_arguments.py tests/test_validation.py tests/experiments/test_z_calibration_routing.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the execution configuration**

```bash
git add watermark/execution.py watermark/config.py utils/arguments.py utils/validation.py experiments/config.py experiments/arguments.py tests/test_arguments.py tests/test_validation.py tests/experiments/test_z_calibration_routing.py
git commit -m "Add batch engine execution configuration"
```

---

### Task 2: Exact Stateless Feistel Partition

**Files:**
- Create: `watermark/partition_v2.py`
- Modify: `watermark/prf.py`
- Test: `tests/test_partition_v2.py`
- Test: `tests/test_prf.py`

**Interfaces:**
- Consumes: partition and allocator base seeds resolved with existing `KeyedPRF` semantics.
- Produces: `partition_round_keys(base_seed: int) -> tuple[int, ...]`
- Produces: `StatelessExactPartitioner.region_for_token(*, round_keys: Sequence[int], token_id: int) -> int | None`
- Produces: `StatelessExactPartitioner.regions_for_vocab(*, round_keys_by_row: Sequence[Sequence[int]], device: torch.device) -> torch.Tensor`
- Produces: `StatelessExactPartitioner.partition_for_context(*, round_keys: Sequence[int]) -> Partition`

- [ ] **Step 1: Write failing scalar exactness tests**

```python
@pytest.mark.parametrize("vocab_size,excluded", [(12, {0}), (17, {0, 1}), (5, set())])
def test_stateless_partition_is_an_exact_bijection(vocab_size, excluded):
    partitioner = StatelessExactPartitioner(vocab_size, excluded)
    keys = partition_round_keys(123456789)
    ranks = [
        partitioner.rank_for_token(round_keys=keys, token_id=token)
        for token in range(vocab_size)
        if token not in excluded
    ]
    assert sorted(ranks) == list(range(vocab_size - len(excluded)))


def test_region_cardinalities_follow_existing_quarter_rule():
    partitioner = StatelessExactPartitioner(13, {0})
    values = partitioner.partition_for_context(round_keys=partition_round_keys(7))
    assert tuple(map(len, (values.bit0, values.bit1, values.lower_a, values.lower_b))) == (3, 3, 3, 3)
```

- [ ] **Step 2: Run scalar tests and verify they fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_partition_v2.py -q
```

Expected: FAIL because `watermark.partition_v2` does not exist.

- [ ] **Step 3: Implement dense eligible-index mapping and six-round PRP**

Implement these constants and functions in `watermark/partition_v2.py`:

```python
MASK64 = (1 << 64) - 1
ROUND_DOMAINS = (
    0x243F6A8885A308D3,
    0x13198A2E03707344,
    0xA4093822299F31D0,
    0x082EFA98EC4E6C89,
    0x452821E638D01377,
    0xBE5466CF34E90C6C,
)


def _avalanche64(value: int) -> int:
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & MASK64
    return value ^ (value >> 31)


def partition_round_keys(base_seed: int) -> tuple[int, ...]:
    return tuple(_avalanche64((int(base_seed) ^ domain) & MASK64) for domain in ROUND_DOMAINS)
```

Choose the smallest even domain width containing the eligible count, use equal half widths, run six Feistel rounds, and cycle-walk until the rank is below the eligible count. Cache token-to-dense and dense-to-token mappings by `(vocab_size, excluded_ids)`.

- [ ] **Step 4: Add tensor/scalar golden-vector tests**

```python
def test_tensor_regions_match_scalar_regions_on_cpu():
    partitioner = StatelessExactPartitioner(31, {0, 3})
    keys = [partition_round_keys(11), partition_round_keys(29)]
    tensor = partitioner.regions_for_vocab(round_keys_by_row=keys, device=torch.device("cpu"))
    for row, row_keys in enumerate(keys):
        expected = [
            partitioner.region_for_token(round_keys=row_keys, token_id=token)
            for token in range(31)
        ]
        assert tensor[row].tolist() == [-1 if value is None else value for value in expected]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_regions_match_cpu_regions():
    partitioner = StatelessExactPartitioner(50272, {0, 1})
    keys = [partition_round_keys(42), partition_round_keys(99)]
    cpu = partitioner.regions_for_vocab(round_keys_by_row=keys, device=torch.device("cpu"))
    cuda = partitioner.regions_for_vocab(round_keys_by_row=keys, device=torch.device("cuda"))
    assert torch.equal(cpu, cuda.cpu())
```

- [ ] **Step 5: Implement the tensor ARX/Feistel path**

Keep the same six rounds and cycle-walking condition as the scalar code. Operate on `torch.int64`; explicitly mask half-domain values after every round. Preserve `-1` for excluded IDs. Cache eligible ID tensors per device without caching context-dependent regions.

- [ ] **Step 6: Run partition and PRF tests**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_partition.py tests/test_partition_v2.py tests/test_prf.py -q
```

Expected: PASS, including legacy partition tests.

- [ ] **Step 7: Commit the stateless partition**

```bash
git add watermark/partition_v2.py watermark/prf.py tests/test_partition_v2.py tests/test_prf.py
git commit -m "Add exact stateless v2 partition"
```

---

### Task 3: Row-Isolated Batched Logits Processor

**Files:**
- Modify: `watermark/logits_processor.py`
- Test: `tests/test_logits_processor.py`

**Interfaces:**
- Consumes: `StatelessExactPartitioner` and `partition_round_keys` from Task 2.
- Produces: `DualLayerLogitsProcessor` constructor support for `encoded_bits_by_row: Sequence[Sequence[int]] | None`, `partition_engine: Literal["v1", "v2"] = "v1"`, and `capture_traces: bool = True`.
- Produces: `last_traces: tuple[EmbeddingTrace, ...]`
- Preserves: legacy `encoded_bits=` and `last_trace` for a one-row call.

- [ ] **Step 1: Write failing batch isolation tests**

```python
def test_batch_rows_use_independent_codewords_and_contexts():
    processor = DualLayerLogitsProcessor(
        secret_key=b"secret",
        context_width=2,
        encoded_bits_by_row=((0, 1, 0, 1), (1, 0, 1, 0)),
        vocab_size=12,
        excluded_token_ids={0},
        presence_mode="soft",
        delta_presence=2.0,
        delta_payload=3.0,
        prf_mode="paper_shared",
        partition_engine="v2",
        capture_traces=True,
    )
    output = processor(torch.tensor([[2, 3], [8, 9]]), torch.zeros((2, 12)))
    assert output.shape == (2, 12)
    assert len(processor.last_traces) == 2
    assert processor.last_traces[0].context_ids == (2, 3)
    assert processor.last_traces[1].context_ids == (8, 9)
    for row, trace in enumerate(processor.last_traces):
        assert torch.all(output[row, list(trace.target_ids)] == 5.0)


def test_batch_size_must_match_codeword_rows():
    processor = make_v2_processor(encoded_bits_by_row=((0, 1),))
    with pytest.raises(ValueError, match="2 score rows but 1 encoded codeword"):
        processor(torch.tensor([[2, 3], [4, 5]]), torch.zeros((2, 12)))
```

- [ ] **Step 2: Run the processor tests and verify they fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_logits_processor.py -q
```

Expected: FAIL because the processor rejects batches and lacks v2 arguments.

- [ ] **Step 3: Normalize one-row and multi-row constructor state**

Accept exactly one of `encoded_bits` and `encoded_bits_by_row`. Normalize to:

```python
self.encoded_bits_by_row: tuple[tuple[int, ...], ...]
self.partition_engine: Literal["v1", "v2"]
self.capture_traces: bool
```

Require every row to have the same nonzero codeword length and binary values. Permit legacy `encoded_bits` only for a one-row call. Keep the existing v1 implementation behind `partition_engine="v1"`.

- [ ] **Step 4: Implement the v2 batched call**

Use one `input_ids[:, -context_width:].detach().cpu().tolist()` transfer. Resolve base seeds per row with existing `paper_shared`/`domain_separated` behavior, expand partition round keys, obtain `[batch, vocab]` regions, allocate each row's code-bit index, gather its embedded bit, and apply masks:

```python
upper_mask = (regions == 0) | (regions == 1)
target_mask = regions == embedded_bits[:, None]
if self.presence_mode == "hard":
    output.masked_fill_(~upper_mask, -torch.inf)
else:
    output.add_(upper_mask.to(output.dtype) * self.delta_presence)
output.add_(target_mask.to(output.dtype) * self.delta_payload)
```

Always retain excluded-token behavior from v1. Build traces only when `capture_traces=True`; otherwise avoid copying region tensors back to CPU.

- [ ] **Step 5: Add backward-compatibility and paper-shared tests**

```python
def test_legacy_single_row_api_remains_available():
    processor = make_processor("soft")
    output = processor(torch.tensor([[2, 3]]), torch.zeros((1, 12)))
    assert processor.last_trace is not None
    assert output.shape == (1, 12)


def test_paper_shared_uses_allocator_base_seed_before_round_expansion():
    processor = make_v2_processor(encoded_bits_by_row=((0, 1, 0, 1),), capture_traces=True)
    processor(torch.tensor([[2, 3]]), torch.zeros((1, 12)))
    trace = processor.last_trace
    assert trace is not None
    assert trace.partition_seed == trace.position_seed
```

- [ ] **Step 6: Run all core watermark tests**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_prf.py tests/test_partition.py tests/test_partition_v2.py tests/test_allocator.py tests/test_logits_processor.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the batched processor**

```bash
git add watermark/logits_processor.py tests/test_logits_processor.py
git commit -m "Support row-isolated batched watermark logits"
```

---

### Task 4: Core Integration Verification

**Files:**
- Modify: `tests/test_paper_conformance.py`
- Modify: `tests/test_integration_local_model.py`

**Interfaces:**
- Consumes: all Task 1-3 interfaces.
- Produces: a stable core contract for generation and detection implementation plans.

- [ ] **Step 1: Add paper-semantics assertions for v2**

```python
def test_v2_partition_keeps_exact_upper_probability():
    partitioner = StatelessExactPartitioner(50272, {0, 1})
    partition = partitioner.partition_for_context(round_keys=partition_round_keys(42))
    assert len(partition.bit0) == len(partition.bit1)
    assert len(partition.bit0) + len(partition.bit1) == 2 * ((50272 - 2) // 4)
```

- [ ] **Step 2: Add a local-model batched logits smoke test**

Add this test beside the existing local-model tests and reuse their model/tokenizer fixture:

```python
def test_local_opt_accepts_two_row_v2_logits_processor(local_model_bundle):
    model, tokenizer, device = local_model_bundle
    prompts = tokenizer(
        ["Batch watermark prompt one", "Batch watermark prompt two"],
        return_tensors="pt",
        padding=True,
    ).to(device)
    processor = make_local_v2_processor(
        model,
        tokenizer,
        encoded_bits_by_row=((0, 1) * 12, (1, 0) * 12),
        capture_traces=True,
    )
    with torch.inference_mode():
        scores = model(**prompts).logits[:, -1, :]
        biased = processor(prompts["input_ids"], scores)
    assert biased.shape == scores.shape
    assert torch.isfinite(biased).any(dim=1).all()
    assert len(processor.last_traces) == 2
    assert processor.last_traces[0].context_ids != processor.last_traces[1].context_ids
```

Gate the fixture with `LOCAL_OPT_MODEL` exactly as the existing integration module does.

- [ ] **Step 3: Run the core and local smoke suites**

Run:

```bash
/usr/bin/env LD_LIBRARY_PATH=/home/yanlu/miniconda3/envs/dual-watermark-blackwell/lib LOCAL_OPT_MODEL=/data/yanlu/BREW/models/facebook/opt-1.3b /home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_paper_conformance.py tests/test_integration_local_model.py -q -s
```

Expected: PASS.

- [ ] **Step 4: Commit core integration coverage**

```bash
git add tests/test_paper_conformance.py tests/test_integration_local_model.py
git commit -m "Verify batch engine v2 core semantics"
```
