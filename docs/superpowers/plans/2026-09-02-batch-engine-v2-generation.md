# Batch Engine v2 Generation and Hybrid Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic fixed-membership batched generation, buffered persistence, and explicit import of completed v1 generation artifacts.

**Architecture:** A shared `BatchGenerator` left-pads prompts, derives one seed per immutable source-position batch, and invokes Transformers once for each baseline or watermarked batch. `MigrationStore` freezes legacy JSONL prefixes and records exact missing-key v2 batches. All PIPER generation runners become adapters that build requests and serialize shared results.

**Tech Stack:** Python 3.10, PyTorch, Transformers generation API, dataclasses, JSON/JSONL, hashlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-batch-engine-v2-design.md`

## Global Constraints

- Complete `2026-09-02-batch-engine-v2-core.md` first.
- Default generation batch size is `16`, with positive runtime overrides.
- Same v2 configuration, input order, software, and hardware must reproduce token IDs.
- Baseline and watermarked calls for one paired batch start from identical RNG state.
- Prompts use left padding and pad IDs never enter watermark context.
- Exact continuation length remains mandatory for every row.
- OOM does not trigger automatic batch-size fallback.
- Legacy completed records are not regenerated during explicit hybrid import.
- New records contain engine and batch provenance; old JSONL records are not rewritten.
- Run tests with the explicit environment-library path from the core plan.

---

### Task 1: Ordered Batch Primitives and Buffered JSONL Writer

**Files:**
- Create: `utils/batching.py`
- Modify: `utils/io.py`
- Test: `tests/test_batching.py`
- Test: `tests/test_io_resume.py`

**Interfaces:**
- Produces: `IndexedBatch[T](batch_id, source_start, items)`
- Produces: `indexed_batches(items, batch_size) -> Iterator[IndexedBatch[T]]`
- Produces: `SeededBatchItem` protocol with `sample_id` and `generation_seed` attributes.
- Produces: `derive_batch_seed(batch, engine_version, configured_batch_size) -> int`
- Produces: `left_pad_prompts(prompt_rows, pad_token_id, device) -> (input_ids, attention_mask)`
- Produces: `append_jsonl_batch(path, records) -> None`

- [ ] **Step 1: Write failing stable-membership and padding tests**

```python
def test_indexed_batches_have_immutable_source_positions():
    batches = list(indexed_batches(list("abcdef"), 4))
    assert [(b.batch_id, b.source_start, b.items) for b in batches] == [
        (0, 0, ("a", "b", "c", "d")),
        (1, 4, ("e", "f")),
    ]


def test_left_pad_prompts_preserves_final_context_tokens():
    ids, mask = left_pad_prompts([(5, 6), (7, 8, 9)], pad_token_id=0, device=torch.device("cpu"))
    assert ids.tolist() == [[0, 5, 6], [7, 8, 9]]
    assert mask.tolist() == [[0, 1, 1], [1, 1, 1]]
    assert ids[:, -2:].tolist() == [[5, 6], [8, 9]]
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_batching.py tests/test_io_resume.py -q
```

Expected: FAIL because the batch utilities do not exist.

- [ ] **Step 3: Implement immutable batches, seed derivation, and left padding**

Use this public data shape:

```python
@dataclass(frozen=True)
class IndexedBatch(Generic[T]):
    batch_id: int
    source_start: int
    items: tuple[T, ...]


class SeededBatchItem(Protocol):
    sample_id: str
    generation_seed: int


def derive_batch_seed(
    batch: IndexedBatch[SeededBatchItem],
    *,
    engine_version: int,
    configured_batch_size: int,
) -> int:
    material = {
        "engine_version": engine_version,
        "configured_batch_size": configured_batch_size,
        "batch_id": batch.batch_id,
        "sample_ids": [item.sample_id for item in batch.items],
        "generation_seeds": [item.generation_seed for item in batch.items],
    }
    digest = hashlib.blake2b(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")
```

Reject empty prompt rows and nonpositive batch sizes. Use `torch.full`, then right-align prompt IDs and masks.

- [ ] **Step 4: Implement one-flush ordered batch writes**

`append_jsonl_batch` serializes all records before opening the file, joins them with terminating newlines, writes once under one file handle, flushes once, and calls `os.fsync` once. Empty batches perform no write. Reuse `plain()` for dataclasses and paths.

- [ ] **Step 5: Test malformed-tail detection and batch writes**

```python
def test_append_jsonl_batch_preserves_order_and_complete_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    append_jsonl_batch(path, [{"id": 1}, {"id": 2}])
    assert path.read_text().endswith("\n")
    assert [row["id"] for row in iter_jsonl(path)] == [1, 2]


def test_iter_jsonl_rejects_truncated_final_record(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id":1}\n{"id":', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSONL"):
        list(iter_jsonl(path))
```

- [ ] **Step 6: Run tests and commit primitives**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_batching.py tests/test_io_resume.py -q
```

Expected: PASS.

Commit:

```bash
git add utils/batching.py utils/io.py tests/test_batching.py tests/test_io_resume.py
git commit -m "Add ordered batch and buffered JSONL primitives"
```

---

### Task 2: Hybrid Migration Store

**Files:**
- Create: `utils/engine_migration.py`
- Test: `tests/test_engine_migration.py`

**Interfaces:**
- Consumes: `BatchExecutionConfig`, `indexed_batches`, `read_json`, `write_json`, `iter_jsonl`.
- Produces: `MigrationStore.initialize(root, execution, resume, import_v1_completed)`
- Produces: `MigrationStore.plan_artifact(path, ordered_keys, key_fn) -> ArtifactMigrationPlan`
- Produces: `ArtifactMigrationPlan.frozen_keys`, `.pending_batches`, `.source_engine_version(key)`
- Produces: `MigrationStore.verify_legacy_prefixes() -> None`

- [ ] **Step 1: Write failing v1-prefix import tests**

```python
def test_import_freezes_v1_prefix_and_batches_only_missing_keys(tmp_path):
    output = tmp_path / "watermarked.jsonl"
    append_jsonl_batch(output, [{"sample_id": f"s{i}", "status": "completed"} for i in range(3)])
    store = MigrationStore.initialize(
        tmp_path,
        execution=BatchExecutionConfig(generation_batch_size=2),
        resume=True,
        import_v1_completed=True,
    )
    plan = store.plan_artifact(
        output,
        ordered_keys=[f"s{i}" for i in range(7)],
        key_fn=lambda row: str(row["sample_id"]),
    )
    assert plan.frozen_keys == frozenset({"s0", "s1", "s2"})
    assert [[item for item in batch.items] for batch in plan.pending_batches] == [
        ["s3", "s4"], ["s5", "s6"]
    ]
```

- [ ] **Step 2: Run the migration test and verify it fails**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_engine_migration.py -q
```

Expected: FAIL because `MigrationStore` does not exist.

- [ ] **Step 3: Implement atomic `engine_migration.json` creation**

The sidecar schema must contain:

```json
{
  "schema_version": 1,
  "mode": "hybrid-v1-v2",
  "execution": {
    "engine_version": 2,
    "generation_batch_size": 16,
    "detection_batch_size": 64,
    "detection_workers": 8
  },
  "artifacts": {}
}
```

Require `resume=True` with `import_v1_completed=True`. Reject the import flag when the sidecar already exists. On later resume, require no import flag and exact execution equality.

- [ ] **Step 4: Implement artifact prefix freezing**

For each existing artifact record:

```text
relative_path
legacy_byte_length
legacy_sha256
legacy_record_count
frozen_key_sha256
frozen_keys
ordered_missing_keys
v2_batches: [{batch_id, keys}]
```

Read every line through `iter_jsonl`, reject duplicate completion keys, hash exactly the first `legacy_byte_length` bytes, and atomically update the sidecar. A new artifact with no legacy records has an empty frozen set and normal v2 batches.

- [ ] **Step 5: Add changed-prefix and repeated-import tests**

```python
def test_resume_rejects_changed_legacy_prefix(tmp_path):
    store, output = create_imported_store(tmp_path)
    data = output.read_bytes()
    output.write_bytes(data.replace(b'"s0"', b'"xx"', 1))
    resumed = MigrationStore.initialize(
        tmp_path,
        execution=BatchExecutionConfig(generation_batch_size=2),
        resume=True,
        import_v1_completed=False,
    )
    with pytest.raises(ValueError, match="legacy prefix digest mismatch"):
        resumed.verify_legacy_prefixes()


def test_repeated_import_flag_is_rejected(tmp_path):
    create_imported_store(tmp_path)
    with pytest.raises(ValueError, match="already initialized"):
        MigrationStore.initialize(
            tmp_path,
            execution=BatchExecutionConfig(),
            resume=True,
            import_v1_completed=True,
        )
```

- [ ] **Step 6: Run tests and commit migration support**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_engine_migration.py tests/test_io_resume.py -q
```

Expected: PASS.

Commit:

```bash
git add utils/engine_migration.py tests/test_engine_migration.py
git commit -m "Add explicit v1 to v2 migration store"
```

---

### Task 3: Shared Batched Generation Engine

**Files:**
- Create: `utils/batch_generation.py`
- Test: `tests/test_batch_generation.py`

**Interfaces:**
- Consumes: `IndexedBatch`, `derive_batch_seed`, `left_pad_prompts`, `DualLayerLogitsProcessor` v2.
- Produces: `GenerationRequest`
- Produces: `GenerationResult`
- Produces: `BatchGenerator.generate(batch, *, watermarked) -> tuple[GenerationResult, ...]`
- Produces: `PerRowLogitsProcessorAdapter`

- [ ] **Step 1: Write failing fake-model batch tests**

```python
def test_batch_generator_left_pads_and_returns_exact_rows():
    requests = (
        GenerationRequest("a", 0, (5, 6), 11, 3, None),
        GenerationRequest("b", 1, (7, 8, 9), 12, 3, None),
    )
    generator = BatchGenerator(fake_model, fake_tokenizer, torch.device("cpu"), execution)
    results = generator.generate(IndexedBatch(0, 0, requests), watermarked=False)
    assert [row.sample_id for row in results] == ["a", "b"]
    assert all(len(row.token_ids) == 3 for row in results)
    assert fake_model.last_input_ids.tolist() == [[0, 5, 6], [7, 8, 9]]


def test_same_v2_batch_repeats_token_ids():
    first = seeded_generator.generate(batch, watermarked=False)
    second = seeded_generator.generate(batch, watermarked=False)
    assert [row.token_ids for row in first] == [row.token_ids for row in second]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_batch_generation.py -q
```

Expected: FAIL because the shared engine does not exist.

- [ ] **Step 3: Implement request/result dataclasses and exact generation**

Use these fields:

```python
@dataclass(frozen=True)
class GenerationRequest:
    sample_id: str
    source_index: int
    prompt_token_ids: tuple[int, ...]
    generation_seed: int
    exact_tokens: int
    encoded_bits: tuple[int, ...] | None


@dataclass(frozen=True)
class GenerationResult:
    sample_id: str
    token_ids: tuple[int, ...]
    text: str
    engine_version: int
    generation_batch_id: int
    generation_batch_size_actual: int
    generation_batch_seconds: float
    amortized_generation_seconds: float
```

Reject mixed exact lengths and mixed watermarked/unwatermarked state in one batch. Construct one v2 logits processor from all encoded codewords for a watermarked call. Pass attention mask, exact `min_new_tokens`/`max_new_tokens`, sampling temperature/top-p, pad/eos IDs, and `use_cache=True` to `model.generate`.

- [ ] **Step 4: Implement paired RNG and OOM diagnostics**

Compute `batch_seed = derive_batch_seed(batch, engine_version=2, configured_batch_size=self.execution.generation_batch_size)` and enter `paired_rng(batch_seed, self.device)` around each model call. Catch only `torch.cuda.OutOfMemoryError` to raise:

```text
CUDA OOM for generation batch configured=16 actual=16; rerun from a new output directory with --generation-batch-size 8
```

Do not retry or mutate configuration.

- [ ] **Step 5: Implement external per-row processor adapter**

`PerRowLogitsProcessorAdapter(processors)` validates one processor per score row, calls each with `input_ids[row:row+1]` and `scores[row:row+1]`, and concatenates outputs in row order. This is the common bridge for MPAC and Segment/RS-BH tasks in the rollout plan.

- [ ] **Step 6: Run fake and local-model exact-length tests**

Run:

```bash
/usr/bin/env LD_LIBRARY_PATH=/home/yanlu/miniconda3/envs/dual-watermark-blackwell/lib LOCAL_OPT_MODEL=/data/yanlu/BREW/models/facebook/opt-1.3b /home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_batch_generation.py tests/test_integration_local_model.py -q -s
```

Expected: PASS for batch sizes 1, 2, and a local batch smoke.

- [ ] **Step 7: Commit the shared engine**

```bash
git add utils/batch_generation.py tests/test_batch_generation.py tests/test_integration_local_model.py
git commit -m "Add shared batched generation engine"
```

---

### Task 4: Generic Paired Generation Migration

**Files:**
- Modify: `utils/generation.py`
- Modify: `run_generation.py`
- Modify: `utils/io.py`
- Create: `tests/fixtures/generation_manifest_rows.jsonl`
- Test: `tests/test_generation.py`
- Test: `tests/test_io_resume.py`

**Interfaces:**
- Consumes: `BatchGenerator`, `MigrationStore`, `append_jsonl_batch`.
- Produces: `run_generation(config: ExperimentConfig, *, resume: bool = False, overwrite: bool = False, import_v1_completed: bool = False, model: Any | None = None, tokenizer: Any | None = None, device: torch.device | None = None) -> dict[str, Any]`.
- Preserves: one output record per sample and all existing three-class fields.

- [ ] **Step 1: Write failing paired-batch output tests**

Create four prepared samples and a fake batch generator. Assert two model batches at size 2, four records in source order, exact watermarked/unwatermarked/natural lengths, and v2 provenance on every new record.

- [ ] **Step 2: Run generic generation tests and verify they fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_generation.py tests/test_io_resume.py -q
```

Expected: FAIL because generic generation remains sample-at-a-time.

- [ ] **Step 3: Refactor `PairedGenerator` onto `BatchGenerator`**

Replace `generate(sample, payload)` with `generate_batch(indexed_batch, samples, payloads)`. Build baseline and watermarked request batches with identical membership and batch ID. Generate both before serializing. Keep per-sample ECC timing and message/codeword values.

- [ ] **Step 4: Persist an immutable valid-sample generation manifest**

Before loading the model for generation, create `generation_manifest.jsonl` containing exactly the target number of prepared valid samples:

```json
{
  "schema_version": 1,
  "manifest_index": 0,
  "sample_id": "sample-0",
  "dataset": "c4",
  "prompt": "prompt text",
  "prompt_token_ids": [2, 10, 11],
  "natural_text": "natural continuation",
  "natural_token_ids": [20, 21],
  "message_bits": "01010101",
  "encoded_bits": "01010101010101010101010",
  "generation_seed": 42,
  "metadata": {}
}
```

On pure-v2 resume, validate and reload this manifest instead of restreaming the dataset. For explicit v1 import where no manifest exists, stream and prepare source data once, match frozen sample IDs, and persist the ordered missing-only migration workload in `engine_migration.json` before generation.

- [ ] **Step 5: Refactor `run_generation` preparation and writes**

Accumulate valid prepared samples until one immutable batch is full, without counting filtered candidates as source positions. For hybrid import, use the migration plan's recorded missing-key batches. Serialize the existing nested `watermarked`, `unwatermarked`, and `natural` objects plus v2 provenance, then call `append_jsonl_batch` once.

Pass `args.import_v1_completed` from `run_generation.py`. Update `generation_summary` to print all three execution values and engine mode.

- [ ] **Step 6: Add pure-v2 manifest resume and partial-v1 import tests**

Interrupt a four-row pure-v2 fixture after its first two-row batch, resume, and assert the persisted manifest produces only the original second batch with the original batch ID and seed.

Seed `samples.jsonl` with two valid v1 records from an ordered four-sample fixture, run with `--resume --import-v1-completed --generation-batch-size 2`, and assert the fake model receives only the two missing samples. Assert the original bytes are an unchanged prefix and the sidecar marks `hybrid-v1-v2`.

- [ ] **Step 7: Run tests and commit generic migration**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/test_generation.py tests/test_io_resume.py tests/test_engine_migration.py -q
```

Expected: PASS.

Commit:

```bash
git add run_generation.py utils/generation.py utils/io.py tests/fixtures/generation_manifest_rows.jsonl tests/test_generation.py tests/test_io_resume.py
git commit -m "Batch generic paired generation"
```

---

### Task 5: Pareto Generation Migration

**Files:**
- Modify: `experiments/generation.py`
- Modify: `experiments/run_pareto_sweep.py`
- Modify: `experiments/manifest.py`
- Test: `tests/experiments/test_manifest.py`
- Create: `tests/experiments/test_generation_batches.py`

**Interfaces:**
- Consumes: shared `BatchGenerator`, `MigrationStore`, and Pareto execution fields.
- Produces: batched `run_shared_baseline` and `run_operating_point_generation`.

- [ ] **Step 1: Write failing Pareto baseline/point batch tests**

Use a six-row manifest, batch size 4, and fake shared engine. Assert baseline calls have actual sizes 4 and 2; each operating point constructs per-row encoded bits and writes six ordered watermarked records.

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_generation_batches.py tests/experiments/test_manifest.py -q
```

Expected: FAIL because Pareto generation loops per row.

- [ ] **Step 3: Add source indices to the Pareto manifest**

Persist `manifest_index` as the immutable generation source index and preserve it when loading. Reject duplicate indices. This index, not current completion count, determines pure-v2 batch membership.

- [ ] **Step 4: Replace Pareto row loops with shared requests**

`ExperimentGenerator` builds `GenerationRequest` values. Shared baseline batches have `encoded_bits=None`. Operating-point batches obtain codewords from manifest rows, share one `DualLayerLogitsProcessor`, and write batch provenance plus amortized generation time.

- [ ] **Step 5: Thread import and execution settings through the runner**

Pass `args.import_v1_completed` into baseline and sweep stage functions. Initialize/verify `MigrationStore` before model loading. Print engine/batch settings in `print_plan`. For pure-v2 runs, include the execution block in `experiment.json`; for hybrid runs, preserve legacy `experiment.json` and use the sidecar. Write the existing `environment_snapshot()` to `environment.json` before generation.

- [ ] **Step 6: Run tests and commit Pareto generation**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_generation_batches.py tests/experiments/test_manifest.py tests/experiments/test_z_calibration_routing.py -q
```

Expected: PASS.

Commit:

```bash
git add experiments/generation.py experiments/run_pareto_sweep.py experiments/manifest.py tests/experiments/test_generation_batches.py tests/experiments/test_manifest.py
git commit -m "Batch Pareto experiment generation"
```

---

### Task 6: Length and Capacity Generation Migration

**Files:**
- Modify: `experiments/run_length_sweep.py`
- Modify: `experiments/run_capacity_fpr.py`
- Test: `tests/experiments/test_length_sweep.py`
- Create: `tests/experiments/test_capacity_fpr.py`

**Interfaces:**
- Consumes: `BatchGenerator`, immutable manifest indices, `MigrationStore`.
- Produces: batch adapters for baseline and watermarked length/capacity points.
- Produces: hybrid import support for `outputs/experiments/piper_length_b8_m2`.

- [ ] **Step 1: Write failing length/capacity adapter tests**

For length, use five manifest rows and batch size 2. Assert each T point receives batches `2/2/1`, its exact token value, and its own codewords. For capacity, assert each b point receives the correct BCH codeword length while preserving the same batch membership.

- [ ] **Step 2: Run focused tests and verify they fail**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_length_sweep.py tests/experiments/test_capacity_fpr.py -q
```

Expected: FAIL because both scripts call `_generate_exact` per sample.

- [ ] **Step 3: Replace duplicated `_generate_exact` paths**

Delete length/capacity local model-generate helpers. Build shared requests from `manifest_index`, exact T, message seed, and each point's BCH codec. Keep current output paths and record fields; add v2 provenance.

- [ ] **Step 4: Add explicit hybrid root preparation**

Both `_prepare_root` implementations accept `import_v1_completed`. A legacy `experiment.json` may differ only by missing v2 execution fields when explicit import is requested. Write `engine_migration.json`, leave `experiment.json` legacy content intact, and write current execution settings in the sidecar. Pure-v2 roots write execution settings into `experiment.json` and both paths write `environment.json` before generation.

- [ ] **Step 5: Add the stopped-length fixture test**

Build a temp fixture with:

```text
T100 baseline=5, negatives=10, watermarked=5, detections=5
T200 baseline=5, negatives=10, watermarked=3, detections=0
```

Run import with batch size 2. Assert T100 generates nothing, T200 watermarked receives only missing IDs 3 and 4, T200 baseline is untouched, and later T points receive all five rows.

- [ ] **Step 6: Run generation migration suites**

Run:

```bash
/home/yanlu/miniconda3/envs/dual-watermark-blackwell/bin/python -m pytest tests/experiments/test_length_sweep.py tests/experiments/test_capacity_fpr.py tests/test_engine_migration.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit length/capacity generation**

```bash
git add experiments/run_length_sweep.py experiments/run_capacity_fpr.py tests/experiments/test_length_sweep.py tests/experiments/test_capacity_fpr.py
git commit -m "Batch length and capacity generation"
```
