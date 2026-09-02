# Batch Engine v2 Design

## Status

Approved for implementation on 2026-09-02.

## Context

PIPER currently invokes `model.generate()` once per sample and the watermark
logits processor accepts only `batch_size=1`. At each generated token, the
processor copies the context from CUDA to Python, constructs a full NumPy
permutation of the eligible vocabulary, and copies large index lists back to
CUDA. Detection repeats the full-vocabulary permutation for every observed
token. The official experiment runners also contain separate generation and
detection loops, so performance fixes are easy to apply inconsistently.

On the target OPT-1.3B and RTX PRO 6000 Blackwell system, this design leaves
the GPU near 20 percent utilization even during continuous generation.

## Goals

1. Support batched generation and detection in every in-repository experiment
   that generates or detects PIPER, MPAC, or Segment/RS-BH text.
2. Preserve exact continuation lengths, per-sample payload isolation, detector
   decisions, output ordering, and resumability.
3. Remove the per-token `O(vocabulary size)` detector operation while retaining
   an exact four-way partition of the eligible vocabulary.
4. Expose runtime batch controls with documented defaults:
   - `--generation-batch-size 16`
   - `--detection-batch-size 64`
   - `--detection-workers 8`
5. Make one Batch Engine v2 configuration reproducible. Bit-for-bit identity
   with legacy batch-one output, or across different batch sizes, is not a
   requirement.

## Non-Goals

- Batch Engine v1 and v2 output are not compared for text identity or speed.
- Existing v1 experiment directories are not resumed with v2.
- The engine does not silently change batch size after an out-of-memory error.
- This work does not alter BCH parameters, presence tests, counting rules,
  payload policies, quality definitions, or paper headline settings.

## Runtime Configuration

Introduce an immutable `BatchExecutionConfig` shared by all runners:

```text
engine_version = 2
generation_batch_size = 16
detection_batch_size = 64
detection_workers = 8
```

All three numeric values must be positive. Common CLI helpers add the same
options and defaults to every relevant command. Experiment metadata records all
four fields. Resume validation rejects a missing or different engine version,
generation batch size, detection batch size, or worker count. This deliberately
prevents a directory from containing mixed sampling streams.

An OOM exception identifies the attempted batch size and recommends rerunning
from a new output directory with a smaller explicit value. The engine never
changes the value automatically because doing so would change the sampling
stream without changing recorded arguments.

## Exact Stateless Partition

### Required Semantics

For each secret-key/context pair, every eligible vocabulary token maps to one
unique rank in `[0, eligible_vocab_size)`. Consecutive rank ranges define
`bit0`, `bit1`, `lower_a`, and `lower_b` with the same quotient-and-remainder
rule used by the existing exact partitioner. Therefore regions are disjoint,
cover all eligible tokens, and retain exact cardinalities.

### PRP Construction

Batch Engine v2 replaces materialized PCG64 permutations with a cycle-walked,
six-round balanced Feistel permutation:

1. Map eligible token IDs to dense eligible indices.
2. Choose the smallest even bit width whose power-of-two domain contains all
   eligible indices.
3. Derive six 64-bit round keys from the existing keyed context PRF with
   domain labels `partition-v2-round-0` through `partition-v2-round-5`.
4. Use a specified 64-bit ARX avalanche function for each Feistel round and
   mask its result to the half-domain width.
5. Cycle-walk outputs outside the eligible-index domain until the result is in
   range.

The scalar and tensor implementations share golden vectors. The scalar path
classifies one observed token in expected constant time. The tensor path
classifies all eligible tokens for a generation row in parallel on the scores'
device. This construction is an exact bijection, independent of batch shape,
and does not allocate or sort a random vocabulary permutation.

The partition implementation exposes only these operations:

```text
region_for_token(context, token_id) -> region | None
regions_for_vocab(contexts, device) -> [batch, vocab] region tensor
partition_for_context(context) -> Partition  # diagnostic/test compatibility
```

`partition_for_context` may materialize the four groups and is not called by
the generation or detection hot path.

## Batched Generation

### Shared Engine

Create one shared generation engine used by the generic runner, Pareto sweep,
length sweep, capacity sweep, and comparison adapters. A generation request
contains:

- sample ID and generation seed;
- prompt token IDs;
- exact continuation length;
- optional encoded payload bits;
- watermark operating point.

Requests are grouped without reordering into batches of at most
`generation_batch_size`. Prompts are left padded. Attention masks ensure pads
do not affect the model, and the final prompt tokens remain the watermark
context. All rows in one request group have the same exact continuation length
and operating point.

Before completion filtering, the engine assigns immutable batch IDs from each
request's source position and the configured batch size. It derives a
deterministic batch seed from that ID, the ordered sample IDs, their per-sample
generation seeds, engine version, and configured batch size. Baseline and
watermarked calls for a paired batch enter identical RNG states. The same v2
configuration on the same recorded software/hardware environment therefore
reproduces the same outputs. Cross-device or cross-library-version bit identity
is not promised because model kernels may differ.

On resume, a partially persisted batch is regenerated with all of its original
members and seed; only missing records are appended. Completed-row filtering
never changes batch membership. Changing grouping or batch size changes the
sampling stream and is rejected by metadata validation.

Generation uses the Transformers KV cache. Optional compile/static-cache
settings are excluded from the first implementation because they can introduce
model-specific behavior; they may be added later as explicit metadata-backed
runtime options.

### Batched Logits Processing

`DualLayerLogitsProcessor` accepts either one encoded codeword or one codeword
per row. On every decoding step it:

1. reads the last `context_width` token IDs for all rows in one device-to-host
   transfer used only for keyed context seed derivation;
2. derives partition and allocator seeds per row;
3. obtains a `[batch, vocab]` region tensor through the vector partition path;
4. selects each row's allocated code bit and applies its own presence/payload
   bias with tensor masks.

No payload, trace, partition seed, or code-bit state is shared between rows.
`last_traces` contains one lightweight trace per row. The legacy `last_trace`
property remains available only when the batch contains one row.

External MPAC and Segment/RS-BH processors are wrapped by a
`PerRowLogitsProcessorAdapter`. Model forward passes remain batched while each
upstream processor receives a one-row view. This provides batch support without
requiring changes in external repositories, although their processor portion
may remain less efficient than PIPER v2.

### Outputs and Timing

Each result remains one JSONL record. Records add:

```text
engine_version
generation_batch_id
generation_batch_size_actual
generation_batch_seconds
amortized_generation_seconds
```

Existing `generation_seconds` fields contain amortized batch time for schema
compatibility. Experiment summaries additionally report tokens per second based
on total batch wall time. A batch is fully converted and validated before its
records are appended in source order.

## Batched Detection

Detection remains a CPU workload because it performs keyed reconstruction,
unique-context selection, binomial statistics, and BCH decoding rather than
model inference.

Add scalar `detect_*` methods backed by the v2 constant-time partition lookup,
plus ordered batch methods:

```text
detect_continuation_batch(requests)
detect_token_ids_batch(requests)
```

The execution layer groups at most `detection_batch_size` requests into one
worker task. A process pool with `detection_workers` long-lived workers creates
one detector per worker and returns indexed result lists. The parent process
writes completed records in input order. Worker initialization includes the
secret key and detector configuration but no model.

Only requested counting modes and decoding policies are evaluated. Official
paper runs compute their configured primary mode and policy. Ablation commands
explicitly request additional modes or policies. This avoids the current cost
of always computing all three counting modes and every decoder policy.

For small workloads or `--detection-workers 1`, the same batch interface runs
in-process. Exceptions are returned with the exact sample/text-class/input-mode
key. Successful rows in the same task are retained, failed rows are written to
the existing error stream, and resume skips completed keys.

## Batched Persistence and Resume

Introduce a buffered ordered JSONL writer. It accepts a completed batch,
serializes records in source order, flushes once per batch, and exposes no
concurrent file handle to workers. Existing per-record schemas stay readable.

Every input item has a stable completion key and immutable source-position
batch ID. Runners form original batches before consulting completed keys. A
fully completed batch is skipped; a partially completed batch is regenerated
in full and only its missing records are written. A crash during a write may
leave only complete newline-terminated records; startup validates the final
line and reports a truncated record rather than silently skipping it.

Batch errors never cause successful records to be marked complete before they
are persisted. Resume begins with the first missing completion key.

## Migration Scope

### PIPER generation and detection

- `run_generation.py` / `utils.generation`
- `run_detection.py` / `utils.detection`
- `experiments.run_pareto_sweep`
- `experiments.run_length_sweep`
- `experiments.run_capacity_fpr`
- `experiments.run_paper_null`
- shared-negative generation and scoring utilities

The duplicated generation loops in length and capacity sweeps become thin
adapters over the shared engine.

### Attacks and rescoring

- synonym replacement/deletion/insertion detection
- paraphrase detection (its paraphraser keeps its existing independent batch
  option)
- MPAC positive and negative rescoring
- shared-negative FPR scoring

### Comparison methods

- MPAC generation/detection
- Segment/RS-BH generation/detection

Existing batched quality and PPL scorers retain their model-specific batch
parameters but consume the common CLI defaults when no specialized override is
provided.

## Error Handling

- Invalid batch or worker values fail during argument validation.
- Engine/version or runtime mismatches fail before loading a model.
- CUDA OOM reports the configured and actual batch sizes and leaves already
  flushed records resumable only under the same configuration.
- Shape mismatches identify the affected sample IDs.
- A per-row payload length mismatch rejects the batch before model execution.
- Worker crashes terminate the detection stage with the last committed key and
  preserve resumability.

## Testing

### Partition

- Golden vectors match between scalar CPU and tensor CPU/CUDA paths.
- Every eligible token receives exactly one rank and region.
- Region cardinalities are exact for even and uneven vocabulary sizes.
- Excluded and out-of-range IDs return no region.

### Generation

- Batch sizes 1 and 16 produce the requested number of exact-length outputs.
- Repeating one v2 configuration produces identical token IDs.
- Different rows use their own prompt context, payload, and allocated bit.
- Left padding never enters a row's watermark context.
- Baseline/watermarked paired batches begin from identical RNG state.
- The external per-row processor adapter preserves row isolation.

### Detection

- Batched and scalar v2 detection records are exactly equal apart from timing.
- Worker counts 1 and 8 preserve input ordering and decisions.
- Known-boundary and blind-text modes preserve their current boundaries.
- Configured-only and all-policy ablation paths emit the expected fields.

### Persistence and CLI

- Defaults are 16/64/8 and every runner accepts explicit overrides.
- Metadata and resume validation include all batch settings and engine version.
- Partial output resumes without duplicates or omissions.
- Partial-batch resume regenerates the original batch and preserves its seed.
- One failed row does not discard successful siblings in its detection task.
- Legacy v1 directories are rejected with an actionable message.

## Documentation and Rollout

Update `README.md`, `PAPER_EXPERIMENTS.md`, and `experiments/README.md` with the
three options and the v1/v2 resume boundary. Existing running v1 jobs continue
unchanged because Python processes have already loaded their modules. New v2
formal experiments use new output directories; no existing formal result is
rewritten in place.
