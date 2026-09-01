# Fixed-Length Generation and Bounded Error-Erasure BCH Design

## 1. Scope

This change modifies the existing direct-entry dual-layer watermark project in two focused ways:

1. Every completed experiment sample must contain exactly 200 continuation tokens for all three text classes: `watermarked`, `unwatermarked`, and `natural`.
2. Detection must add a bounded error-erasure BCH decoding policy that uses the existing BCH backend and explicitly consumes erased code-bit positions.

The existing `strict` and `hard_fill` policies remain available for comparison. No unrelated watermark partition, allocation, threshold, or model-loading behavior changes in this work.

## 2. Exact-Length Three-Class Sampling

### 2.1 Experiment invariant

For the configured continuation length `L` (default `L = 200`), every record with `status = "completed"` must satisfy:

```text
len(watermarked.token_ids)   == L
len(unwatermarked.token_ids) == L
len(natural.token_ids)       == L
```

A run requesting 200 samples therefore completes only after it has written 200 valid three-class records satisfying this invariant.

### 2.2 Watermarked and unwatermarked model generation

Both model-generated continuations use the same exact-length stopping policy:

```python
min_new_tokens = L
max_new_tokens = L
```

The paired generation setup remains unchanged otherwise:

- same prompt;
- same generation seed;
- same sampling configuration;
- watermarked generation additionally receives the watermark logits processor.

EOS remains a valid generated token unless excluded by the watermark configuration, but generation is not allowed to terminate before `L` new tokens. The generated continuation token IDs, not re-tokenized decoded text, are the source of truth.

The existing CLI option `--max-new-tokens` remains accepted in this change to avoid unnecessary interface churn, but its semantics become “exact number of generated continuation tokens”. The terminal summary and experiment metadata explicitly label it as an exact continuation length.

### 2.3 Natural C4 continuation

For a C4 document tokenized without special tokens, let the token sequence be `x_0, ..., x_{R-1}` and let the exact continuation length be `L`.

A document is eligible only if it contains at least:

```text
context_width + L
```

tokens. Ineligible documents are filtered before generation and do not count toward the requested completed-sample total.

For an eligible document:

```text
natural_token_ids = raw_token_ids[-L:]
prompt_source_ids = raw_token_ids[:-L]
```

The model-facing prompt is built from the suffix of `prompt_source_ids`. The code must account for tokenizer-added special tokens (for example OPT's leading BOS token) and trim source tokens until the final model input satisfies:

```text
len(model_prompt_token_ids) + L <= model_max_length
```

`SampleRecord` therefore carries exact `prompt_token_ids` and `natural_token_ids`. Generation constructs `input_ids` directly from `prompt_token_ids`; it does not decode the prompt and tokenize it again. The prompt must still contain at least `context_width` history tokens after special-token preparation. The natural continuation is decoded only for human-readable storage; its original 200 token IDs are stored directly and used during detection. It is never reconstructed by decoding and re-tokenizing.

### 2.4 Candidate streaming and filtering

Dataset iteration must not stop after merely reading `max_samples` raw rows, because some C4 documents may be too short. The generation loop instead:

1. skips `sample_offset` raw rows;
2. streams candidate rows;
3. filters candidates that cannot provide a valid prompt plus `L` natural tokens;
4. generates the paired watermarked and unwatermarked continuations;
5. writes a completed record only after all three token-length checks pass;
6. stops when the number of unique completed sample IDs reaches `max_samples`.

Filtered candidates are appended to:

```text
outputs/<run_id>/errors/filtered_samples.jsonl
```

with the sample ID, reason, raw token count, required token count, and timestamp. Filtering is not counted as a generation error.

For a finite local dataset, exhaustion before reaching the requested completed count is a global run failure with a clear error showing requested, completed, filtered, and failed counts.

### 2.5 Resume behavior

Resume remains append-only. Completed sample IDs are skipped. The loop continues reading candidates until the target number of unique completed samples is reached. Before appending a completed record, the run validates all three exact-length invariants.

## 3. Bounded Error-Erasure BCH Decoder

### 3.1 Input representation

For BCH code length `n`, counting produces per-position votes:

```text
N0[j] = number of tokens supporting code bit 0
N1[j] = number of tokens supporting code bit 1
```

The ternary received word is:

```text
0, if N0[j] > N1[j] and evidence is sufficient
1, if N1[j] > N0[j] and evidence is sufficient
?, if evidence is insufficient or the vote is tied
```

The erasure set is the existing `CountingResult.erasures` field. The first implementation does not introduce additional low-confidence erasures beyond the current minimum-evidence and tie rules.

### 3.2 Decoding method

The new method is a bounded enumeration wrapper around the validated ordinary BCH decoder.

Given erasure set `E`, with `e = |E|`:

1. Reject immediately when `e >= d`, where `d` is the BCH minimum/design distance used by the codec. No codeword can satisfy the unique error-erasure condition `2v + e < d` in that case.
2. Enumerate all `2^e` assignments to the erased positions.
3. For each filled hard word, call the existing ordinary BCH decoder.
4. For every successful decoded message, re-encode the message to obtain a canonical valid BCH codeword `c`.
5. Compute `v`, the Hamming disagreement count between `c` and the observed hard decisions at non-erased positions only.
6. Retain the candidate only when:

```text
2 * v + e < d
```

7. Deduplicate retained candidates by canonical codeword/message.
8. Return:
   - `decoded` when exactly one candidate remains;
   - `decode_failed` when no candidate remains;
   - `ambiguous` when multiple distinct candidates remain.

Although uniqueness theory implies at most one codeword inside the strict radius, the explicit `ambiguous` status protects against backend anomalies, parameter mistakes, and future decoder changes.

### 3.3 Computational bound

The decoder is intentionally bounded. The maximum number of assignments is configurable as:

```text
max_erasure_assignments
```

with default `64`. For BCH `(23, 8, t=3)` with `d=7`, the unique-decoding rule already restricts useful erasure counts to at most 6, so the default bound covers all guaranteed cases:

```text
2^6 = 64
```

If `2^e` exceeds the configured bound, the decoder returns `too_many_erasures` rather than silently applying hard fill or an unbounded search.

### 3.4 Result structure

`DecodeResult` is extended with optional diagnostics while preserving the existing fields:

```text
status
message_bits
corrected_errors
error
codeword_bits
erasure_count
known_position_errors
distance_cost
candidate_count
assignments_tested
```

For error-erasure decoding:

```text
distance_cost = 2 * known_position_errors + erasure_count
```

`corrected_errors` remains the ordinary BCH backend's correction count only when it has a clear meaning; error-erasure evaluation must use `known_position_errors` and `distance_cost` for interpretation.

### 3.5 Detection policies

The supported policies become:

```text
strict
hard_fill
error_erasure
```

Every detection record contains results for all three policies when `evaluate_all_policies` is enabled. The default primary policy becomes:

```text
error_erasure
```

The selected primary policy controls `decode_status`, `decoded_message`, and the presence-gated message output. Presence detection itself remains unchanged.

## 4. Output and Metrics

### 4.1 Sample records

Each completed sample stores exact token counts:

```json
{
  "watermarked": {"token_ids": [], "num_tokens": 200},
  "unwatermarked": {"token_ids": [], "num_tokens": 200},
  "natural": {"token_ids": [], "num_tokens": 200}
}
```

### 4.2 Detection records

Each detection record adds:

```text
error_erasure
```

alongside `strict` and `hard_fill`, including all diagnostics listed above.

### 4.3 Evaluation metrics

For each decoding policy, evaluation reports both coverage and correctness:

```text
<policy>_decode_success_rate
<policy>_exact_message_recovery
<policy>_conditional_message_ber
```

It also retains the existing headline fields for the selected primary policy:

```text
exact_message_recovery
message_ber
codeword_ber
```

The summary must not equate decoder success with correct recovery. Error-erasure status counts are reported separately, including `decoded`, `decode_failed`, `ambiguous`, and `too_many_erasures`.

## 5. Error Handling

Global failures terminate the run:

- model context length cannot fit `context_width + L`;
- finite dataset exhausts before the requested completed count;
- existing resume configuration disagrees with the exact-length configuration;
- completed record violates any three-class length invariant.

Per-candidate filtering continues the run:

- C4 document has fewer than `context_width + L` tokens;
- prompt remains shorter than `context_width` after splitting.

Per-sample generation failures continue to use the existing generation error log and do not count toward completed samples.

Error-erasure decoding never silently falls back to `hard_fill`. It returns an explicit non-decoded status when outside its bound or uniqueness condition.

## 6. Tests

### 6.1 Exact-length generation tests

Tests must verify:

- both model generations receive equal `min_new_tokens` and `max_new_tokens`;
- watermarked and unwatermarked outputs contain exactly `L` continuation IDs;
- natural continuation stores exactly the original last `L` C4 token IDs;
- a C4 document shorter than `context_width + L` is filtered;
- candidate iteration continues past filtered documents until the requested number of completed samples is reached;
- finite source exhaustion raises a clear global error;
- resume preserves the target completed count without duplicate completed records.

### 6.2 Error-erasure decoder tests

For random messages and controlled corruptions, the decoder must recover the original message for all guaranteed BCH `(23, 8, 3)` patterns:

```text
(v, e) = (3, 0), (2, 2), (1, 4), (0, 6)
```

because `2v + e <= 6 < 7`.

Tests also exercise patterns outside the guaranteed channel radius:

```text
(v, e) = (3, 1), (2, 3), (1, 5), (0, 7)
```

For these patterns, the decoder must remain bounded, deterministic, and internally consistent, but the test must not require recovery of the transmitted message. A bounded-distance decoder can miscorrect to a different valid codeword when the actual corruption exceeds the code's guarantee; the detector cannot identify that situation without an external payload integrity check such as CRC or MAC, which is outside this change. Every returned `decoded` result must nevertheless correspond to exactly one canonical codeword satisfying the decoder's observed-word condition `2v + e < d`; otherwise the status must be `decode_failed`, `ambiguous`, or `too_many_erasures`.

Additional tests cover:

- zero erasures matches ordinary BCH decoding;
- duplicate candidates from different fills are deduplicated;
- assignment bound is enforced;
- shortened-code position mapping remains correct;
- detection output contains all three policies;
- metrics distinguish decode coverage from exact message recovery.

## 7. Acceptance Criteria

The change is complete when:

1. A 200-sample C4 run produces 200 completed records and every class in every record has exactly 200 token IDs.
2. Early EOS no longer produces short watermarked or unwatermarked continuations.
3. Natural documents unable to provide 200 continuation tokens are filtered and replaced by later candidates.
4. `error_erasure` is available as a detection policy and is the default primary policy.
5. Guaranteed error-erasure patterns for BCH `(23, 8, 3)` pass automated tests.
6. Detection JSONL and summary metrics separately expose strict, hard-fill, and error-erasure coverage and correctness.
7. Existing presence-detection behavior and thresholds remain unchanged.
