# PIPER Top-50 Selfhash Design

## Goal

Improve PIPER generation quality by applying watermark pressure only to the
50 highest-logit candidates while matching MPAC's candidate-conditioned
selfhash definition. Preserve model-free detection and keep existing PIPER
artifacts readable under their original history-only partition semantics.

## Generation semantics

At generation step `t`, let `raw_scores` be the model logits before any
watermark transformation.

1. Select `C_t = TopK(raw_scores, 50)` independently for each batch row.
2. Allocate one PIPER BCH code-bit index from the last `context_width`
   already-generated tokens. Candidate tokens do not affect allocation.
3. For each candidate `u` in `C_t`, build the selfhash context from the last
   `context_width - 1` history tokens followed by `u`.
4. Use the keyed PRF and the v2 stateless exact partitioner to determine the
   candidate's region in `{bit0, bit1, lower_a, lower_b}` without materializing
   a full-vocabulary permutation.
5. Apply PIPER's existing presence and payload rules only to candidates in
   `C_t`. For the main `soft_p0_m2` point, only candidates in the selected
   payload child receive `+2.0`.
6. Divide by temperature, mask EOS as currently configured, apply sampling
   `top-k=50`, then `top-p=0.95`, then sample.

The top-50 membership must be computed from `raw_scores`, not from scores after
watermark bias. Excluded special tokens never receive watermark bias.

## Detection semantics

For an observed token `x_t`, reconstruct its partition from the last
`context_width - 1` previous tokens followed by `x_t`. Reconstruct the BCH
code-bit allocation from the last `context_width` previous tokens only. This
keeps the detector model-free: it never needs the generation-time top-50 list
or language-model logits.

For selfhash runs, repeated-event suppression uses the candidate-conditioned
context, which is the four-token n-gram when `context_width=4`. Presence is an
upper hit exactly when the v2 region is `bit0` or `bit1`.

## Compatibility

- Existing JSON files without the new fields load as history-only, v1,
  unlimited-candidate runs.
- New commands explicitly record the seeding scheme, partition engine,
  watermark candidate limit, and sampling top-k in experiment metadata.
- History-only mode remains available for reproducing prior PIPER results.
- Selfhash requires the v2 stateless exact partition engine.

## Initial paper configuration

- `context_width=4`
- `seeding_scheme=selfhash`
- `partition_engine=v2`
- `candidate_top_k=50`
- `sampling_top_k=50`
- `temperature=1.0`
- `top_p=0.95`
- `presence_mode=soft`
- `delta_presence=0.0`
- `delta_payload=2.0`
- BCH `(23, 8, 3)`
- exactly 300 generated tokens

## Verification

- A candidate outside the raw top-50 is never biased.
- Candidate-conditioned contexts differ when only the candidate differs.
- Generation and detection assign the same observed token to the same region.
- Sampling applies top-k before top-p after watermark processing.
- Old configuration dictionaries load with legacy semantics.
- Existing and new unit tests pass in Python 3.10 with PyTorch 2.8.
