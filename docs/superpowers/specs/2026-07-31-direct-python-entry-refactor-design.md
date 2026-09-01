# Direct-Python Dual-Layer Watermark Project Design

**Date:** 2026-07-31  
**Status:** User-approved design, pending implementation plan  
**Scope:** A clean, from-scratch project for dual-layer multi-bit LLM watermark generation, detection, evaluation, and threshold calibration.

## 1. Goals

Build a new project that:

1. Runs directly through ordinary Python entry files.
2. Does not use YAML configuration files.
3. Does not require editable installation, `PYTHONPATH=src`, or a console-script command.
4. Separates generation, detection, evaluation, and calibration into four independent entry scripts.
5. Keeps core logic organized under `watermark/`, `evaluation/`, and `utils/`.
6. Supports local OPT-1.3B generation on an NVIDIA RTX PRO 6000 Blackwell GPU.
7. Generates paired watermarked and unwatermarked continuations and preserves the C4 natural continuation.
8. Uses the same watermark key, BCH parameters, context width, partition mode, PRF mode, allocation mode, and embedding strengths for every sample in a run.
9. Uses a deterministic but different random payload for each sample.
10. Persists experiment metadata so detection can reconstruct the generation-time watermark configuration exactly.

## 2. Non-Goals

This project will not include:

- the old `src/dual_layer_watermark/` layout;
- YAML configuration readers;
- the old `dual-watermark` console command;
- `pyproject.toml` or editable package installation;
- compatibility adapters for the old output schema;
- a `legacy/` directory;
- migration code from the previous project;
- CPU environment support in the first release;
- balanced allocation or offset search in the first release;
- paraphrase, insertion, deletion, or substitution attacks in the first release;
- PPL, BERTScore, or semantic-quality experiments in the first release.

## 3. Project Structure

```text
dual_layer_watermark/
├── run_generation.py
├── run_detection.py
├── evaluate_results.py
├── calibrate_threshold.py
│
├── watermark/
│   ├── __init__.py
│   ├── config.py
│   ├── prf.py
│   ├── partition.py
│   ├── allocator.py
│   ├── ecc.py
│   ├── logits_processor.py
│   ├── detector.py
│   └── result_types.py
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── roc.py
│   ├── calibration.py
│   └── report.py
│
├── utils/
│   ├── __init__.py
│   ├── arguments.py
│   ├── validation.py
│   ├── datasets.py
│   ├── model.py
│   ├── generation.py
│   ├── io.py
│   ├── logging.py
│   ├── seeds.py
│   └── environment.py
│
├── tests/
│   ├── test_arguments.py
│   ├── test_validation.py
│   ├── test_prf.py
│   ├── test_partition.py
│   ├── test_allocator.py
│   ├── test_ecc.py
│   ├── test_logits_processor.py
│   ├── test_detector.py
│   ├── test_metrics.py
│   ├── test_io_resume.py
│   └── test_integration_local_model.py
│
├── scripts/
│   ├── create_environment.sh
│   ├── smoke_test.sh
│   └── run_c4_opt13b_200.sh
│
├── outputs/
├── environment.yml
├── requirements.txt
├── README.md
└── .gitignore
```

### 3.1 Responsibility Boundaries

- `watermark/`: watermark PRF, exact vocabulary partitioning, code-bit allocation, BCH encoding/decoding, logits modification, detection, and typed result records.
- `evaluation/`: TPR/FPR/BER, ROC/AUC, threshold calibration, CSV/JSON summaries.
- `utils/`: command-line parsing, parameter validation, model loading, dataset processing, paired generation, deterministic seeds, logging, JSONL persistence, and environment checks.
- Top-level scripts: parse arguments, validate inputs, print a run summary, and invoke one workflow. They must not duplicate watermark or metric logic.

## 4. Entry Scripts

All scripts are non-interactive. They perform:

```text
parse CLI arguments
→ validate
→ print a complete parameter summary
→ execute immediately
```

Each script supports `--dry-run`. Dry-run performs argument parsing, validation, environment checks that do not require loading the full model, and summary printing, then exits without modifying outputs.

### 4.1 `run_generation.py`

Responsibilities:

- load local OPT-1.3B and tokenizer;
- load C4 or OpenGen data;
- derive one deterministic random payload per sample;
- BCH-encode the payload;
- generate paired watermarked and unwatermarked continuations using the same sample-level generation seed;
- retain the dataset's natural continuation;
- write `experiment.json`, `samples.jsonl`, logs, and generation error records.

Typical invocation:

```bash
python run_generation.py \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --device cuda \
  --dtype float16 \
  --dataset c4 \
  --dataset-name allenai/c4 \
  --dataset-config realnewslike \
  --dataset-split validation \
  --streaming \
  --max-samples 200 \
  --max-new-tokens 200 \
  --secret-key dual-layer-key-2026 \
  --context-width 4 \
  --presence-mode hard \
  --delta-presence 2.0 \
  --delta-payload 2.0 \
  --prf-mode paper_shared \
  --partition-mode exact_permutation \
  --allocation-mode hash_mod \
  --ecc-n 23 \
  --ecc-k 8 \
  --ecc-t 3 \
  --temperature 1.0 \
  --top-p 0.95 \
  --global-seed 42 \
  --message-seed 42 \
  --run-id c4_opt13b_hard_200
```

Defaults:

- `device=cuda`
- `dtype=float16`
- `local_files_only=true`
- `dataset=c4`
- `dataset_name=allenai/c4`
- `dataset_config=realnewslike`
- `dataset_split=validation`
- `streaming=true`
- `max_samples=200`
- `sample_offset=0`
- `max_new_tokens=200`
- `temperature=1.0`
- `top_p=0.95`
- `do_sample=true`
- `global_seed=42`
- `message_seed=42`
- `context_width=4`
- `presence_mode=hard`
- `delta_presence=2.0`
- `delta_payload=2.0`
- `prf_mode=paper_shared`
- `partition_mode=exact_permutation`
- `allocation_mode=hash_mod`
- `ecc_n=23`
- `ecc_k=8`
- `ecc_t=3`
- `output_root=outputs`
- `resume=false`
- `overwrite=false`

`--secret-key`, `--model-path`, and `--run-id` are required. `--resume` and `--overwrite` are mutually exclusive.

### 4.2 `run_detection.py`

Supports two input modes.

#### Samples mode

```bash
python run_detection.py \
  --input-type samples \
  --run-dir outputs/c4_opt13b_hard_200
```

The script reads `experiment.json` and `samples.jsonl`. It must use the generation-time secret key, BCH parameters, context width, partition mode, PRF mode, allocation mode, and tokenizer path from `experiment.json`. It must not silently override these parameters.

For each of `watermarked`, `unwatermarked`, and `natural`, it writes separate records for:

- `known_boundary`
- `blind_text`

#### External text mode

```bash
python run_detection.py \
  --input-type text \
  --input-file data/external_texts.jsonl \
  --text-field text \
  --output-dir outputs/external_detection \
  --model-path /data/yanlu/BREW/models/facebook/opt-1.3b \
  --secret-key dual-layer-key-2026 \
  --context-width 4 \
  --ecc-n 23 \
  --ecc-k 8 \
  --ecc-t 3
```

External text defaults to blind detection. If `--prompt-field` and `--continuation-field` are provided, known-boundary detection is used.

Detection defaults:

- `threshold_mode=theoretical`
- `target_fpr=0.01`
- `counting_mode=all_tokens`
- `primary_decoding_policy=strict`
- `min_tokens_per_code_bit=1`
- compute both strict and hard-fill decoding results
- compute both Z-score and exact binomial right-tail p-value

### 4.3 `evaluate_results.py`

```bash
python evaluate_results.py \
  --run-dir outputs/c4_opt13b_hard_200
```

Reads:

- `experiment.json`
- `samples.jsonl`
- `detections.jsonl`

Writes:

- `metrics.json`
- `metrics.csv`
- `roc_points.csv`
- `summary.txt`

Known-boundary and blind-text metrics must be reported separately. The evaluator must not combine them into one TPR, FPR, BER, or AUC.

Defaults:

- evaluate both input modes;
- use strict decoding as the headline policy;
- report hard-fill as a diagnostic;
- no bootstrap confidence intervals by default;
- confidence level is 0.95 if bootstrap is requested.

### 4.4 `calibrate_threshold.py`

```bash
python calibrate_threshold.py \
  --run-dir outputs/c4_opt13b_hard_200 \
  --negative-source unwatermarked \
  --input-mode known_boundary \
  --target-fpr 0.01
```

Negative sources:

- `unwatermarked`
- `natural`
- `combined`

The calibrated threshold is:

\[
\tau_{\mathrm{cal}}
=
\operatorname{Quantile}_{1-\alpha}
\left(Z_1^{(0)},\ldots,Z_N^{(0)}\right).
\]

The script writes `calibration.json` and never overwrites `experiment.json` or the original detection records.

## 5. Watermark Algorithm

### 5.1 Shared Run-Level Parameters

All generated texts in one run share:

\[
\Theta_{\mathrm{wm}}
=
(K,n,k,t,h,
\text{presence mode},
\text{PRF mode},
\text{partition mode},
\text{allocation mode},
\delta_{\mathrm{presence}},
\delta_{\mathrm{payload}}).
\]

Only the payload changes by sample.

### 5.2 Per-Sample Payload

For sample identifier `i`:

\[
m_i = \operatorname{PRG}(\texttt{message_seed}, i)
\in \{0,1\}^{k}.
\]

The payload is deterministic across reruns with the same message seed and sample identifier, but differs between samples.

The detector never uses the stored payload or stored encoded codeword to decode. They are ground truth for evaluation only.

### 5.3 BCH

The user supplies a valid binary BCH triple `(n,k,t)`. In the default experiment:

\[
(n,k,t)=(23,8,3).
\]

The implementation validates that the configured code can be constructed and that the effective correction capability is at least `t`. Message length is exactly `k`; there is no external message truncation step.

### 5.4 PRF and Partition

Default PRF mode: `paper_shared`.

For token step `t`, the key and previous `h` token IDs determine a deterministic seed. The first implementation uses `exact_permutation` only.

The allowed vocabulary is deterministically permuted and split into four near-equal disjoint subsets:

- `V0`
- `V1`
- two lower-half subsets

Then:

\[
V^{\mathrm{up}} = V^{(0)} \dot\cup V^{(1)},
\qquad
V^{\mathrm{down}} = V \setminus V^{\mathrm{up}}.
\]

Special non-generatable tokens are excluded. EOS remains eligible by default.

### 5.5 Allocation

The first release implements:

\[
j_t = s_t \bmod n.
\]

This is `hash_mod`. The interface must leave room for `balanced_permutation`, but requesting it in the first release must raise a clear `NotImplementedError`; it must not silently fall back to `hash_mod`.

### 5.6 Embedding

Default mode is hard-soft:

- tokens in `V_down` receive `-inf` logits;
- tokens in the target payload subset `V^(b_t)` receive `delta_payload`;
- EOS remains usable;
- generation follows the configured sampling parameters.

Soft presence remains an optional CLI mode. In soft mode, `delta_presence` biases `V_up` rather than masking `V_down`.

## 6. Detection

### 6.1 Known Boundary

The prompt contributes to the first `h` contexts but does not contribute to detection counts. Only continuation tokens contribute to:

- upper-half count;
- Z-score;
- exact binomial p-value;
- per-code-bit votes.

### 6.2 Blind Text

For tokenized text `y_0,...,y_{N-1}`, blind detection scores tokens from index `h` onward:

\[
s_t = H(K,y_{t-h:t-1}),
\qquad t=h,\ldots,N-1.
\]

The whole text after warm-up is treated as the detection region.

### 6.3 Presence Statistic

For `N` scored tokens and `S` upper-half hits:

\[
Z = \frac{S-N/2}{\sqrt{N/4}}.
\]

Theoretical threshold:

\[
\tau = \Phi^{-1}(1-\alpha),
\qquad \alpha=0.01 \text{ by default}.
\]

The detector also computes:

\[
p_{\mathrm{bin}}
=\Pr(\operatorname{Bin}(N,1/2)\ge S).
\]

### 6.4 Payload Voting and Decoding

For code bit `j`:

\[
C_j=N_j^{(0)}+N_j^{(1)},
\qquad
D_j=N_j^{(1)}-N_j^{(0)}.
\]

A position is an erasure if:

- `C_j < min_tokens_per_code_bit`, or
- `D_j = 0`.

Two policies are always computed:

- `strict`: any erasure yields `insufficient_evidence` before BCH decoding;
- `hard_fill`: erase positions are deterministically filled and BCH decoding is attempted.

The configured primary policy controls headline metrics only.

## 7. C4 and Paired Generation

### 7.1 Three Text Classes

For every selected C4 document, save:

1. `watermarked`
2. `unwatermarked`
3. `natural`

Watermarked and unwatermarked generation use the same prompt, sampling parameters, and sample-level random seed. Natural text is the held-out suffix of the C4 document.

### 7.2 Context-Length Safety

Let model capacity be `L_max` and requested continuation length be `T`. The prompt must satisfy:

\[
L_{\mathrm{prompt}} + T \le L_{\max}.
\]

The dataset processor tokenizes the document, reserves enough tokens for generation, and produces a non-empty natural continuation. Documents that cannot supply the required prompt and natural suffix are recorded as `InsufficientDocumentLength` errors and do not terminate the run.

### 7.3 Streaming

C4 defaults to streaming mode. The first release does not shuffle streaming C4 because materializing or buffering the remote iterable would add unnecessary complexity and memory risk.

## 8. Output Protocol

### 8.1 Run Directory

```text
outputs/<run_id>/
├── experiment.json
├── samples.jsonl
├── detections.jsonl
├── calibration.json
├── metrics.json
├── metrics.csv
├── roc_points.csv
├── summary.txt
├── logs/
│   ├── generation.log
│   ├── detection.log
│   ├── evaluation.log
│   └── calibration.log
└── errors/
    ├── generation_errors.jsonl
    └── detection_errors.jsonl
```

Files that do not apply yet are absent rather than empty placeholders.

### 8.2 `experiment.json`

Stores run-level parameters once, including:

- schema version;
- run ID and creation timestamp;
- model path, device, dtype, tokenizer and context length;
- dataset source and selection parameters;
- generation parameters and seeds;
- watermark parameters;
- BCH parameters;
- package versions and CUDA/GPU information.

The secret key is stored because the user explicitly requires automatic reproduction of generation-time detection parameters. The README must warn that the run directory is sensitive and should not be distributed when the key must remain secret.

### 8.3 `samples.jsonl`

Each completed record contains:

- schema version;
- sample ID;
- status;
- prompt text and token IDs;
- random message bits and BCH codeword;
- generation seed;
- watermarked text, token IDs, length, and timing;
- unwatermarked text, token IDs, length, and timing;
- natural text, token IDs, and length.

Each error record contains:

- sample ID;
- status `error`;
- exception type;
- message;
- traceback tail.

### 8.4 `detections.jsonl`

The deduplication key is:

\[
(\texttt{sample_id},\texttt{text_class},\texttt{input_mode}).
\]

Each record contains presence statistics, per-bit counts and margins, erasures, strict and hard-fill decoding results, and elapsed detection time.

## 9. Resume and Overwrite

### 9.1 Generation

- default: fail if the run directory already contains an experiment;
- `--resume`: compare every load-bearing CLI parameter with `experiment.json` before continuing;
- completed sample IDs are skipped;
- error sample IDs are retried;
- `--overwrite`: requires an explicit flag and recreates the run directory;
- `--resume` and `--overwrite` are mutually exclusive.

No mismatched configuration may be appended to an existing run.

### 9.2 Detection

- `--resume` skips existing `(sample_id,text_class,input_mode)` records;
- error records are retried;
- alternate threshold experiments must write a separately named detection file and may not overwrite the baseline file silently.

## 10. Validation

Validation occurs before loading the full model or beginning dataset iteration.

Checks include:

- positive sample counts, token lengths, and context width;
- sampling ranges (`temperature>0`, `0<top_p<=1`);
- existing local model directory and required model/tokenizer files;
- CUDA availability when requested;
- requested dtype compatibility;
- Blackwell capability `(12,0)` and `sm_120` support in the installed PyTorch wheel;
- valid mode choices;
- valid and constructible BCH parameters;
- `ecc_k` equals payload length by definition;
- output directory conflict rules;
- resume metadata equality;
- required input files for detection/evaluation/calibration.

Validation must produce actionable error messages that identify the invalid field and value.

## 11. Error Handling and Logging

### 11.1 Global Errors

Terminate immediately for:

- missing model path;
- unavailable CUDA;
- PyTorch wheel without `sm_120` support;
- invalid BCH configuration;
- unreadable dataset source;
- run directory conflicts;
- resume parameter mismatch;
- missing required run files.

### 11.2 Per-Sample Errors

A single sample failure is recorded and processing continues. Generation and detection error files contain the full error classification and traceback tail.

### 11.3 Progress

Terminal output is concise, for example:

```text
Generating: 74/200 | completed=73 | failed=1
```

Detailed environment, timing, memory, sample identifiers, and tracebacks go to log files.

## 12. Environment

### 12.1 Target

- NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- compute capability 12.0
- Python 3.10
- PyTorch 2.8.0 CUDA 12.8 wheel

### 12.2 `environment.yml`

```yaml
name: dual-watermark-blackwell

channels:
  - conda-forge

dependencies:
  - python=3.10
  - pip
  - setuptools
  - wheel
  - git
  - pip:
      - -r requirements.txt
```

### 12.3 `requirements.txt`

`torch` is intentionally excluded.

```text
transformers==4.56.2
datasets==2.21.0
accelerate==1.10.1
safetensors==0.6.2
galois==0.4.11
numpy==1.26.4
scipy==1.13.1
scikit-learn==1.5.2
joblib==1.4.2
threadpoolctl==3.5.0
tqdm==4.67.1
charset-normalizer==3.4.3
requests==2.32.4
pytest==8.3.5
pytest-cov==6.0.0
```

### 12.4 PyTorch Installation

After creating the Conda environment:

```bash
python -m pip install \
  torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

The environment script verifies:

- PyTorch version;
- CUDA runtime;
- CUDA availability;
- GPU name;
- compute capability `(12,0)`;
- `sm_120` in `torch.cuda.get_arch_list()`;
- one real float16 CUDA matrix multiplication.

## 13. Test Strategy

### 13.1 Unit Tests

Tests must cover:

- CLI defaults, required fields, choices, and mutually exclusive flags;
- validation errors and resume mismatch reporting;
- deterministic PRF and domain handling;
- exact, disjoint vocabulary partitioning and special-token exclusion;
- hash-mod allocation;
- explicit `NotImplementedError` for balanced allocation;
- BCH round trip and correction up to `t` errors;
- hard and soft logits behavior;
- EOS handling;
- known-boundary and blind detection;
- Z-score and exact binomial p-value;
- strict and hard-fill behavior;
- TPR/FPR/BER/ROC/AUC calculations;
- separation of known-boundary and blind metrics;
- append-only JSONL and resume deduplication.

### 13.2 Integration Test

With `LOCAL_OPT_MODEL` set, the integration test loads the local model and performs:

```text
unwatermarked generation
→ watermarked generation
→ detection
→ BCH payload recovery
```

It generates only a few tokens and is skipped with a clear message when the environment variable is absent.

### 13.3 C4 Smoke Test

`scripts/smoke_test.sh` runs three C4 samples with 32 new tokens, followed by detection and evaluation.

### 13.4 Formal C4 Run

`scripts/run_c4_opt13b_200.sh` runs 200 C4 samples with 200 new tokens, then detection and evaluation, with resume enabled.

## 14. Security and Reproducibility

- Python's built-in `hash()` must not be used for watermark seeds.
- Use a stable keyed cryptographic hash such as BLAKE2b or HMAC-SHA256 over canonical byte serialization of the key, domain label, and token IDs.
- Store package versions, model metadata, random seeds, and GPU information in `experiment.json`.
- Watermarked and unwatermarked generations use common random numbers within each sample.
- Message generation uses a separate deterministic seed stream.
- Detection is blind with respect to the stored ground-truth payload.
- The experiment directory contains the secret key and is therefore sensitive.

## 15. Acceptance Criteria

The first release is accepted when all of the following hold:

1. A clean environment can be created with `environment.yml`, `requirements.txt`, and the documented cu128 PyTorch command.
2. The environment validation script confirms `sm_120` and executes a CUDA matrix multiplication.
3. Each top-level script runs directly with `python <script>.py` and requires no installation step.
4. A dry-run validates and prints parameters without loading the full model or writing experiment data.
5. The local OPT-1.3B model can be loaded offline.
6. The three-sample C4 smoke test completes generation, detection, and evaluation.
7. Generation produces one experiment record and paired three-class sample records.
8. Detection reconstructs generation-time parameters from `experiment.json` in samples mode.
9. Known-boundary and blind results remain separate in all output metrics.
10. Unit tests pass and the local-model integration test passes in the target environment.
11. Resume rejects parameter mismatches and does not duplicate completed records.
12. The 200-sample C4 script can resume after interruption and produce final metrics.
