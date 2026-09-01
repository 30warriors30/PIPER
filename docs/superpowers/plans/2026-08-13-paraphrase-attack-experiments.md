# Paraphrase Attack Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable paraphrase attack runner for the existing `soft_p0_m2` Pareto outputs and run it against the current method.

**Architecture:** Reuse the synonym attack runner's data loading, runtime config, detector construction, threshold freezing, ROC writing, and output preparation. Add a small paraphrase-specific result dataclass, seq2seq paraphraser wrapper, attack record builder, detection adapter, metrics diagnostics, and CLI.

**Tech Stack:** Python 3.10, Hugging Face Transformers, local PEGASUS paraphrase model, existing dual-layer detector utilities, existing `evaluation.metrics.evaluate_records`, matplotlib, pytest.

## Global Constraints

- Work in the current workspace because the experiment outputs live here.
- Do not modify unrelated user changes in the dirty worktree.
- Use the existing experiment tokenizer for attacked-token ids.
- Use the experiment's existing calibrated threshold for fixed TPR/FPR decisions.
- Include `watermarked`, `unwatermarked`, and `natural` examples.
- Decode payload only for `watermarked` examples.
- Run a `--max-samples 2` smoke before the full run.

---

### Task 1: Tests For Paraphrase Runner Interfaces

**Files:**
- Create: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_paraphrase_attacks.py`

**Interfaces:**
- Produces expected imports from `experiments.attack.run_paraphrase_attacks`:
  - `ParaphraseResult`
  - `build_parser() -> argparse.ArgumentParser`
  - `default_output_dir(experiment_dir: Path, point_id: str, paraphraser_slug: str) -> Path`
  - `make_paraphrase_result(text: str, attacked_text: str, tokenizer: Any, paraphrase_seconds: float, model_path: str, generation: Mapping[str, Any]) -> ParaphraseResult`
  - `detect_paraphrased_example(...) -> list[dict[str, Any]]`
  - `evaluate_paraphrase_attack(...) -> dict[str, Any]`

- [ ] **Step 1: Write failing tests**

Add tests for parser defaults, output directory naming, paraphrase diagnostics, detection input modes, negative payload-decode skip, frozen threshold metrics, and module help.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run --no-capture-output -n BREW pytest dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_paraphrase_attacks.py -q
```

Expected: import failure for `experiments.attack.run_paraphrase_attacks`.

### Task 2: Implement Paraphrase Runner

**Files:**
- Create: `dual_layer_watermark_fixed_length_error_erasure/experiments/attack/run_paraphrase_attacks.py`
- Test: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_paraphrase_attacks.py`

**Interfaces:**
- Consumes helpers from `experiments.attack.run_synonym_attacks`:
  - `AttackExample`
  - `INPUT_MODES`
  - `_prepare_output`
  - `_sample_seed`
  - `_timed_detect`
  - `_write_jsonl`
  - `build_runtime_config`
  - `load_attack_examples`
  - `resolve_input_path`
  - `resolve_output_path`
  - `write_roc_points`
- Produces the CLI module `python -m experiments.attack.run_paraphrase_attacks`.

- [ ] **Step 1: Implement minimal code for tests**

Create the dataclass, parser, output path helper, result maker, detection adapter, evaluation wrapper, CSV writer, ROC plotter, and run loop.

- [ ] **Step 2: Run tests to verify they pass**

Run:

```bash
conda run --no-capture-output -n BREW pytest dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_paraphrase_attacks.py -q
```

Expected: all tests pass.

### Task 3: Regression Test Existing Attack Code

**Files:**
- Test: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_synonym_attacks.py`
- Test: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_paraphrase_attacks.py`

- [ ] **Step 1: Run focused regression suite**

Run:

```bash
conda run --no-capture-output -n BREW pytest dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_synonym_attacks.py dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_paraphrase_attacks.py -q
```

Expected: all tests pass.

### Task 4: Smoke And Full Experiment

**Files:**
- Writes: `dual_layer_watermark_fixed_length_error_erasure/outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_paraphrase_pegasus*`

- [ ] **Step 1: Run CLI help**

Run:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
conda run --no-capture-output -n BREW python -m experiments.attack.run_paraphrase_attacks --help
```

Expected: exits 0 and includes `--paraphraser-model`.

- [ ] **Step 2: Run 2-sample smoke**

Run:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
conda run --no-capture-output -n BREW python -m experiments.attack.run_paraphrase_attacks \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --point-id soft_p0_m2 \
  --output-dir outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_paraphrase_pegasus_smoke2 \
  --max-samples 2 \
  --overwrite
```

Expected: creates `summary.csv`, `summary.json`, `paraphrase/metrics.json`, and ROC figure.

- [ ] **Step 3: Run 200-sample experiment**

Run:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
conda run --no-capture-output -n BREW python -m experiments.attack.run_paraphrase_attacks \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --point-id soft_p0_m2 \
  --output-dir outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_paraphrase_pegasus \
  --max-samples 200 \
  --overwrite
```

Expected: creates the full paraphrase attack outputs and summary metrics.
