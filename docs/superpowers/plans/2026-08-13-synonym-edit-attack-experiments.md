# Synonym Edit Attack Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a paper-aligned synonym substitution attack runner that evaluates replacement, deletion-like, and insertion-like robustness for the existing `soft_p0_m2` Pareto outputs.

**Architecture:** Split the work into a small attack-core module and a CLI runner. The attack core handles word spans, synonym candidates, tokenizer-length classification, and deterministic edits; the runner handles experiment I/O, detector reuse, metrics, CSV/JSONL outputs, and ROC figures.

**Tech Stack:** Python 3.10, NLTK WordNet for real synonym candidates, Hugging Face tokenizer loading through existing utilities, existing `DualLayerDetector`, existing `evaluation.metrics.evaluate_records`, matplotlib for ROC figures, pytest.

## Global Constraints

- Use `outputs/experiments/opt13b_pareto_49x200` as the default experiment directory.
- Use `soft_p0_m2` as the default operating point.
- Use word-level synonym substitution as the main attack, not random token edits.
- Default attack rate is `0.10`.
- Attack classes are exactly `replacement`, `deletion`, and `insertion`.
- Categorize attack classes by full-continuation tokenizer length delta after the word synonym replacement.
- Use the frozen calibrated threshold from `shared/calibration.json`.
- Do not regenerate text.
- Do not require CUDA for the attack or detection pass.
- Real runs require NLTK WordNet data; tests use injected synonym providers.
- Preserve the project's existing `known_boundary` and `blind_text` detection conventions.

---

## File Structure

- Create `dual_layer_watermark_fixed_length_error_erasure/experiments/attack/__init__.py`
  - Marks the attack package and exports no runtime side effects.
- Create `dual_layer_watermark_fixed_length_error_erasure/experiments/attack/synonym_attack.py`
  - Owns attack kinds, word spans, synonym provider protocol, WordNet provider, candidate filtering, deterministic edit selection, and attack diagnostics.
- Create `dual_layer_watermark_fixed_length_error_erasure/experiments/attack/run_synonym_attacks.py`
  - Owns CLI parsing, path resolution, experiment row loading, detector construction, detection record writing, metric summarization, ROC CSV writing, and ROC plotting.
- Create `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_synonym_attacks.py`
  - Covers attack-core behavior, dependency behavior, row loading, metrics, and a CLI smoke path with fakes.
- Modify `dual_layer_watermark_fixed_length_error_erasure/experiments/README.md`
  - Adds the command for the new attack experiment and notes the WordNet requirement.

---

### Task 1: Attack Core Spans And Candidate Classification

**Files:**
- Create: `dual_layer_watermark_fixed_length_error_erasure/experiments/attack/__init__.py`
- Create: `dual_layer_watermark_fixed_length_error_erasure/experiments/attack/synonym_attack.py`
- Test: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_synonym_attacks.py`

**Interfaces:**
- Produces: `AttackKind = Literal["replacement", "deletion", "insertion"]`
- Produces: `WordSpan(text: str, start: int, end: int)`
- Produces: `Candidate(word: str, replacement: str, delta_tokens: int, attacked_text: str)`
- Produces: `collect_word_spans(text: str) -> list[WordSpan]`
- Produces: `replace_span(text: str, span: WordSpan, replacement: str) -> str`
- Produces: `candidate_matches_attack(delta_tokens: int, attack: AttackKind) -> bool`
- Produces: `candidate_for_span(text: str, span: WordSpan, replacement: str, tokenizer: Any, original_token_count: int | None = None) -> Candidate`
- Produces: `filter_candidates(text: str, span: WordSpan, replacements: Sequence[str], tokenizer: Any, attack: AttackKind, original_token_count: int | None = None) -> list[Candidate]`

- [ ] **Step 1: Write failing span and classification tests**

Add these tests to `tests/experiments/test_synonym_attacks.py`:

```python
import pytest

from experiments.attack.synonym_attack import (
    candidate_for_span,
    candidate_matches_attack,
    collect_word_spans,
    filter_candidates,
    replace_span,
)


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class LengthMapTokenizer:
    lengths = {
        "a small house": 3,
        "a tiny house": 3,
        "a house": 2,
        "a very small house": 4,
    }

    def encode(self, text, add_special_tokens=False):
        return list(range(self.lengths[text]))


def test_collect_word_spans_preserves_text_reconstruction():
    text = "Hello, watermarked-world! It's stable."
    spans = collect_word_spans(text)

    assert [span.text for span in spans] == ["Hello", "watermarked-world", "It's", "stable"]
    assert "".join(text[span.start : span.end] for span in spans) == "Hellowatermarked-worldIt'sstable"


def test_replace_span_changes_only_selected_word():
    text = "a small house"
    span = collect_word_spans(text)[1]

    assert replace_span(text, span, "tiny") == "a tiny house"


def test_candidate_matching_by_token_delta():
    assert candidate_matches_attack(0, "replacement") is True
    assert candidate_matches_attack(-1, "deletion") is True
    assert candidate_matches_attack(1, "insertion") is True
    assert candidate_matches_attack(1, "replacement") is False


def test_filter_candidates_uses_full_continuation_token_delta():
    text = "a small house"
    span = collect_word_spans(text)[1]
    tokenizer = LengthMapTokenizer()

    assert [c.replacement for c in filter_candidates(text, span, ["tiny", "very small"], tokenizer, "replacement")] == ["tiny"]
    assert [c.replacement for c in filter_candidates(text, span, ["house"], tokenizer, "deletion")] == ["house"]
    assert [c.replacement for c in filter_candidates(text, span, ["very small"], tokenizer, "insertion")] == ["very small"]


def test_candidate_for_span_records_attacked_text_and_delta():
    text = "a small house"
    span = collect_word_spans(text)[1]

    candidate = candidate_for_span(text, span, "tiny", LengthMapTokenizer(), original_token_count=3)

    assert candidate.word == "small"
    assert candidate.replacement == "tiny"
    assert candidate.attacked_text == "a tiny house"
    assert candidate.delta_tokens == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
pytest tests/experiments/test_synonym_attacks.py -q
```

Expected: import failure for `experiments.attack.synonym_attack`.

- [ ] **Step 3: Implement attack core span and candidate functions**

Create `experiments/attack/__init__.py`:

```python
"""Attack experiment utilities for the fixed-length error-erasure pipeline."""
```

Create the initial `experiments/attack/synonym_attack.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

AttackKind = Literal["replacement", "deletion", "insertion"]

_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")


@dataclass(frozen=True)
class WordSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Candidate:
    word: str
    replacement: str
    delta_tokens: int
    attacked_text: str


def collect_word_spans(text: str) -> list[WordSpan]:
    return [WordSpan(match.group(0), match.start(), match.end()) for match in _WORD_RE.finditer(text)]


def replace_span(text: str, span: WordSpan, replacement: str) -> str:
    return text[: span.start] + replacement + text[span.end :]


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def candidate_matches_attack(delta_tokens: int, attack: AttackKind) -> bool:
    if attack == "replacement":
        return delta_tokens == 0
    if attack == "deletion":
        return delta_tokens < 0
    if attack == "insertion":
        return delta_tokens > 0
    raise ValueError(f"Unsupported attack kind: {attack}")


def candidate_for_span(
    text: str,
    span: WordSpan,
    replacement: str,
    tokenizer: Any,
    *,
    original_token_count: int | None = None,
) -> Candidate:
    attacked_text = replace_span(text, span, replacement)
    before = _token_count(tokenizer, text) if original_token_count is None else int(original_token_count)
    after = _token_count(tokenizer, attacked_text)
    return Candidate(
        word=span.text,
        replacement=replacement,
        delta_tokens=after - before,
        attacked_text=attacked_text,
    )


def filter_candidates(
    text: str,
    span: WordSpan,
    replacements: Sequence[str],
    tokenizer: Any,
    attack: AttackKind,
    *,
    original_token_count: int | None = None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for replacement in replacements:
        normalized = replacement.strip().replace("_", " ")
        if not normalized or normalized.lower() == span.text.lower() or normalized in seen:
            continue
        seen.add(normalized)
        candidate = candidate_for_span(
            text,
            span,
            normalized,
            tokenizer,
            original_token_count=original_token_count,
        )
        if candidate_matches_attack(candidate.delta_tokens, attack):
            candidates.append(candidate)
    return candidates
```

- [ ] **Step 4: Run tests to verify Task 1 passes**

Run:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
pytest tests/experiments/test_synonym_attacks.py -q
```

Expected: Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add dual_layer_watermark_fixed_length_error_erasure/experiments/attack/__init__.py \
  dual_layer_watermark_fixed_length_error_erasure/experiments/attack/synonym_attack.py \
  dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_synonym_attacks.py
git commit -m "Add synonym attack core"
```

---

### Task 2: Deterministic Synonym Attack Assembly And WordNet Provider

**Files:**
- Modify: `dual_layer_watermark_fixed_length_error_erasure/experiments/attack/synonym_attack.py`
- Test: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_synonym_attacks.py`

**Interfaces:**
- Consumes: `AttackKind`, `WordSpan`, `Candidate`, `collect_word_spans`, `filter_candidates`
- Produces: `SynonymProvider` protocol with `synonyms(word: str) -> Sequence[str]`
- Produces: `AttackResult(original_text: str, attacked_text: str, attacked_token_ids: list[int], original_token_count: int, attacked_token_count: int, editable_word_count: int, target_edit_count: int, achieved_edit_count: int, achieved_attack_rate: float, token_length_delta: int, edits: list[AppliedEdit])`
- Produces: `AppliedEdit(word: str, replacement: str, start: int, end: int, delta_tokens: int)`
- Produces: `target_edit_count(editable_word_count: int, attack_rate: float) -> int`
- Produces: `attack_text(text: str, tokenizer: Any, provider: SynonymProvider, attack: AttackKind, attack_rate: float, seed: int, require_full_rate: bool = False) -> AttackResult`
- Produces: `WordNetSynonymProvider`

- [ ] **Step 1: Write failing deterministic attack and provider tests**

Append these tests:

```python
from experiments.attack.synonym_attack import (
    WordNetSynonymProvider,
    attack_text,
    target_edit_count,
)


class StaticProvider:
    def __init__(self, mapping):
        self.mapping = mapping

    def synonyms(self, word):
        return self.mapping.get(word.lower(), [])


def test_target_edit_count_rounds_and_handles_empty_text():
    assert target_edit_count(0, 0.10) == 0
    assert target_edit_count(3, 0.10) == 1
    assert target_edit_count(20, 0.10) == 2


def test_attack_text_is_deterministic_and_reports_diagnostics():
    provider = StaticProvider({"small": ["tiny"], "house": ["home"]})

    left = attack_text("a small house", LengthMapTokenizer(), provider, "replacement", attack_rate=0.50, seed=7)
    right = attack_text("a small house", LengthMapTokenizer(), provider, "replacement", attack_rate=0.50, seed=7)

    assert left == right
    assert left.attacked_text in {"a tiny house", "a small home"}
    assert left.original_token_count == 3
    assert left.attacked_token_count == 3
    assert left.target_edit_count == 2
    assert left.achieved_edit_count == 1
    assert left.achieved_attack_rate == pytest.approx(1 / 3)
    assert left.token_length_delta == 0
    assert len(left.attacked_token_ids) == 3


def test_attack_text_require_full_rate_raises_when_candidates_are_missing():
    provider = StaticProvider({"small": ["tiny"]})

    with pytest.raises(RuntimeError, match="Could only apply 1 synonym edits"):
        attack_text("a small house", LengthMapTokenizer(), provider, "replacement", attack_rate=0.90, seed=0, require_full_rate=True)


def test_wordnet_provider_wraps_missing_resource(monkeypatch):
    provider = WordNetSynonymProvider()

    def raise_lookup(_word):
        raise LookupError("missing wordnet")

    monkeypatch.setattr(provider, "_synsets", raise_lookup)

    with pytest.raises(RuntimeError, match="NLTK WordNet data is required"):
        provider.synonyms("good")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
pytest tests/experiments/test_synonym_attacks.py -q
```

Expected: missing `attack_text`, `target_edit_count`, or `WordNetSynonymProvider`.

- [ ] **Step 3: Implement deterministic attack assembly**

Add to `experiments/attack/synonym_attack.py`:

```python
import random
from typing import Protocol


class SynonymProvider(Protocol):
    def synonyms(self, word: str) -> Sequence[str]:
        ...


@dataclass(frozen=True)
class AppliedEdit:
    word: str
    replacement: str
    start: int
    end: int
    delta_tokens: int


@dataclass(frozen=True)
class AttackResult:
    original_text: str
    attacked_text: str
    attacked_token_ids: list[int]
    original_token_count: int
    attacked_token_count: int
    editable_word_count: int
    target_edit_count: int
    achieved_edit_count: int
    achieved_attack_rate: float
    token_length_delta: int
    edits: list[AppliedEdit]


def target_edit_count(editable_word_count: int, attack_rate: float) -> int:
    if editable_word_count <= 0:
        return 0
    if not 0.0 <= attack_rate <= 1.0:
        raise ValueError("attack_rate must be in [0, 1]")
    return max(1, int(round(editable_word_count * attack_rate)))


def _preserve_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def attack_text(
    text: str,
    tokenizer: Any,
    provider: SynonymProvider,
    attack: AttackKind,
    *,
    attack_rate: float,
    seed: int,
    require_full_rate: bool = False,
) -> AttackResult:
    spans = collect_word_spans(text)
    original_token_count = _token_count(tokenizer, text)
    target = target_edit_count(len(spans), attack_rate)
    rng = random.Random(int(seed))
    shuffled = list(spans)
    rng.shuffle(shuffled)

    current_text = text
    offset = 0
    edits: list[AppliedEdit] = []
    edited_ranges: set[tuple[int, int]] = set()

    for original_span in shuffled:
        if len(edits) >= target:
            break
        if (original_span.start, original_span.end) in edited_ranges:
            continue
        span = WordSpan(
            text=current_text[original_span.start + offset : original_span.end + offset],
            start=original_span.start + offset,
            end=original_span.end + offset,
        )
        replacements = [_preserve_case(span.text, value) for value in provider.synonyms(original_span.text)]
        candidates = filter_candidates(
            current_text,
            span,
            replacements,
            tokenizer,
            attack,
            original_token_count=_token_count(tokenizer, current_text),
        )
        if not candidates:
            continue
        chosen = candidates[rng.randrange(len(candidates))]
        current_text = chosen.attacked_text
        offset += len(chosen.replacement) - len(span.text)
        edited_ranges.add((original_span.start, original_span.end))
        edits.append(
            AppliedEdit(
                word=span.text,
                replacement=chosen.replacement,
                start=span.start,
                end=span.end,
                delta_tokens=chosen.delta_tokens,
            )
        )

    if require_full_rate and len(edits) < target:
        raise RuntimeError(f"Could only apply {len(edits)} synonym edits out of requested {target}")

    attacked_token_ids = [int(value) for value in tokenizer.encode(current_text, add_special_tokens=False)]
    attacked_token_count = len(attacked_token_ids)
    return AttackResult(
        original_text=text,
        attacked_text=current_text,
        attacked_token_ids=attacked_token_ids,
        original_token_count=original_token_count,
        attacked_token_count=attacked_token_count,
        editable_word_count=len(spans),
        target_edit_count=target,
        achieved_edit_count=len(edits),
        achieved_attack_rate=0.0 if not spans else len(edits) / len(spans),
        token_length_delta=attacked_token_count - original_token_count,
        edits=edits,
    )
```

- [ ] **Step 4: Implement WordNet provider with clear missing-data error**

Add to `experiments/attack/synonym_attack.py`:

```python
class WordNetSynonymProvider:
    def _synsets(self, word: str):
        from nltk.corpus import wordnet as wn

        return wn.synsets(word)

    def synonyms(self, word: str) -> list[str]:
        try:
            synsets = self._synsets(word)
        except LookupError as exc:
            raise RuntimeError(
                "NLTK WordNet data is required for paper-aligned synonym attacks. "
                "Run: python -m nltk.downloader wordnet omw-1.4"
            ) from exc
        values: list[str] = []
        seen: set[str] = set()
        for synset in synsets:
            for lemma in synset.lemma_names():
                value = lemma.replace("_", " ").strip()
                key = value.lower()
                if value and key != word.lower() and key not in seen:
                    seen.add(key)
                    values.append(value)
        return values
```

- [ ] **Step 5: Run tests to verify Task 2 passes**

Run:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
pytest tests/experiments/test_synonym_attacks.py -q
```

Expected: Task 1 and Task 2 tests pass.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add dual_layer_watermark_fixed_length_error_erasure/experiments/attack/synonym_attack.py \
  dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_synonym_attacks.py
git commit -m "Add deterministic synonym attack assembly"
```

---

### Task 3: CLI, Path Resolution, And Experiment Row Loading

**Files:**
- Create: `dual_layer_watermark_fixed_length_error_erasure/experiments/attack/run_synonym_attacks.py`
- Test: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_synonym_attacks.py`

**Interfaces:**
- Consumes: `AttackKind`
- Produces: `AttackExample(sample_id: str, split: str, text_class: str, prompt_token_ids: list[int], text: str, token_ids: list[int], message_bits: str | None, encoded_bits: str | None)`
- Produces: `build_parser() -> argparse.ArgumentParser`
- Produces: `project_dir() -> Path`
- Produces: `resolve_input_path(value: str | Path) -> Path`
- Produces: `resolve_output_path(value: str | Path) -> Path`
- Produces: `default_output_dir(experiment_dir: Path, point_id: str, attack_rate: float) -> Path`
- Produces: `load_attack_examples(experiment_dir: Path, point_id: str, max_samples: int | None = None) -> list[AttackExample]`

- [ ] **Step 1: Write failing CLI and loader tests**

Append these tests:

```python
import json
import subprocess
import sys
from pathlib import Path

from experiments.attack.run_synonym_attacks import (
    build_parser,
    default_output_dir,
    load_attack_examples,
    resolve_input_path,
    resolve_output_path,
)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_attack_parser_defaults_match_confirmed_experiment():
    args = build_parser().parse_args(["--experiment-dir", "outputs/experiments/opt13b_pareto_49x200"])

    assert args.point_id == "soft_p0_m2"
    assert args.attack_rate == 0.10
    assert args.attacks == ["replacement", "deletion", "insertion"]
    assert args.headline_input_mode == "known_boundary"


def test_default_output_dir_uses_rate_slug():
    root = Path("/tmp/project/outputs/experiments/opt13b_pareto_49x200")

    assert default_output_dir(root, "soft_p0_m2", 0.10) == root / "attacks" / "soft_p0_m2_synonym_rate10"


def test_attack_paths_resolve_under_project_when_called_from_repo_root(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    project = repo_root / "dual_layer_watermark_fixed_length_error_erasure"
    monkeypatch.chdir(repo_root)

    assert resolve_input_path("outputs/experiments/x") == project / "outputs/experiments/x"
    assert resolve_output_path("outputs/experiments/x/attacks/y") == project / "outputs/experiments/x/attacks/y"


def test_load_attack_examples_uses_only_test_split(tmp_path):
    experiment_dir = tmp_path / "exp"
    write_jsonl(
        experiment_dir / "runs" / "soft_p0_m2" / "watermarked.jsonl",
        [
            {"sample_id": "0", "split": "calibration", "prompt_token_ids": [1], "watermarked_text": "cal", "watermarked_token_ids": [2], "message_bits": "00000000", "encoded_bits": "0"},
            {"sample_id": "1", "split": "test", "prompt_token_ids": [1], "watermarked_text": "wm", "watermarked_token_ids": [3], "message_bits": "11111111", "encoded_bits": "1"},
        ],
    )
    write_jsonl(
        experiment_dir / "shared" / "baseline.jsonl",
        [
            {"sample_id": "1", "split": "test", "prompt_token_ids": [1], "unwatermarked_text": "uw", "unwatermarked_token_ids": [4], "natural_text": "nat", "natural_token_ids": [5]},
        ],
    )

    examples = load_attack_examples(experiment_dir, "soft_p0_m2")

    assert [(row.sample_id, row.text_class, row.text) for row in examples] == [
        ("1", "watermarked", "wm"),
        ("1", "unwatermarked", "uw"),
        ("1", "natural", "nat"),
    ]


def test_attack_module_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        [sys.executable, "-m", "experiments.attack.run_synonym_attacks", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--attack-rate" in completed.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
pytest tests/experiments/test_synonym_attacks.py -q
```

Expected: missing `experiments.attack.run_synonym_attacks`.

- [ ] **Step 3: Implement parser, paths, and row loading**

Create `experiments/attack/run_synonym_attacks.py` with:

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from utils.io import iter_jsonl

from experiments.attack.synonym_attack import AttackKind

DEFAULT_ATTACKS: tuple[AttackKind, ...] = ("replacement", "deletion", "insertion")


@dataclass(frozen=True)
class AttackExample:
    sample_id: str
    split: str
    text_class: str
    prompt_token_ids: list[int]
    text: str
    token_ids: list[int]
    message_bits: str | None
    encoded_bits: str | None


def project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path, *, for_output: bool) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    base = project_dir()
    return base / path


def resolve_input_path(value: str | Path) -> Path:
    return _resolve_path(value, for_output=False)


def resolve_output_path(value: str | Path) -> Path:
    return _resolve_path(value, for_output=True)


def _rate_slug(attack_rate: float) -> str:
    return f"{attack_rate * 100:g}".replace(".", "p")


def default_output_dir(experiment_dir: Path, point_id: str, attack_rate: float) -> Path:
    return experiment_dir / "attacks" / f"{point_id}_synonym_rate{_rate_slug(attack_rate)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper-aligned synonym substitution attacks for the dual-layer watermark")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--point-id", default="soft_p0_m2")
    parser.add_argument("--attack-rate", type=float, default=0.10)
    parser.add_argument("--attacks", nargs="+", choices=list(DEFAULT_ATTACKS), default=list(DEFAULT_ATTACKS))
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--headline-input-mode", choices=["known_boundary", "blind_text"], default="known_boundary")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-full-rate", action="store_true")
    return parser


def _ints(values: Sequence[int]) -> list[int]:
    return [int(value) for value in values]


def load_attack_examples(experiment_dir: Path, point_id: str, max_samples: int | None = None) -> list[AttackExample]:
    watermarked_path = experiment_dir / "runs" / point_id / "watermarked.jsonl"
    baseline_path = experiment_dir / "shared" / "baseline.jsonl"
    examples: list[AttackExample] = []

    for row in iter_jsonl(watermarked_path):
        if row.get("split") != "test":
            continue
        examples.append(
            AttackExample(
                sample_id=str(row["sample_id"]),
                split="test",
                text_class="watermarked",
                prompt_token_ids=_ints(row["prompt_token_ids"]),
                text=str(row["watermarked_text"]),
                token_ids=_ints(row["watermarked_token_ids"]),
                message_bits=str(row["message_bits"]),
                encoded_bits=str(row["encoded_bits"]),
            )
        )
        if max_samples is not None and len([x for x in examples if x.text_class == "watermarked"]) >= max_samples:
            break

    selected_ids = {row.sample_id for row in examples if row.text_class == "watermarked"}
    for row in iter_jsonl(baseline_path):
        if row.get("split") != "test" or str(row["sample_id"]) not in selected_ids:
            continue
        for text_class, text_field, token_field in (
            ("unwatermarked", "unwatermarked_text", "unwatermarked_token_ids"),
            ("natural", "natural_text", "natural_token_ids"),
        ):
            examples.append(
                AttackExample(
                    sample_id=str(row["sample_id"]),
                    split="test",
                    text_class=text_class,
                    prompt_token_ids=_ints(row["prompt_token_ids"]),
                    text=str(row[text_field]),
                    token_ids=_ints(row[token_field]),
                    message_bits=None,
                    encoded_bits=None,
                )
            )
    return examples


def main(argv: Sequence[str] | None = None) -> None:
    build_parser().parse_args(argv)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify Task 3 passes**

Run:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
pytest tests/experiments/test_synonym_attacks.py -q
python -m experiments.attack.run_synonym_attacks --help
```

Expected: tests pass and help prints attack CLI options.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add dual_layer_watermark_fixed_length_error_erasure/experiments/attack/run_synonym_attacks.py \
  dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_synonym_attacks.py
git commit -m "Add synonym attack CLI loading"
```

---

### Task 4: Detection, Metrics, Outputs, And ROC Figures

**Files:**
- Modify: `dual_layer_watermark_fixed_length_error_erasure/experiments/attack/run_synonym_attacks.py`
- Test: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_synonym_attacks.py`

**Interfaces:**
- Consumes: `AttackExample`, `attack_text`, `AttackResult`, existing `detection_record`
- Produces: `build_runtime_config(experiment_dir: Path, calibrated_threshold: float) -> ExperimentConfig`
- Produces: `detect_attacked_example(example: AttackExample, result: AttackResult, detector: Any, runtime_config: ExperimentConfig, attack: AttackKind, elapsed_attack_seconds: float) -> list[dict[str, Any]]`
- Produces: `evaluate_attack(detections: Sequence[dict[str, Any]], attacked_records: Sequence[dict[str, Any]], threshold: float, input_mode: str) -> dict[str, Any]`
- Produces: `write_roc_points(path: Path, roc: dict[str, list[float]]) -> None`
- Produces: `plot_rocs(metrics_by_attack: Mapping[str, dict[str, Any]], output_dir: Path) -> None`
- Produces: `run(args: argparse.Namespace) -> dict[str, Any]`

- [ ] **Step 1: Write failing metrics and detection tests**

Append these tests:

```python
from experiments.attack.run_synonym_attacks import (
    evaluate_attack,
    write_roc_points,
)


def test_evaluate_attack_uses_frozen_threshold_and_reports_diagnostics(tmp_path):
    detections = [
        {"sample_id": "1", "text_class": "watermarked", "input_mode": "known_boundary", "z_score": 3.0, "message_bits": "10101010", "decoded_message": "10101010", "encoded_bits": "111", "recovered_codeword_bits": "111", "strict": {"status": "decoded", "message_bits": "10101010"}, "hard_fill": {"status": "decoded", "message_bits": "10101010"}, "error_erasure": {"status": "decoded", "message_bits": "10101010"}, "decode_status": "decoded"},
        {"sample_id": "1", "text_class": "unwatermarked", "input_mode": "known_boundary", "z_score": 2.0, "strict": {}, "hard_fill": {}, "error_erasure": {}},
        {"sample_id": "1", "text_class": "natural", "input_mode": "known_boundary", "z_score": 0.0, "strict": {}, "hard_fill": {}, "error_erasure": {}},
    ]
    attacked = [
        {"sample_id": "1", "text_class": "watermarked", "achieved_edit_count": 2, "target_edit_count": 2, "achieved_attack_rate": 0.1, "token_length_delta": 0},
        {"sample_id": "1", "text_class": "unwatermarked", "achieved_edit_count": 1, "target_edit_count": 2, "achieved_attack_rate": 0.05, "token_length_delta": 0},
        {"sample_id": "1", "text_class": "natural", "achieved_edit_count": 0, "target_edit_count": 2, "achieved_attack_rate": 0.0, "token_length_delta": 0},
    ]

    metrics = evaluate_attack(detections, attacked, threshold=2.5, input_mode="known_boundary")

    assert metrics["presence"]["tpr"] == 1.0
    assert metrics["presence"]["model_fpr"] == 0.0
    assert metrics["presence"]["natural_fpr"] == 0.0
    assert metrics["payload"]["exact_message_recovery"] == 1.0
    assert metrics["attack"]["mean_achieved_attack_rate"] == pytest.approx(0.05)
    assert metrics["attack"]["mean_candidate_coverage"] == pytest.approx(0.5)


def test_write_roc_points_creates_csv(tmp_path):
    output = tmp_path / "roc_points.csv"

    write_roc_points(output, {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [float("inf"), 0.1]})

    text = output.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "fpr,tpr,threshold"
    assert "0.0,1.0,0.1" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
pytest tests/experiments/test_synonym_attacks.py -q
```

Expected: missing `evaluate_attack` or `write_roc_points`.

- [ ] **Step 3: Implement runtime config, detection helpers, metrics, and files**

Add these imports and functions to `experiments/attack/run_synonym_attacks.py`:

```python
import csv
import math
import time
from collections.abc import Mapping
from statistics import mean
from typing import Any

from evaluation.metrics import evaluate_records
from experiments.config import ParetoExperimentConfig
from experiments.detection import _runtime_config
from utils.detection import build_detector, detection_record, load_tokenizer
from utils.io import append_jsonl, read_json, write_json

from experiments.attack.synonym_attack import AttackResult, WordNetSynonymProvider, attack_text


def build_pareto_config(experiment_dir: Path) -> ParetoExperimentConfig:
    payload = read_json(experiment_dir / "experiment.json")["config"]
    values = dict(payload)
    values["output_root"] = experiment_dir.parent
    values["experiment_id"] = experiment_dir.name
    return ParetoExperimentConfig(**{key: value for key, value in values.items() if key in ParetoExperimentConfig.__dataclass_fields__})


def build_runtime_config(experiment_dir: Path, calibrated_threshold: float):
    return _runtime_config(build_pareto_config(experiment_dir), calibrated_threshold)


def _timed_detect(function: Any, *args: Any):
    started = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - started


def _attack_record(example: AttackExample, attack: AttackKind, result: AttackResult, seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sample_id": example.sample_id,
        "split": example.split,
        "text_class": example.text_class,
        "attack": attack,
        "attack_seed": seed,
        "original_text": result.original_text,
        "attacked_text": result.attacked_text,
        "original_token_ids": example.token_ids,
        "attacked_token_ids": result.attacked_token_ids,
        "original_token_count": result.original_token_count,
        "attacked_token_count": result.attacked_token_count,
        "editable_word_count": result.editable_word_count,
        "target_edit_count": result.target_edit_count,
        "achieved_edit_count": result.achieved_edit_count,
        "achieved_attack_rate": result.achieved_attack_rate,
        "token_length_delta": result.token_length_delta,
        "edits": [edit.__dict__ for edit in result.edits],
    }


def detect_attacked_example(
    example: AttackExample,
    result: AttackResult,
    detector: Any,
    runtime_config: Any,
    attack: AttackKind,
    *,
    attack_seed: int,
    attack_seconds: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for detection_result, elapsed in (
        _timed_detect(detector.detect_continuation, example.prompt_token_ids, result.attacked_token_ids),
        _timed_detect(detector.detect_token_ids, result.attacked_token_ids),
    ):
        record = detection_record(
            config=runtime_config,
            detector=detector,
            sample_id=example.sample_id,
            text_class=example.text_class,
            result=detection_result,
            elapsed=elapsed,
            message_bits=example.message_bits,
            encoded_bits=example.encoded_bits,
        )
        record.update(
            {
                "split": example.split,
                "attack": attack,
                "attack_seed": attack_seed,
                "attack_seconds": attack_seconds,
                "original_token_count": result.original_token_count,
                "attacked_token_count": result.attacked_token_count,
                "target_edit_count": result.target_edit_count,
                "achieved_edit_count": result.achieved_edit_count,
                "achieved_attack_rate": result.achieved_attack_rate,
                "token_length_delta": result.token_length_delta,
            }
        )
        records.append(record)
    return records


def _attack_diagnostics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    target_total = sum(int(row["target_edit_count"]) for row in records)
    achieved_total = sum(int(row["achieved_edit_count"]) for row in records)
    return {
        "count": len(records),
        "target_edit_count": target_total,
        "achieved_edit_count": achieved_total,
        "mean_achieved_attack_rate": mean(float(row["achieved_attack_rate"]) for row in records),
        "mean_candidate_coverage": None if target_total == 0 else achieved_total / target_total,
        "mean_token_length_delta": mean(float(row["token_length_delta"]) for row in records),
    }


def evaluate_attack(
    detections: Sequence[dict[str, Any]],
    attacked_records: Sequence[dict[str, Any]],
    *,
    threshold: float,
    input_mode: str,
) -> dict[str, Any]:
    frozen: list[dict[str, Any]] = []
    for row in detections:
        updated = dict(row)
        updated["threshold"] = threshold
        updated["detected"] = float(updated.get("z_score", 0.0)) >= threshold
        frozen.append(updated)
    metrics = evaluate_records(frozen, input_mode=input_mode)
    return {
        "schema_version": 1,
        "presence": metrics["presence"],
        "presence_by_input_mode": metrics["presence_by_input_mode"],
        "payload": metrics["payload"],
        "payload_by_input_mode": metrics["payload_by_input_mode"],
        "runtime": metrics.get("runtime", {}),
        "attack": _attack_diagnostics(attacked_records),
    }


def write_roc_points(path: Path, roc: Mapping[str, Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fpr", "tpr", "threshold"])
        for fpr, tpr, threshold in zip(roc.get("fpr", []), roc.get("tpr", []), roc.get("thresholds", []), strict=True):
            writer.writerow([fpr, tpr, threshold])
```

- [ ] **Step 4: Implement `run` orchestration and summary CSV**

Extend `run_synonym_attacks.py`:

```python
def _prepare_output(path: Path, *, resume: bool, overwrite: bool) -> None:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if path.exists() and any(path.iterdir()) and not (resume or overwrite):
        raise FileExistsError(f"Output directory exists: {path}. Use --resume or --overwrite.")
    path.mkdir(parents=True, exist_ok=True)


def _sample_seed(base_seed: int, attack: str, sample_id: str, text_class: str) -> int:
    return abs(hash((int(base_seed), attack, sample_id, text_class))) % (2**32)


def _write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "attack",
        "tpr",
        "model_fpr",
        "natural_fpr",
        "auc",
        "exact_message_recovery",
        "message_ber",
        "mean_achieved_attack_rate",
        "mean_candidate_coverage",
        "mean_token_length_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def run(args: argparse.Namespace) -> dict[str, Any]:
    experiment_dir = resolve_input_path(args.experiment_dir)
    output_dir = resolve_output_path(args.output_dir) if args.output_dir else default_output_dir(experiment_dir, args.point_id, args.attack_rate)
    _prepare_output(output_dir, resume=args.resume, overwrite=args.overwrite)

    calibration = read_json(experiment_dir / "shared" / "calibration.json")
    threshold = float(calibration["calibrated_threshold"])
    runtime_config = build_runtime_config(experiment_dir, threshold)
    tokenizer, vocab_size = load_tokenizer(runtime_config.model.path, local_files_only=runtime_config.model.local_files_only)
    detector = build_detector(runtime_config, tokenizer, int(vocab_size))
    provider = WordNetSynonymProvider()
    examples = load_attack_examples(experiment_dir, args.point_id, max_samples=args.max_samples)

    metadata = {
        "schema_version": 1,
        "experiment_dir": str(experiment_dir),
        "point_id": args.point_id,
        "attack_rate": args.attack_rate,
        "attacks": list(args.attacks),
        "calibrated_threshold": threshold,
        "headline_input_mode": args.headline_input_mode,
        "max_samples": args.max_samples,
        "seed": args.seed,
    }
    write_json(output_dir / "metadata.json", metadata)

    summary_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"schema_version": 1, "metadata": metadata, "attacks": {}}
    for attack in args.attacks:
        attack_dir = output_dir / attack
        attacked_records: list[dict[str, Any]] = []
        detection_records: list[dict[str, Any]] = []
        for example in examples:
            seed = _sample_seed(args.seed, attack, example.sample_id, example.text_class)
            started = time.perf_counter()
            attack_result = attack_text(
                example.text,
                tokenizer,
                provider,
                attack,
                attack_rate=args.attack_rate,
                seed=seed,
                require_full_rate=args.require_full_rate,
            )
            attack_seconds = time.perf_counter() - started
            attacked_record = _attack_record(example, attack, attack_result, seed)
            append_jsonl(attack_dir / "attacked_records.jsonl", attacked_record)
            attacked_records.append(attacked_record)
            for detection in detect_attacked_example(
                example,
                attack_result,
                detector,
                runtime_config,
                attack,
                attack_seed=seed,
                attack_seconds=attack_seconds,
            ):
                append_jsonl(attack_dir / "detections.jsonl", detection)
                detection_records.append(detection)

        metrics = evaluate_attack(
            detection_records,
            attacked_records,
            threshold=threshold,
            input_mode=args.headline_input_mode,
        )
        write_json(attack_dir / "metrics.json", metrics)
        write_roc_points(attack_dir / "roc_points.csv", metrics["presence"]["roc"])
        summary["attacks"][attack] = metrics
        summary_rows.append(
            {
                "attack": attack,
                "tpr": metrics["presence"]["tpr"],
                "model_fpr": metrics["presence"]["model_fpr"],
                "natural_fpr": metrics["presence"]["natural_fpr"],
                "auc": metrics["presence"]["auc"],
                "exact_message_recovery": metrics["payload"]["exact_message_recovery"],
                "message_ber": metrics["payload"]["message_ber"],
                "mean_achieved_attack_rate": metrics["attack"]["mean_achieved_attack_rate"],
                "mean_candidate_coverage": metrics["attack"]["mean_candidate_coverage"],
                "mean_token_length_delta": metrics["attack"]["mean_token_length_delta"],
            }
        )
    write_json(output_dir / "summary.json", summary)
    _write_summary_csv(output_dir / "summary.csv", summary_rows)
    plot_rocs(summary["attacks"], output_dir)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)
```

- [ ] **Step 5: Implement ROC plotting**

Add:

```python
def plot_rocs(metrics_by_attack: Mapping[str, dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.8, 5.2))
    for attack, metrics in metrics_by_attack.items():
        roc = metrics["presence"]["roc"]
        axis.plot(roc.get("fpr", []), roc.get("tpr", []), marker="o", linewidth=1.8, label=attack)
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1.0)
    axis.set_xlabel("FPR")
    axis.set_ylabel("TPR")
    axis.set_title("Synonym substitution attacks")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "roc_synonym_attacks.png", dpi=180)
    figure.savefig(figures / "roc_synonym_attacks.pdf")
    plt.close(figure)
```

- [ ] **Step 6: Run tests to verify Task 4 passes**

Run:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
pytest tests/experiments/test_synonym_attacks.py -q
```

Expected: Task 1 through Task 4 tests pass.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add dual_layer_watermark_fixed_length_error_erasure/experiments/attack/run_synonym_attacks.py \
  dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_synonym_attacks.py
git commit -m "Add synonym attack detection metrics"
```

---

### Task 5: CLI Smoke, README, And Real-Run Verification Path

**Files:**
- Modify: `dual_layer_watermark_fixed_length_error_erasure/experiments/attack/run_synonym_attacks.py`
- Modify: `dual_layer_watermark_fixed_length_error_erasure/experiments/README.md`
- Test: `dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_synonym_attacks.py`

**Interfaces:**
- Consumes: `run(args)`, `build_parser()`
- Produces: command documentation for the real run
- Produces: smoke invocation instructions that do not require CUDA

- [ ] **Step 1: Add overwrite cleanup behavior test**

Append:

```python
from experiments.attack.run_synonym_attacks import _prepare_output


def test_prepare_output_rejects_existing_directory_without_resume_or_overwrite(tmp_path):
    output = tmp_path / "attack"
    output.mkdir()
    (output / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Output directory exists"):
        _prepare_output(output, resume=False, overwrite=False)


def test_prepare_output_accepts_overwrite_directory(tmp_path):
    output = tmp_path / "attack"
    output.mkdir()
    (output / "metadata.json").write_text("{}", encoding="utf-8")

    _prepare_output(output, resume=False, overwrite=True)

    assert output.exists()
```

- [ ] **Step 2: Run tests to verify overwrite behavior test fails if needed**

Run:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
pytest tests/experiments/test_synonym_attacks.py::test_prepare_output_accepts_overwrite_directory -q
```

Expected: fail if `_prepare_output` leaves stale files or has incompatible visibility.

- [ ] **Step 3: Make overwrite behavior deterministic**

Modify `_prepare_output`:

```python
import shutil


def _prepare_output(path: Path, *, resume: bool, overwrite: bool) -> None:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()) and not resume:
        raise FileExistsError(f"Output directory exists: {path}. Use --resume or --overwrite.")
    path.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Document the attack runner**

Add this section to `experiments/README.md` after the Pareto/ROC section:

````markdown
## Synonym Substitution Attack Experiments

The paper-aligned attack runner evaluates 10% word-level synonym substitution and groups results by tokenizer-level effect:

- `replacement`: token-preserving substitutions
- `deletion`: token-reducing substitutions
- `insertion`: token-increasing substitutions

Real runs require NLTK WordNet data:

```bash
conda run -n BREW python -m nltk.downloader wordnet omw-1.4
```

Run the default `soft_p0_m2` attack suite:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
export NUMBA_CACHE_DIR=/tmp/numba-cache
conda run -n BREW python -m experiments.attack.run_synonym_attacks \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --point-id soft_p0_m2 \
  --attack-rate 0.10 \
  --attacks replacement deletion insertion \
  --overwrite
```

Outputs are written to:

```text
outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_synonym_rate10/
```
````

- [ ] **Step 5: Run focused tests and CLI help**

Run:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
pytest tests/experiments/test_synonym_attacks.py -q
python -m experiments.attack.run_synonym_attacks --help
```

Expected: tests pass and help succeeds.

- [ ] **Step 6: Run a real smoke command if WordNet data exists**

Run:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
export NUMBA_CACHE_DIR=/tmp/numba-cache
conda run -n BREW python -m experiments.attack.run_synonym_attacks \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --point-id soft_p0_m2 \
  --attack-rate 0.10 \
  --attacks replacement \
  --max-samples 3 \
  --overwrite
```

Expected if WordNet data exists: `summary.json`, `summary.csv`, `replacement/metrics.json`, and `figures/roc_synonym_attacks.png` are created under the default attack output directory.

Expected if WordNet data is missing: command exits with `NLTK WordNet data is required for paper-aligned synonym attacks. Run: python -m nltk.downloader wordnet omw-1.4`.

- [ ] **Step 7: Commit Task 5**

Run:

```bash
git add dual_layer_watermark_fixed_length_error_erasure/experiments/attack/run_synonym_attacks.py \
  dual_layer_watermark_fixed_length_error_erasure/experiments/README.md \
  dual_layer_watermark_fixed_length_error_erasure/tests/experiments/test_synonym_attacks.py
git commit -m "Document synonym attack runner"
```

---

## Final Verification

- [ ] Run the focused attack test file:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
pytest tests/experiments/test_synonym_attacks.py -q
```

- [ ] Run related existing experiment tests:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
pytest tests/experiments/test_mpac_comparison.py tests/experiments/test_segment_rsbh_comparison.py tests/test_metrics.py -q
```

- [ ] Verify CLI help from the project directory:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
python -m experiments.attack.run_synonym_attacks --help
```

- [ ] Verify CLI help from the repository root:

```bash
python -m experiments.attack.run_synonym_attacks --help
```

- [ ] Run the real attack suite after WordNet data is present:

```bash
cd dual_layer_watermark_fixed_length_error_erasure
export NUMBA_CACHE_DIR=/tmp/numba-cache
conda run -n BREW python -m experiments.attack.run_synonym_attacks \
  --experiment-dir outputs/experiments/opt13b_pareto_49x200 \
  --point-id soft_p0_m2 \
  --attack-rate 0.10 \
  --attacks replacement deletion insertion \
  --overwrite
```

Expected output files:

```text
outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_synonym_rate10/metadata.json
outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_synonym_rate10/summary.json
outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_synonym_rate10/summary.csv
outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_synonym_rate10/replacement/metrics.json
outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_synonym_rate10/deletion/metrics.json
outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_synonym_rate10/insertion/metrics.json
outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_synonym_rate10/figures/roc_synonym_attacks.png
outputs/experiments/opt13b_pareto_49x200/attacks/soft_p0_m2_synonym_rate10/figures/roc_synonym_attacks.pdf
```

## Self-Review

- Spec coverage: the plan covers paper-aligned synonym attacks, three attack classes, default `soft_p0_m2`, existing generated outputs, frozen calibration threshold, metrics, ROC files, WordNet dependency behavior, tests, and README command documentation.
- Placeholder scan: no placeholder markers or unresolved requirements remain.
- Type consistency: `AttackKind`, `AttackResult`, `AttackExample`, `evaluate_attack`, `write_roc_points`, and `run(args)` signatures are defined before they are consumed by subsequent tasks.
