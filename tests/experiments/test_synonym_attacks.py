import json
import subprocess
import sys
from pathlib import Path

import pytest

from watermark.result_types import CountingResult, DecodeResult

from experiments.attack.run_synonym_attacks import (
    _prepare_output,
    _sample_seed,
    _write_jsonl,
    build_parser,
    build_runtime_config,
    default_output_dir,
    detect_attacked_example,
    evaluate_attack,
    load_attack_examples,
    resolve_input_path,
    resolve_output_path,
    validate_input_modes,
    write_roc_points,
)
from experiments.config import ParetoExperimentConfig

from experiments.attack.synonym_attack import (
    AttackResult,
    WordNetSynonymProvider,
    attack_text,
    candidate_for_span,
    candidate_matches_attack,
    collect_word_spans,
    filter_candidates,
    replace_span,
    target_edit_count,
)


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class WhitespaceIdTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


class WordIdTokenizer:
    token_ids = {
        "a": 1,
        "small": 2,
        "tiny": 3,
        "house": 4,
    }

    def encode(self, text, add_special_tokens=False):
        return [self.token_ids[word] for word in text.split()]


class LengthMapTokenizer:
    lengths = {
        "a small house": 3,
        "a tiny house": 3,
        "a home house": 2,
        "a very small house": 4,
    }

    def encode(self, text, add_special_tokens=False):
        return list(range(self.lengths[text]))


class LengthRecordingTokenizer:
    def __init__(self):
        self.max_seen_length = 0

    def encode(self, text, add_special_tokens=False):
        self.max_seen_length = max(self.max_seen_length, len(text))
        return text.split()


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
    assert [c.replacement for c in filter_candidates(text, span, ["home"], tokenizer, "deletion")] == ["home"]
    assert [c.replacement for c in filter_candidates(text, span, ["very small"], tokenizer, "insertion")] == ["very small"]


def test_filter_candidates_computes_candidate_delta_from_local_window():
    prefix = " ".join(f"prefix{i}" for i in range(100))
    suffix = " ".join(f"suffix{i}" for i in range(100))
    text = f"{prefix} small {suffix}"
    span = collect_word_spans(text)[100]
    tokenizer = LengthRecordingTokenizer()

    candidates = filter_candidates(text, span, ["tiny"], tokenizer, "replacement")

    assert [candidate.replacement for candidate in candidates] == ["tiny"]
    assert tokenizer.max_seen_length < len(text) // 2


def test_candidate_for_span_records_attacked_text_and_delta():
    text = "a small house"
    span = collect_word_spans(text)[1]

    candidate = candidate_for_span(text, span, "tiny", LengthMapTokenizer(), original_token_count=3)

    assert candidate.word == "small"
    assert candidate.replacement == "tiny"
    assert candidate.attacked_text == "a tiny house"
    assert candidate.delta_tokens == 0


class StaticProvider:
    def __init__(self, mapping):
        self.mapping = mapping

    def synonyms(self, word):
        return self.mapping.get(word.lower(), [])


def test_target_edit_count_rounds_and_handles_empty_text():
    assert target_edit_count(0, 0.10) == 0
    assert target_edit_count(3, 0.0) == 0
    assert target_edit_count(3, 0.10) == 1
    assert target_edit_count(20, 0.10) == 2


def test_attack_text_with_zero_rate_leaves_nonempty_text_and_token_ids_unchanged():
    text = "a small house"
    tokenizer = WordIdTokenizer()
    provider = StaticProvider({"small": ["tiny"]})

    result = attack_text(text, tokenizer, provider, "replacement", attack_rate=0.0, seed=7)

    assert result.attacked_text == text
    assert result.attacked_token_ids == [1, 2, 4]
    assert result.target_edit_count == 0
    assert result.achieved_edit_count == 0
    assert result.edits == []


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


def test_attack_text_resolves_original_spans_when_edits_run_out_of_order():
    provider = StaticProvider({"alpha": ["a"], "gamma": ["long"]})

    result = attack_text(
        "alpha beta gamma",
        WhitespaceIdTokenizer(),
        provider,
        "replacement",
        attack_rate=0.67,
        seed=7,
        require_full_rate=True,
    )

    assert result.attacked_text == "a beta long"
    assert result.achieved_edit_count == 2
    assert [edit.word for edit in result.edits] == ["gamma", "alpha"]
    assert [edit.replacement for edit in result.edits] == ["long", "a"]


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


def test_wordnet_provider_caches_synset_lookup(monkeypatch):
    provider = WordNetSynonymProvider()
    calls = []

    class Synset:
        def lemma_names(self):
            return ["excellent", "good"]

    def synsets(word):
        calls.append(word)
        return [Synset()]

    monkeypatch.setattr(provider, "_synsets", synsets)

    assert provider.synonyms("good") == ["excellent"]
    assert provider.synonyms("good") == ["excellent"]
    assert calls == ["good"]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_attack_parser_defaults_match_confirmed_experiment():
    args = build_parser().parse_args(["--experiment-dir", "outputs/experiments/opt13b_pareto_49x200"])

    assert args.point_id == "soft_p0_m2"
    assert args.attack_rate == 0.10
    assert args.attacks == ["replacement", "deletion", "insertion"]
    assert args.headline_input_mode == "known_boundary"
    assert args.input_modes == ["known_boundary", "blind_text"]
    assert args.evaluate_all_policies is False


def test_attack_parser_accepts_single_input_mode():
    args = build_parser().parse_args(["--input-modes", "known_boundary"])

    assert args.input_modes == ["known_boundary"]


def test_attack_parser_accepts_policy_breakdown_flag():
    args = build_parser().parse_args(["--evaluate-all-policies"])

    assert args.evaluate_all_policies is True


def test_validate_input_modes_rejects_missing_headline_mode():
    with pytest.raises(ValueError, match="headline input mode"):
        validate_input_modes("known_boundary", ["blind_text"])


def test_validate_input_modes_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        validate_input_modes("known_boundary", ["known_boundary", "known_boundary"])


def test_default_output_dir_uses_rate_slug():
    root = Path("/tmp/project/outputs/experiments/opt13b_pareto_49x200")

    assert default_output_dir(root, "soft_p0_m2", 0.10) == root / "attacks" / "soft_p0_m2_synonym_rate10"


def test_attack_paths_resolve_under_project_when_called_from_repo_root(monkeypatch):
    project = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(project.parent)

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


def test_load_attack_examples_with_zero_max_samples_returns_no_examples(tmp_path):
    experiment_dir = tmp_path / "exp"
    write_jsonl(
        experiment_dir / "runs" / "soft_p0_m2" / "watermarked.jsonl",
        [{"sample_id": "1", "split": "test", "prompt_token_ids": [1], "watermarked_text": "wm", "watermarked_token_ids": [3], "message_bits": "1", "encoded_bits": "0"}],
    )

    assert load_attack_examples(experiment_dir, "soft_p0_m2", max_samples=0) == []


def test_load_attack_examples_preserves_null_watermarked_metadata(tmp_path):
    experiment_dir = tmp_path / "exp"
    write_jsonl(
        experiment_dir / "runs" / "soft_p0_m2" / "watermarked.jsonl",
        [{"sample_id": "1", "split": "test", "prompt_token_ids": [1], "watermarked_text": "wm", "watermarked_token_ids": [3], "message_bits": None, "encoded_bits": None}],
    )

    examples = load_attack_examples(experiment_dir, "soft_p0_m2")

    assert examples[0].message_bits is None
    assert examples[0].encoded_bits is None


def test_attack_module_help_runs_from_repo_root():
    project_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "-m", "experiments.attack.run_synonym_attacks", "--help"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--attack-rate" in completed.stdout


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
    assert "1.0,1.0,0.1" in text


def test_write_jsonl_replaces_file_with_compact_records(tmp_path):
    output = tmp_path / "records.jsonl"
    output.write_text('{"stale":true}\n', encoding="utf-8")

    _write_jsonl(output, [{"sample_id": "1", "score": 1}, {"sample_id": "2", "score": 0}])

    assert output.read_text(encoding="utf-8").splitlines() == [
        '{"sample_id":"1","score":1}',
        '{"sample_id":"2","score":0}',
    ]


def test_detect_attacked_example_respects_requested_input_modes(monkeypatch):
    calls = []

    class Detector:
        def detect_continuation(self, prompt_ids, token_ids, *, decode_payload=True):
            calls.append(("known_boundary", list(prompt_ids), list(token_ids), decode_payload))
            return type("Result", (), {"input_mode": "known_boundary"})()

        def detect_token_ids(self, token_ids, *, decode_payload=True):
            raise AssertionError("blind_text should not be called")

    def fake_detection_record(**kwargs):
        return {
            "sample_id": kwargs["sample_id"],
            "text_class": kwargs["text_class"],
            "input_mode": kwargs["result"].input_mode,
        }

    monkeypatch.setattr("experiments.attack.run_synonym_attacks.detection_record", fake_detection_record)
    example = type(
        "Example",
        (),
        {
            "sample_id": "1",
            "split": "test",
            "text_class": "watermarked",
            "prompt_token_ids": [10],
            "message_bits": "10101010",
            "encoded_bits": "111",
        },
    )()
    result = AttackResult(
        original_text="a small house",
        attacked_text="a tiny house",
        attacked_token_ids=[1, 2, 3],
        original_token_count=3,
        attacked_token_count=3,
        editable_word_count=3,
        target_edit_count=1,
        achieved_edit_count=1,
        achieved_attack_rate=1 / 3,
        token_length_delta=0,
        edits=[],
    )

    records = detect_attacked_example(
        example,
        result,
        Detector(),
        runtime_config=object(),
        attack="replacement",
        attack_seed=123,
        attack_seconds=0.5,
        input_modes=("known_boundary",),
    )

    assert calls == [("known_boundary", [10], [1, 2, 3], True)]
    assert [row["input_mode"] for row in records] == ["known_boundary"]


def test_detect_attacked_example_skips_payload_decode_for_negative_text(monkeypatch):
    calls = []

    class Detector:
        def detect_continuation(self, prompt_ids, token_ids, *, decode_payload=True):
            calls.append(decode_payload)
            return type("Result", (), {"input_mode": "known_boundary"})()

    monkeypatch.setattr(
        "experiments.attack.run_synonym_attacks.detection_record",
        lambda **kwargs: {"input_mode": kwargs["result"].input_mode},
    )
    example = type(
        "Example",
        (),
        {
            "sample_id": "1",
            "split": "test",
            "text_class": "natural",
            "prompt_token_ids": [10],
            "message_bits": None,
            "encoded_bits": None,
        },
    )()
    result = AttackResult(
        original_text="a small house",
        attacked_text="a tiny house",
        attacked_token_ids=[1, 2, 3],
        original_token_count=3,
        attacked_token_count=3,
        editable_word_count=3,
        target_edit_count=1,
        achieved_edit_count=1,
        achieved_attack_rate=1 / 3,
        token_length_delta=0,
        edits=[],
    )

    detect_attacked_example(
        example,
        result,
        Detector(),
        runtime_config=object(),
        attack="replacement",
        attack_seed=123,
        attack_seconds=0.5,
        input_modes=("known_boundary",),
    )

    assert calls == [False]


def test_detection_record_does_not_recover_codeword_for_not_evaluated_decode(monkeypatch):
    from experiments.attack.run_synonym_attacks import detection_record

    class Detector:
        def recover_codeword(self, *_args, **_kwargs):
            raise AssertionError("not_evaluated decode should not recover codeword")

    result = type(
        "Result",
        (),
        {
            "input_mode": "known_boundary",
            "threshold": 0.0,
            "detected": False,
            "primary_counting_mode": "all_tokens",
            "gated_message_bits": None,
            "strict_decode": DecodeResult(status="not_evaluated", message_bits=None, corrected_errors=None),
            "hard_fill_decode": DecodeResult(status="not_evaluated", message_bits=None, corrected_errors=None),
            "error_erasure_decode": DecodeResult(status="not_evaluated", message_bits=None, corrected_errors=None),
            "counting": {
                "all_tokens": CountingResult(
                    mode="all_tokens",
                    scored_tokens=1,
                    upper_hits=0,
                    z_score=0.0,
                    exact_p_value=1.0,
                    n0=(0,),
                    n1=(0,),
                    erasures=(0,),
                )
            },
        },
    )()
    config = type(
        "Config",
        (),
        {
            "decoding": type("Decoding", (), {"primary_policy": "strict"})(),
        },
    )()

    record = detection_record(
        config=config,
        detector=Detector(),
        sample_id="1",
        text_class="natural",
        result=result,
        elapsed=0.1,
        message_bits=None,
        encoded_bits=None,
    )

    assert record["recovered_codeword_bits"] is None
    assert record["decode_status"] == "not_evaluated"


def test_build_runtime_config_forces_cpu_for_attack_detection(monkeypatch, tmp_path):
    pareto = ParetoExperimentConfig(
        model_path="/models/opt",
        secret_key="test-key",
        output_root=tmp_path,
        experiment_id="exp",
        device="cuda",
    )
    monkeypatch.setattr(
        "experiments.attack.run_synonym_attacks.build_pareto_config",
        lambda _experiment_dir: pareto,
    )

    runtime = build_runtime_config(tmp_path / "exp", calibrated_threshold=2.5)

    assert runtime.model.device == "cpu"
    assert runtime.decoding.evaluate_all_policies is False


def test_build_runtime_config_can_keep_policy_breakdown(monkeypatch, tmp_path):
    pareto = ParetoExperimentConfig(
        model_path="/models/opt",
        secret_key="test-key",
        output_root=tmp_path,
        experiment_id="exp",
        device="cuda",
    )
    monkeypatch.setattr(
        "experiments.attack.run_synonym_attacks.build_pareto_config",
        lambda _experiment_dir: pareto,
    )

    runtime = build_runtime_config(tmp_path / "exp", calibrated_threshold=2.5, evaluate_all_policies=True)

    assert runtime.decoding.evaluate_all_policies is True


def test_sample_seed_is_stable_across_python_processes():
    code = (
        "from experiments.attack.run_synonym_attacks import _sample_seed; "
        "print(_sample_seed(20260813, 'replacement', '17', 'watermarked'))"
    )
    project_dir = Path(__file__).resolve().parents[2]

    first = subprocess.check_output([sys.executable, "-c", code], cwd=project_dir, text=True)
    second = subprocess.check_output([sys.executable, "-c", code], cwd=project_dir, text=True)

    assert first == second
    assert int(first) == _sample_seed(20260813, "replacement", "17", "watermarked")


def test_prepare_output_overwrite_removes_previous_records(tmp_path):
    output = tmp_path / "attacks"
    stale = output / "replacement" / "attacked_records.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"sample_id":"stale"}\n', encoding="utf-8")

    _prepare_output(output, resume=False, overwrite=True)

    assert output.exists()
    assert list(output.iterdir()) == []


def test_prepare_output_rejects_existing_directory_without_resume_or_overwrite(tmp_path):
    output = tmp_path / "attack"
    output.mkdir()
    (output / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Output directory exists"):
        _prepare_output(output, resume=False, overwrite=False)


def test_prepare_output_rejects_unimplemented_resume(tmp_path):
    with pytest.raises(NotImplementedError, match="resume is not supported"):
        _prepare_output(tmp_path / "attacks", resume=True, overwrite=False)
