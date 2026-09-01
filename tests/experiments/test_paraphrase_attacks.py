from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from experiments.attack.run_paraphrase_attacks import (
    ParaphraseResult,
    Seq2SeqParaphraser,
    _bounded_token_limit,
    build_parser,
    default_output_dir,
    detect_paraphrased_example,
    evaluate_paraphrase_attack,
    make_paraphrase_result,
)
from experiments.attack.run_synonym_attacks import AttackExample


class SpaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return list(range(len(text.split())))


class FakeDetector:
    def __init__(self):
        self.calls = []

    def detect_continuation(self, prompt_token_ids, continuation_token_ids, *, decode_payload):
        self.calls.append(("known_boundary", list(prompt_token_ids), list(continuation_token_ids), decode_payload))
        return {"z_score": 3.0, "payload": "ok"}

    def detect_token_ids(self, token_ids, *, decode_payload):
        self.calls.append(("blind_text", list(token_ids), decode_payload))
        return {"z_score": 1.0, "payload": "ok"}


def _example(text_class: str = "watermarked") -> AttackExample:
    return AttackExample(
        sample_id="7",
        split="test",
        text_class=text_class,
        prompt_token_ids=[10, 11],
        text="original text",
        token_ids=[1, 2],
        message_bits="10101010" if text_class == "watermarked" else None,
        encoded_bits="10101010101010101010101" if text_class == "watermarked" else None,
    )


def test_paraphrase_parser_defaults_match_confirmed_experiment():
    args = build_parser().parse_args([])

    assert args.experiment_dir == "outputs/experiments/opt13b_pareto_49x200"
    assert args.point_id == "soft_p0_m2"
    assert args.attack == "paraphrase"
    assert args.paraphraser_slug == "pegasus"
    assert args.headline_input_mode == "known_boundary"
    assert args.input_modes == ["known_boundary", "blind_text"]


def test_paraphrase_default_output_dir_uses_point_and_model_slug():
    root = Path("outputs/experiments/demo")

    assert default_output_dir(root, "soft_p0_m2", "pegasus") == root / "attacks" / "soft_p0_m2_paraphrase_pegasus"


def test_bounded_token_limit_clamps_requested_value_to_model_limit():
    assert _bounded_token_limit(80, 60, "--max-output-tokens") == 60
    assert _bounded_token_limit(32, 60, "--max-output-tokens") == 32


def test_make_paraphrase_result_records_tokens_and_length_diagnostics():
    result = make_paraphrase_result(
        "one two three four",
        "one altered sentence",
        SpaceTokenizer(),
        paraphrase_seconds=0.25,
        model_path="/models/paraphraser",
        generation={"num_beams": 4, "max_new_tokens": 32},
    )

    assert isinstance(result, ParaphraseResult)
    assert result.original_token_count == 4
    assert result.attacked_token_ids == [0, 1, 2]
    assert result.attacked_token_count == 3
    assert result.token_length_delta == -1
    assert result.compression_ratio == pytest.approx(0.75)
    assert result.paraphraser_model == "/models/paraphraser"
    assert result.generation == {"num_beams": 4, "max_new_tokens": 32}


def test_paraphraser_chunks_split_single_sentence_above_token_limit():
    paraphraser = object.__new__(Seq2SeqParaphraser)
    paraphraser.tokenizer = SpaceTokenizer()
    paraphraser.max_input_tokens = 3

    chunks = paraphraser._chunks("one two three four five. six seven.")

    assert chunks == ["one two three", "four five.", "six seven."]
    assert all(len(chunk.split()) <= 3 for chunk in chunks)


def test_detect_paraphrased_example_respects_input_modes(monkeypatch):
    captured = []

    def fake_detection_record(**kwargs):
        captured.append(kwargs)
        return {
            "sample_id": kwargs["sample_id"],
            "text_class": kwargs["text_class"],
            "z_score": kwargs["result"]["z_score"],
            "message_bits": kwargs["message_bits"],
            "encoded_bits": kwargs["encoded_bits"],
        }

    monkeypatch.setattr("experiments.attack.run_paraphrase_attacks.detection_record", fake_detection_record)
    detector = FakeDetector()
    result = make_paraphrase_result("original text", "rewritten text now", SpaceTokenizer(), 0.5, "/model", {})

    records = detect_paraphrased_example(
        _example("watermarked"),
        result,
        detector,
        runtime_config=object(),
        attack="paraphrase",
        attack_seed=123,
        input_modes=["known_boundary", "blind_text"],
    )

    assert [row["input_mode"] for row in records] == ["known_boundary", "blind_text"]
    assert detector.calls == [
        ("known_boundary", [10, 11], [0, 1, 2], True),
        ("blind_text", [0, 1, 2], True),
    ]
    assert all(row["attack"] == "paraphrase" for row in records)
    assert all(row["attack_seed"] == 123 for row in records)
    assert all(row["paraphrase_seconds"] == pytest.approx(0.5) for row in records)
    assert captured[0]["message_bits"] == "10101010"


def test_detect_paraphrased_example_skips_payload_decode_for_negative_text(monkeypatch):
    decode_values = []

    class Detector:
        def detect_continuation(self, prompt_token_ids, continuation_token_ids, *, decode_payload):
            decode_values.append(decode_payload)
            return {"z_score": 0.0}

    monkeypatch.setattr(
        "experiments.attack.run_paraphrase_attacks.detection_record",
        lambda **kwargs: {"z_score": kwargs["result"]["z_score"], "text_class": kwargs["text_class"]},
    )
    result = make_paraphrase_result("original text", "rewritten text", SpaceTokenizer(), 0.1, "/model", {})

    detect_paraphrased_example(
        _example("unwatermarked"),
        result,
        Detector(),
        runtime_config=object(),
        attack="paraphrase",
        attack_seed=11,
        input_modes=["known_boundary"],
    )

    assert decode_values == [False]


def test_evaluate_paraphrase_attack_uses_frozen_threshold_and_reports_diagnostics():
    detections = [
        {"sample_id": "1", "text_class": "watermarked", "input_mode": "known_boundary", "z_score": 3.0},
        {"sample_id": "1", "text_class": "unwatermarked", "input_mode": "known_boundary", "z_score": 1.0},
        {"sample_id": "1", "text_class": "natural", "input_mode": "known_boundary", "z_score": 2.0},
    ]
    attacked = [
        {
            "sample_id": "1",
            "text_class": "watermarked",
            "original_token_count": 10,
            "attacked_token_count": 8,
            "token_length_delta": -2,
            "compression_ratio": 0.9,
            "paraphrase_seconds": 0.1,
        },
        {
            "sample_id": "1",
            "text_class": "unwatermarked",
            "original_token_count": 10,
            "attacked_token_count": 11,
            "token_length_delta": 1,
            "compression_ratio": 1.1,
            "paraphrase_seconds": 0.2,
        },
        {
            "sample_id": "1",
            "text_class": "natural",
            "original_token_count": 10,
            "attacked_token_count": 10,
            "token_length_delta": 0,
            "compression_ratio": 1.0,
            "paraphrase_seconds": 0.3,
        },
    ]

    metrics = evaluate_paraphrase_attack(detections, attacked, threshold=2.5, input_mode="known_boundary")

    assert metrics["presence"]["tpr"] == 1.0
    assert metrics["presence"]["model_fpr"] == 0.0
    assert metrics["presence"]["natural_fpr"] == 0.0
    assert metrics["attack"]["count"] == 3
    assert metrics["attack"]["mean_token_length_delta"] == pytest.approx(-1 / 3)
    assert metrics["attack"]["mean_compression_ratio"] == pytest.approx(1.0)
    assert metrics["attack"]["mean_paraphrase_seconds"] == pytest.approx(0.2)


def test_paraphrase_module_help_runs_from_repo_root():
    project = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "-m", "experiments.attack.run_paraphrase_attacks", "--help"],
        cwd=project,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0
    assert "--paraphraser-model" in completed.stdout
    assert "--max-samples" in completed.stdout
