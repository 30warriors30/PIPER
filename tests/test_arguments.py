from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys

import pytest

import run_detection
from experiments.arguments import config_from_args, experiment_parser
from utils.arguments import generation_config_from_args, generation_parser
from watermark.config import ExperimentConfig
from watermark.execution import BatchExecutionConfig


ROOT = Path(__file__).resolve().parents[1]


def test_generation_parser_builds_config() -> None:
    args = generation_parser().parse_args([
        "--model-path", "/tmp/model", "--secret-key", "secret", "--run-id", "test",
        "--ecc-n", "23", "--ecc-k", "8", "--ecc-t", "3",
    ])
    config = generation_config_from_args(args)
    assert config.ecc.k == 8
    assert config.dataset.kind == "c4"
    assert config.output.run_id == "test"
    assert config.watermark.presence_mode == "soft"
    assert config.watermark.delta_presence == 0.0
    assert config.watermark.delta_payload == 2.0
    assert config.detection.presence_test == "exact_binomial"
    assert config.detection.counting_mode == "unique_context"
    assert config.decoding.primary_policy == "tie_zero"
    assert config.decoding.max_erasure_assignments == 64
    assert config.generation.top_k == 50
    assert config.watermark.candidate_top_k == 50
    assert config.watermark.seeding_scheme == "selfhash"
    assert config.watermark.partition_engine == "v2"


def test_batch_execution_defaults_and_validation() -> None:
    value = BatchExecutionConfig()

    assert value.engine_version == 2
    assert value.generation_batch_size == 16
    assert value.detection_batch_size == 64
    assert value.detection_workers == 8
    with pytest.raises(FrozenInstanceError):
        value.generation_batch_size = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="generation_batch_size must be positive"):
        BatchExecutionConfig(generation_batch_size=0)
    with pytest.raises(ValueError, match="detection_batch_size must be positive"):
        BatchExecutionConfig(detection_batch_size=0)
    with pytest.raises(ValueError, match="detection_workers must be positive"):
        BatchExecutionConfig(detection_workers=0)
    with pytest.raises(ValueError, match="engine_version must be 2"):
        BatchExecutionConfig(engine_version=1)


def test_generation_parser_exposes_batch_defaults() -> None:
    args = generation_parser().parse_args(
        ["--model-path", "model", "--secret-key", "key", "--run-id", "run"]
    )

    assert args.generation_batch_size == 16
    assert args.detection_batch_size == 64
    assert args.detection_workers == 8
    assert args.import_v1_completed is False

    config = generation_config_from_args(args)
    assert config.execution.generation_batch_size == 16
    assert config.execution.detection_batch_size == 64
    assert config.execution.detection_workers == 8
    assert config.to_dict()["execution"] == {
        "generation_batch_size": 16,
        "detection_batch_size": 64,
        "detection_workers": 8,
        "engine_version": 2,
    }
    assert ExperimentConfig.from_dict(config.to_dict()).execution == config.execution


def test_pareto_parser_passes_batch_overrides_to_config() -> None:
    args = experiment_parser().parse_args(
        [
            "--model-path",
            "model",
            "--secret-key",
            "key",
            "--experiment-id",
            "experiment",
            "--generation-batch-size",
            "3",
            "--detection-batch-size",
            "5",
            "--detection-workers",
            "2",
        ]
    )

    config = config_from_args(args)

    assert config.generation_batch_size == 3
    assert config.detection_batch_size == 5
    assert config.detection_workers == 2
    assert config.top_k == 50
    assert config.candidate_top_k == 50
    assert config.seeding_scheme == "selfhash"
    assert config.partition_engine == "v2"


def test_old_experiment_dictionary_loads_with_legacy_partition_semantics() -> None:
    args = generation_parser().parse_args(
        ["--model-path", "model", "--secret-key", "key", "--run-id", "run"]
    )
    data = generation_config_from_args(args).to_dict()
    data["generation"].pop("top_k")
    data["watermark"].pop("candidate_top_k")
    data["watermark"].pop("seeding_scheme")
    data["watermark"].pop("partition_engine")

    restored = ExperimentConfig.from_dict(data)

    assert restored.generation.top_k is None
    assert restored.watermark.candidate_top_k is None
    assert restored.watermark.seeding_scheme == "history"
    assert restored.watermark.partition_engine == "v1"


def test_detection_entrypoint_rejects_import_without_resume() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_detection.py"),
            "--input-type",
            "samples",
            "--run-dir",
            "run",
            "--dry-run",
            "--import-v1-completed",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "--import-v1-completed requires --resume" in completed.stderr


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--generation-batch-size", "0", "generation_batch_size must be positive"),
        ("--detection-batch-size", "-1", "detection_batch_size must be positive"),
        ("--detection-workers", "0", "detection_workers must be positive"),
    ],
)
def test_detection_entrypoint_rejects_non_positive_execution_values(
    option: str,
    value: str,
    message: str,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_detection.py"),
            "--input-type",
            "samples",
            "--run-dir",
            "run",
            "--dry-run",
            option,
            value,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert message in completed.stderr


@pytest.mark.parametrize(
    ("runner_name", "mode_args"),
    [
        ("run_samples_detection", ["--input-type", "samples", "--run-dir", "run"]),
        (
            "run_text_detection",
            [
                "--input-type",
                "text",
                "--input-file",
                "input.jsonl",
                "--output-dir",
                "output",
                "--model-path",
                "model",
                "--secret-key",
                "key",
            ],
        ),
    ],
)
def test_detection_entrypoint_threads_execution_overrides(
    monkeypatch: pytest.MonkeyPatch,
    runner_name: str,
    mode_args: list[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_runner(*args: object, **kwargs: object) -> dict[str, int]:
        captured.update(kwargs)
        return {"failed": 0}

    monkeypatch.setattr(run_detection, runner_name, fake_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_detection.py",
            *mode_args,
            "--generation-batch-size",
            "3",
            "--detection-batch-size",
            "5",
            "--detection-workers",
            "2",
        ],
    )

    assert run_detection.main() == 0
    assert captured["execution"] == BatchExecutionConfig(
        generation_batch_size=3,
        detection_batch_size=5,
        detection_workers=2,
    )
    if runner_name == "run_text_detection":
        assert captured["seeding_scheme"] == "selfhash"
        assert captured["partition_engine"] == "v2"
