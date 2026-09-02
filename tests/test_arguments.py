from dataclasses import FrozenInstanceError

import pytest

from experiments.arguments import config_from_args, experiment_parser
from utils.arguments import generation_config_from_args, generation_parser
from watermark.config import ExperimentConfig
from watermark.execution import BatchExecutionConfig


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
