from argparse import Namespace

import pytest
from utils.arguments import generation_config_from_args, generation_parser
from utils.validation import validate_batch_execution_args, validate_experiment_config


def config(extra: list[str] | None = None):
    args = ["--model-path", "/tmp/model", "--secret-key", "secret", "--run-id", "test"]
    args.extend(extra or [])
    return generation_config_from_args(generation_parser().parse_args(args))


def test_balanced_allocation_is_reserved() -> None:
    with pytest.raises(NotImplementedError):
        validate_experiment_config(config(["--allocation-mode", "balanced_permutation"]), check_model_path=False)


def test_default_bch_is_valid() -> None:
    codec = validate_experiment_config(config(), check_model_path=False)
    assert (codec.n, codec.k, codec.t) == (23, 8, 3)


def test_import_v1_completed_requires_resume() -> None:
    with pytest.raises(ValueError, match="--import-v1-completed requires --resume"):
        validate_batch_execution_args(
            Namespace(import_v1_completed=True, resume=False)
        )

    validate_batch_execution_args(Namespace(import_v1_completed=True, resume=True))
