import pytest
from utils.arguments import generation_config_from_args, generation_parser
from utils.validation import validate_experiment_config


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
