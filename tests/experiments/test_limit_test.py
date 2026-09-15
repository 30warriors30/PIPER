import json

import pytest

from experiments.arguments import config_from_args, experiment_parser
from experiments.config import ParetoExperimentConfig
from experiments.manifest import load_selected_test_rows, selected_test_sample_ids
from experiments.mpac_comparison import load_manifest_rows as load_mpac_manifest_rows


def _config(tmp_path, *, test_samples=5, limit_test=2):
    return ParetoExperimentConfig(
        model_path="/tmp/model",
        secret_key="secret",
        output_root=tmp_path,
        experiment_id="limit-test",
        calibration_samples=1,
        test_samples=test_samples,
        limit_test=limit_test,
    )


def test_cli_parses_limit_test_for_positive_watermarked_samples(tmp_path) -> None:
    args = experiment_parser().parse_args(
        [
            "--model-path",
            "/tmp/model",
            "--secret-key",
            "secret",
            "--experiment-id",
            "limit-test",
            "--output-root",
            str(tmp_path),
            "--calibration-samples",
            "1000",
            "--test-samples",
            "5000",
            "--limit-test",
            "1000",
        ]
    )

    config = config_from_args(args)

    assert config.test_samples == 5000
    assert config.limit_test == 1000
    assert config.watermarked_test_samples == 1000
    assert config.total_manifest_samples == 6000


def test_limit_test_must_not_exceed_test_samples(tmp_path) -> None:
    with pytest.raises(ValueError, match="limit_test must not exceed test_samples"):
        _config(tmp_path, test_samples=2, limit_test=3)


def test_selected_test_rows_use_manifest_order_and_leave_negatives_unlimited(tmp_path) -> None:
    config = _config(tmp_path)
    config.experiment_dir.mkdir(parents=True)
    rows = [
        {"sample_id": "cal-0", "split": "calibration", "status": "completed"},
        {"sample_id": "test-0", "split": "test", "status": "completed"},
        {"sample_id": "test-1", "split": "test", "status": "completed"},
        {"sample_id": "test-2", "split": "test", "status": "completed"},
    ]
    config.experiment_dir.joinpath("manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    selected = load_selected_test_rows(config)

    assert [row["sample_id"] for row in selected] == ["test-0", "test-1"]
    assert selected_test_sample_ids(config) == {"test-0", "test-1"}


def test_piper_and_mpac_select_the_same_positive_manifest_rows(tmp_path) -> None:
    config = _config(tmp_path)
    config.experiment_dir.mkdir(parents=True)
    rows = [
        {"sample_id": "cal-0", "split": "calibration", "status": "completed"},
        {"sample_id": "test-0", "split": "test", "status": "completed"},
        {"sample_id": "test-1", "split": "test", "status": "completed"},
        {"sample_id": "test-2", "split": "test", "status": "completed"},
    ]
    manifest = config.experiment_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    piper_rows = load_selected_test_rows(config)
    mpac_rows = load_mpac_manifest_rows(
        manifest,
        split="test",
        limit=config.watermarked_test_samples,
    )

    assert [row["sample_id"] for row in piper_rows] == [
        row["sample_id"] for row in mpac_rows
    ]
