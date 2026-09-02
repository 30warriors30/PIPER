from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from experiments.config import ParetoExperimentConfig
from experiments.detection import _runtime_config


ROOT = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path, *, presence_test: str) -> ParetoExperimentConfig:
    return ParetoExperimentConfig(
        model_path="model",
        secret_key="secret",
        output_root=tmp_path,
        experiment_id="experiment",
        calibration_samples=2,
        test_samples=2,
        presence_test=presence_test,
    )


def _write_negative_scores(config: ParetoExperimentConfig) -> None:
    output = config.experiment_dir / "shared" / "negative_detections.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "sample_id": str(index),
            "split": "calibration",
            "input_mode": "known_boundary",
            "text_class": text_class,
            "z_score": score,
        }
        for index, (text_class, score) in enumerate(
            (("unwatermarked", -0.5), ("natural", 1.25))
        )
    ]
    output.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_exact_binomial_skips_shared_z_calibration(tmp_path: Path) -> None:
    from experiments.detection import calibrate_shared_z_threshold

    config = _config(tmp_path, presence_test="exact_binomial")
    _write_negative_scores(config)

    result = calibrate_shared_z_threshold(config)

    assert result is None
    assert not (config.experiment_dir / "shared" / "z_calibration.json").exists()


def test_z_score_writes_and_loads_shared_z_calibration(tmp_path: Path) -> None:
    from experiments.detection import (
        calibrate_shared_z_threshold,
        load_shared_z_threshold,
    )

    config = _config(tmp_path, presence_test="z_score")
    _write_negative_scores(config)

    result = calibrate_shared_z_threshold(config)
    threshold = load_shared_z_threshold(config)

    assert result is not None
    assert result["score_type"] == "z_score"
    assert threshold == result["calibrated_threshold"]
    assert (config.experiment_dir / "shared" / "z_calibration.json").exists()


def test_exact_binomial_runtime_does_not_require_calibrated_threshold(tmp_path: Path) -> None:
    config = _config(tmp_path, presence_test="exact_binomial")

    runtime = _runtime_config(config, calibrated_threshold=None)

    assert runtime.detection.presence_test == "exact_binomial"
    assert runtime.detection.threshold_mode == "theoretical"
    assert runtime.detection.calibrated_threshold is None
    assert runtime.execution.engine_version == 2
    assert runtime.execution.generation_batch_size == 16
    assert runtime.execution.detection_batch_size == 64
    assert runtime.execution.detection_workers == 8


def test_exact_binomial_attack_does_not_require_z_calibration_file(tmp_path: Path) -> None:
    from experiments.attack.run_synonym_attacks import load_attack_z_threshold

    config = _config(tmp_path, presence_test="exact_binomial")
    config.experiment_dir.mkdir(parents=True)
    (config.experiment_dir / "experiment.json").write_text(
        json.dumps({"config": config.to_dict()}),
        encoding="utf-8",
    )

    assert load_attack_z_threshold(config.experiment_dir) is None


def test_z_calibration_cli_has_explicit_name_and_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    detections = [
        {
            "status": "completed",
            "input_mode": "known_boundary",
            "text_class": "unwatermarked",
            "z_score": score,
        }
        for score in (-1.0, 0.0, 1.0, 2.0)
    ]
    (run_dir / "detections.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in detections),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "calibrate_z_threshold.py"),
            "--run-dir",
            str(run_dir),
            "--target-fpr",
            "0.25",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((run_dir / "z_calibration.json").read_text(encoding="utf-8"))
    assert payload["score_type"] == "z_score"
    assert payload["num_negative_samples"] == 4
