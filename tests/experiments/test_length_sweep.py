import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.run_length_sweep import (
    DEFAULT_T_VALUES,
    aggregate_length_results,
    build_parser,
    parse_t_values,
    run_id,
)
from utils.io import write_json


def test_parse_t_values_defaults_include_1000():
    assert parse_t_values(None) == (100, 200, 300, 500, 1000)
    assert DEFAULT_T_VALUES == (100, 200, 300, 500, 1000)


def test_parse_t_values_sorts_unique_positive_values():
    assert parse_t_values(["500", "100", "500", "1000"]) == (100, 500, 1000)


def test_parse_t_values_rejects_non_positive_values():
    with pytest.raises(ValueError, match="positive"):
        parse_t_values(["100", "0"])


def test_build_parser_defaults_match_length_sweep_design():
    args = build_parser().parse_args(
        [
            "--model-path",
            "/data/yanlu/BREW/models/facebook/opt-1.3b",
            "--secret-key",
            "dual-layer-key-2026",
        ]
    )

    assert args.b == 8
    assert parse_t_values(args.t_values) == (100, 200, 300, 500, 1000)
    assert args.delta_payload == 2.0
    assert args.delta_presence == 0.0
    assert args.target_fpr == 0.01
    assert args.ecc_n == 23
    assert args.ecc_k == 8
    assert args.ecc_t == 3


def test_run_id_names_one_directory_per_text_length():
    assert run_id(100, 8) == "T100_b8"
    assert run_id(1000, 8) == "T1000_b8"


def test_aggregate_length_results_sorts_by_text_length_and_writes_outputs(tmp_path: Path):
    root = tmp_path / "length_sweep"
    write_json(
        root / "runs" / "T500_b8" / "metrics.json",
        {
            "schema_version": 1,
            "exact_tokens": 500,
            "capacity_bits": 8,
            "model_fpr": 0.01,
            "natural_fpr": 0.02,
            "tpr": 0.95,
            "error_erasure_exact_recovery": 0.9,
            "end_to_end_exact_recovery": 0.88,
        },
    )
    write_json(
        root / "runs" / "T100_b8" / "metrics.json",
        {
            "schema_version": 1,
            "exact_tokens": 100,
            "capacity_bits": 8,
            "model_fpr": 0.02,
            "natural_fpr": 0.03,
            "tpr": 0.75,
            "error_erasure_exact_recovery": 0.5,
            "end_to_end_exact_recovery": 0.48,
        },
    )

    rows = aggregate_length_results(root, [500, 100, 1000], capacity_bits=8)

    assert [row["exact_tokens"] for row in rows] == [100, 500]
    assert (root / "length_results.json").exists()
    with (root / "length_results.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert [row["exact_tokens"] for row in payload] == [100, 500]
    with (root / "length_results.csv").open(encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["exact_tokens"] for row in csv_rows] == ["100", "500"]


def test_cli_help_exposes_t_values_and_b():
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.run_length_sweep", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--t-values" in completed.stdout
    assert "--b" in completed.stdout


def test_cli_dry_run_prints_plan_without_writing_outputs(tmp_path: Path):
    output_dir = tmp_path / "length_sweep"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.run_length_sweep",
            "--model-path",
            "/data/yanlu/BREW/models/facebook/opt-1.3b",
            "--secret-key",
            "dual-layer-key-2026",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "T values:         [100, 200, 300, 500, 1000]" in completed.stdout
    assert "Dry run completed" in completed.stdout
    assert not output_dir.exists()
