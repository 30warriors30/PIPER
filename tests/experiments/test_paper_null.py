from __future__ import annotations

import pytest

from experiments.run_paper_null import (
    _build_detector,
    build_parser,
    extract_token_ids,
    summarize_p_values,
)


def test_extract_token_ids_supports_main_project_record_shapes() -> None:
    assert extract_token_ids({"token_ids": [1, 2]}, "natural") == (1, 2)
    assert extract_token_ids({"natural_token_ids": [3, 4]}, "natural") == (3, 4)
    assert extract_token_ids({"natural": {"token_ids": [5, 6]}}, "natural") == (5, 6)


def test_extract_token_ids_rejects_missing_requested_text_class() -> None:
    with pytest.raises(KeyError, match="natural"):
        extract_token_ids({"unwatermarked_token_ids": [1, 2]}, "natural")


def test_summarize_p_values_reports_exact_empirical_fpr_and_interval() -> None:
    summary = summarize_p_values([0.001, 0.02, 0.5, 0.9], [0.01, 0.05])

    assert summary["0.01"]["false_positives"] == 1
    assert summary["0.01"]["empirical_fpr"] == 0.25
    assert summary["0.05"]["false_positives"] == 2
    assert summary["0.05"]["empirical_fpr"] == 0.5
    assert 0.0 <= summary["0.01"]["ci95_low"] <= 0.25
    assert 0.25 <= summary["0.01"]["ci95_high"] <= 1.0


def test_paper_null_defaults_to_selfhash_v2_detection() -> None:
    args = build_parser().parse_args(
        [
            "--input-file",
            "negative.jsonl",
            "--output-dir",
            "out",
            "--secret-key",
            "secret",
            "--vocab-size",
            "12",
        ]
    )

    detector = _build_detector(args, "secret")

    assert args.seeding_scheme == "selfhash"
    assert args.partition_engine == "v2"
    assert detector.seeding_scheme == "selfhash"
    assert detector.partition_engine == "v2"
