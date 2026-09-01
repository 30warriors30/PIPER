from __future__ import annotations

import json

from experiments.mpac_positive_tpr import (
    build_positive_row,
    load_empirical_thresholds,
    repair_jsonl_tail,
    summarize_positive_records,
)
from experiments.score_shared_negative_fpr import build_output_prefix


def test_build_output_prefix_adds_tag_before_sample_limit() -> None:
    prefix = build_output_prefix(
        algorithm="mpac",
        text_class="unwatermarked",
        output_tag="none_m23",
        max_samples=10,
    )

    assert prefix == "mpac_unwatermarked_none_m23_10"


def test_build_positive_row_extends_no_bch_23bit_message_from_source_payload() -> None:
    source = {
        "sample_id": "42",
        "prompt_token_ids": [1, 2, 3],
        "generation_seed": 99,
        "message_bits": "10101010",
    }

    row = build_positive_row(
        source,
        message_length=23,
        mpac_ecc="none",
        message_seed=7,
    )
    repeated = build_positive_row(
        source,
        message_length=23,
        mpac_ecc="none",
        message_seed=7,
    )

    assert row == repeated
    assert row["sample_id"] == "42"
    assert row["split"] == "test"
    assert row["prompt_token_ids"] == [1, 2, 3]
    assert row["generation_seed"] == 99
    assert row["source_message_bits"] == "10101010"
    assert row["message_bits"].startswith("10101010")
    assert len(row["message_bits"]) == 23


def test_build_positive_row_bch23_keeps_original_8bit_payload() -> None:
    source = {
        "sample_id": "7",
        "prompt_token_ids": [4, 5],
        "generation_seed": 12,
        "message_bits": "00110001",
    }

    row = build_positive_row(
        source,
        message_length=8,
        mpac_ecc="bch23",
        message_seed=123,
    )

    assert row["message_bits"] == "00110001"
    assert row["source_message_bits"] == "00110001"


def test_load_empirical_thresholds_accepts_summary_or_empirical_file(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "fixed_threshold_results": {},
                "empirical_threshold_results": {
                    "fpr_0p01": {"threshold": 9.5},
                    "fpr_0p001": {"threshold": 23.0},
                },
            }
        ),
        encoding="utf-8",
    )

    thresholds = load_empirical_thresholds(summary_path)

    assert thresholds == {"fpr_0p01": 9.5, "fpr_0p001": 23.0}


def test_summarize_positive_records_counts_tpr_and_end_to_end_recovery() -> None:
    records = [
        {"z_score": 1.0, "message_recovered": True, "original_bit_acc": 1.0},
        {"z_score": 3.0, "message_recovered": True, "original_bit_acc": 1.0},
        {"z_score": 5.0, "message_recovered": False, "original_bit_acc": 0.5},
    ]

    summary = summarize_positive_records(
        records,
        thresholds={"fpr_0p01": 2.5, "fpr_0p001": 4.5},
    )

    assert summary["num_watermarked"] == 3
    assert summary["threshold_results"]["fpr_0p01"]["true_positives"] == 2
    assert summary["threshold_results"]["fpr_0p01"]["tpr"] == 2 / 3
    assert summary["threshold_results"]["fpr_0p01"]["end_to_end_exact_recovery"] == 1 / 3
    assert summary["threshold_results"]["fpr_0p001"]["true_positives"] == 1
    assert summary["threshold_results"]["fpr_0p001"]["tpr"] == 1 / 3
    assert summary["exact_message_recovery"] == 2 / 3


def test_repair_jsonl_tail_removes_truncated_final_record(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"sample_id":"0","status":"completed"}\n'
        '{"sample_id":"1","status":"completed"}\n'
        '{"sample_id":"2","status":',
        encoding="utf-8",
    )

    result = repair_jsonl_tail(path)

    assert result == {"valid_records": 2, "removed_invalid_records": 1}
    assert path.read_text(encoding="utf-8") == (
        '{"sample_id":"0","status":"completed"}\n'
        '{"sample_id":"1","status":"completed"}\n'
    )
