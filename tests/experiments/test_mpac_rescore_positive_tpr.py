from __future__ import annotations

import argparse
import json

import pytest

from experiments import mpac_rescore_positive_tpr as subject
from experiments.mpac_rescore_positive_tpr import validate_record_shape


def test_validate_record_shape_accepts_no_bch_23bit_record() -> None:
    record = {
        "sample_id": "1",
        "message_bits": "1" * 23,
        "embedded_message_bits": "1" * 23,
        "token_ids": [1, 2, 3],
        "sampled_positions": "000",
    }

    validate_record_shape(record, message_length=23, mpac_ecc="none")


def test_validate_record_shape_rejects_bch23_with_non_8bit_original_message() -> None:
    record = {
        "sample_id": "2",
        "message_bits": "1" * 23,
        "embedded_message_bits": "1" * 23,
        "token_ids": [1, 2, 3],
        "sampled_positions": "000",
    }

    with pytest.raises(ValueError, match="8-bit original message"):
        validate_record_shape(record, message_length=8, mpac_ecc="bch23")


def test_run_records_worker_count_in_metadata(tmp_path, monkeypatch) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        json.dumps(
            {
                "sample_id": "1",
                "status": "completed",
                "message_bits": "10101010",
                "embedded_message_bits": "10101010",
                "token_ids": [1, 2, 3, 4],
                "sampled_positions": "1234",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text(
        json.dumps({"empirical_threshold_results": {"fpr_0p001": {"threshold": 3.0}}}),
        encoding="utf-8",
    )

    def fake_rescore(args, records):
        assert args.workers == 7
        return [
            {
                **records[0],
                "z_score": 4.0,
                "pred_message": "170",
                "message_recovered": True,
                "original_bit_acc": 1.0,
                "embedded_bit_acc": 1.0,
                "num_tokens_scored": 4,
            }
        ]

    monkeypatch.setattr(subject, "_rescore_records", fake_rescore)

    args = argparse.Namespace(
        records_path=str(records_path),
        threshold_summary=str(threshold_path),
        output_dir=str(tmp_path / "out"),
        model_path="model",
        mb_repo="/tmp/mb-lm-watermarking",
        message_length=8,
        mpac_ecc="none",
        base=4,
        gamma=0.25,
        seeding_scheme="lefthash",
        use_position_prf=False,
        use_fixed_position=False,
        ignore_repeated_ngrams=True,
        device="cpu",
        local_files_only=True,
        limit=None,
        workers=7,
        overwrite=True,
    )

    summary = subject.run(args)

    assert summary["metadata"]["workers"] == 7
