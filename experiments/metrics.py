from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from evaluation.metrics import evaluate_records
from experiments.config import OperatingPoint, ParetoExperimentConfig
from experiments.detection import load_shared_z_threshold
from experiments.manifest import selected_test_sample_ids
from experiments.quality import quality_summary
from utils.io import iter_jsonl, write_json


def _bits(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(int(char) for char in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(bit) for bit in value)
    return None


def _payload_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    decoded = 0
    exact = 0
    wrong = 0
    erasures: list[int] = []
    raw_errors = 0
    raw_decided = 0
    raw_erased = 0
    total_code_bits = 0
    statuses: Counter[str] = Counter()
    for record in records:
        payload = record.get("tie_zero")
        status = str(payload.get("status", "missing")) if isinstance(payload, dict) else "missing"
        statuses[status] += 1
        expected_message = _bits(record.get("message_bits"))
        observed_message = _bits(payload.get("message_bits")) if isinstance(payload, dict) else None
        if status == "decoded" and observed_message is not None:
            decoded += 1
            if expected_message == observed_message:
                exact += 1
            else:
                wrong += 1

        expected_code = _bits(record.get("encoded_bits"))
        n0 = record.get("code_bit_counts_0") or []
        n1 = record.get("code_bit_counts_1") or []
        erased = {int(index) for index in record.get("erasure_positions", [])}
        erasures.append(len(erased))
        if expected_code is not None and len(n0) == len(expected_code) and len(n1) == len(expected_code):
            total_code_bits += len(expected_code)
            raw_erased += len(erased)
            for index, expected in enumerate(expected_code):
                if index in erased:
                    continue
                decision = 1 if int(n1[index]) > int(n0[index]) else 0
                raw_decided += 1
                raw_errors += int(decision != expected)

    return {
        "count": total,
        "decoded_count": decoded,
        "exact_count": exact,
        "wrong_message_count": wrong,
        "abstention_count": total - decoded,
        "decode_coverage": None if total == 0 else decoded / total,
        "exact_message_recovery": None if total == 0 else exact / total,
        "wrong_message_rate": None if total == 0 else wrong / total,
        "abstention_rate": None if total == 0 else (total - decoded) / total,
        "mean_erasures": None if not erasures else float(np.mean(erasures)),
        "median_erasures": None if not erasures else float(np.median(erasures)),
        "max_erasures": None if not erasures else int(max(erasures)),
        "raw_decided_bit_ber": None if raw_decided == 0 else raw_errors / raw_decided,
        "raw_erasure_rate": None if total_code_bits == 0 else raw_erased / total_code_bits,
        "status_counts": dict(sorted(statuses.items())),
    }


def compute_operating_point_metrics(
    config: ParetoExperimentConfig,
    point: OperatingPoint,
    *,
    input_mode: str = "known_boundary",
) -> dict[str, Any]:
    selected_ids = selected_test_sample_ids(config)
    negative_records = [
        row
        for row in iter_jsonl(config.experiment_dir / "shared" / "negative_detections.jsonl")
        if row.get("split") == "test" and row.get("input_mode") == input_mode
    ]
    watermarked_records = [
        row
        for row in iter_jsonl(
            config.experiment_dir / "runs" / point.point_id / "watermarked_detections.jsonl"
        )
        if row.get("split", "test") == "test"
        and row.get("input_mode") == input_mode
        and str(row["sample_id"]) in selected_ids
    ]
    frozen_threshold = load_shared_z_threshold(config)
    if frozen_threshold is not None:
        combined = []
        for row in negative_records + watermarked_records:
            updated = dict(row)
            updated["threshold"] = frozen_threshold
            updated["detected"] = float(updated.get("z_score", 0.0)) >= frozen_threshold
            combined.append(updated)
    else:
        combined = negative_records + watermarked_records
    evaluated = evaluate_records(combined, input_mode=input_mode)
    shared_quality = [
        row
        for row in iter_jsonl(config.experiment_dir / "shared" / "quality.jsonl")
        if row.get("split", "test") == "test"
    ]
    point_quality = [
        row
        for row in iter_jsonl(
            config.experiment_dir / "runs" / point.point_id / "quality.jsonl"
        )
        if str(row["sample_id"]) in selected_ids
    ]
    result = {
        "schema_version": 1,
        "operating_point": point.to_dict(),
        "input_mode": input_mode,
        "presence": evaluated["presence"],
        "payload": evaluated["payload"],
        "payload_diagnostics": _payload_diagnostics(watermarked_records),
        "quality": quality_summary(shared_quality + point_quality),
        "runtime": evaluated.get("runtime", {}),
    }
    write_json(config.experiment_dir / "runs" / point.point_id / "metrics.json", result)
    return result
