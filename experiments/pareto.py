from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from experiments.config import OperatingPoint, ParetoExperimentConfig
from utils.io import read_json, write_json


def _get(row: dict[str, Any], dotted: str) -> Any:
    value: Any = row
    for key in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def pareto_front(
    rows: Sequence[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> list[dict[str, Any]]:
    valid = [row for row in rows if _get(row, x_key) is not None and _get(row, y_key) is not None]
    front: list[dict[str, Any]] = []
    for candidate in valid:
        x = float(_get(candidate, x_key))
        y = float(_get(candidate, y_key))
        dominated = False
        for other in valid:
            if other is candidate:
                continue
            ox = float(_get(other, x_key))
            oy = float(_get(other, y_key))
            if ox <= x and oy >= y and (ox < x or oy > y):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda row: (float(_get(row, x_key)), -float(_get(row, y_key))))


def _flat_row(metrics: dict[str, Any]) -> dict[str, Any]:
    point = metrics["operating_point"]
    quality = metrics.get("quality", {})
    presence = metrics.get("presence", {})
    payload = metrics.get("payload", {})
    diagnostics = metrics.get("payload_diagnostics", {})
    return {
        "point_id": point["point_id"],
        "presence_mode": point["presence_mode"],
        "delta_presence": point["delta_presence"],
        "delta_payload": point["delta_payload"],
        "tpr": presence.get("tpr"),
        "model_fpr": presence.get("model_fpr"),
        "natural_fpr": presence.get("natural_fpr"),
        "auc": presence.get("auc"),
        "correct_attribution_rate": payload.get("correct_attribution_rate"),
        "conditional_decoding_accuracy": payload.get("conditional_decoding_accuracy"),
        "tie_zero_exact_recovery": payload.get("tie_zero_exact_message_recovery"),
        "error_erasure_exact_recovery": payload.get("error_erasure_exact_message_recovery"),
        "strict_exact_recovery": payload.get("strict_exact_message_recovery"),
        "hard_fill_exact_recovery": payload.get("hard_fill_exact_message_recovery"),
        "wrong_message_rate": diagnostics.get("wrong_message_rate"),
        "abstention_rate": diagnostics.get("abstention_rate"),
        "mean_erasures": diagnostics.get("mean_erasures"),
        "raw_decided_bit_ber": diagnostics.get("raw_decided_bit_ber"),
        "raw_erasure_rate": diagnostics.get("raw_erasure_rate"),
        "paired_delta_nll": quality.get("paired_delta_nll"),
        "ppl_ratio": quality.get("ppl_ratio"),
        "relative_ppl_increase": quality.get("relative_ppl_increase"),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _plot(
    rows: Sequence[dict[str, Any]],
    front: Sequence[dict[str, Any]],
    *,
    y_key: str,
    ylabel: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    x = [100.0 * float(row["relative_ppl_increase"]) for row in rows if row.get("relative_ppl_increase") is not None and row.get(y_key) is not None]
    y = [float(row[y_key]) for row in rows if row.get("relative_ppl_increase") is not None and row.get(y_key) is not None]
    figure, axis = plt.subplots()
    axis.scatter(x, y)
    if front:
        fx = [100.0 * float(row["relative_ppl_increase"]) for row in front]
        fy = [float(row[y_key]) for row in front]
        axis.plot(fx, fy, marker="o")
    axis.set_xlabel("Relative PPL increase (%)")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def aggregate_experiment(
    config: ParetoExperimentConfig,
    points: Sequence[OperatingPoint],
) -> dict[str, Any]:
    metrics_rows: list[dict[str, Any]] = []
    for point in points:
        path = config.experiment_dir / "runs" / point.point_id / "metrics.json"
        if path.exists():
            metrics_rows.append(read_json(path))
    flat = [_flat_row(row) for row in metrics_rows]
    _write_csv(config.experiment_dir / "sweep_results.csv", flat)
    (config.experiment_dir / "sweep_results.json").write_text(
        json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    nested_presence_front = pareto_front(
        metrics_rows,
        x_key="quality.relative_ppl_increase",
        y_key="presence.tpr",
    )
    nested_payload_front = pareto_front(
        metrics_rows,
        x_key="quality.relative_ppl_increase",
        y_key="payload.correct_attribution_rate",
    )
    presence_front = [_flat_row(row) for row in nested_presence_front]
    payload_front = [_flat_row(row) for row in nested_payload_front]
    _write_csv(config.experiment_dir / "pareto_presence.csv", presence_front)
    _write_csv(config.experiment_dir / "pareto_payload.csv", payload_front)

    _plot(
        flat,
        presence_front,
        y_key="tpr",
        ylabel="TPR at frozen threshold",
        path=config.experiment_dir / "figures" / "quality_vs_tpr.png",
    )
    _plot(
        flat,
        payload_front,
        y_key="correct_attribution_rate",
        ylabel="Correct attribution rate",
        path=config.experiment_dir / "figures" / "quality_vs_recovery.png",
    )
    result = {
        "point_count": len(flat),
        "presence_front_count": len(presence_front),
        "payload_front_count": len(payload_front),
    }
    write_json(config.experiment_dir / "aggregate_summary.json", result)
    return result
