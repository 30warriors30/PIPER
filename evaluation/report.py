from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(item, child))
    elif not isinstance(value, list):
        rows.append((prefix, value))
    return rows


def write_reports(metrics: dict[str, Any], run_dir: Path) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in _flatten(metrics):
            writer.writerow([key, "" if value is None else value])
    with (run_dir / "roc_points.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input_mode", "fpr", "tpr", "threshold"])
        for mode, payload in metrics.get("presence_by_input_mode", {}).items():
            roc = payload.get("roc", {})
            for fpr, tpr, threshold in zip(roc.get("fpr", []), roc.get("tpr", []), roc.get("thresholds", [])):
                writer.writerow([mode, fpr, tpr, threshold])
    lines = [f"Headline input mode: {metrics.get('headline_input_mode')}"]
    for mode, presence in metrics.get("presence_by_input_mode", {}).items():
        payload = metrics.get("payload_by_input_mode", {}).get(mode, {})
        lines.extend(
            [
                "",
                f"[{mode}]",
                f"TPR: {presence.get('tpr')}",
                f"Model FPR: {presence.get('model_fpr')}",
                f"Natural FPR: {presence.get('natural_fpr')}",
                f"AUC: {presence.get('auc')}",
                f"Exact message recovery: {payload.get('exact_message_recovery')}",
                f"Message BER: {payload.get('message_ber')}",
                f"Codeword BER: {payload.get('codeword_ber')}",
            ]
        )
        for policy in ("tie_zero", "strict", "hard_fill", "error_erasure"):
            lines.extend(
                [
                    f"{policy} decode success: {payload.get(policy + '_decode_success_rate')}",
                    f"{policy} exact recovery: {payload.get(policy + '_exact_message_recovery')}",
                    f"{policy} conditional message BER: {payload.get(policy + '_conditional_message_ber')}",
                    f"{policy} statuses: {payload.get(policy + '_decode_status_counts')}",
                ]
            )
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
