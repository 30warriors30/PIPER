from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from utils.validation import compare_experiment_configs


def plain(value: Any) -> Any:
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plain(value), ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} must be an object")
            yield value


class RunStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.experiment_path = run_dir / "experiment.json"
        self.samples_path = run_dir / "samples.jsonl"
        self.detections_path = run_dir / "detections.jsonl"
        self.logs_dir = run_dir / "logs"
        self.errors_dir = run_dir / "errors"

    @classmethod
    def initialize(
        cls,
        run_dir: Path,
        experiment: dict[str, Any],
        *,
        resume: bool = False,
        overwrite: bool = False,
    ) -> "RunStore":
        if resume and overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        if run_dir.exists() and overwrite:
            shutil.rmtree(run_dir)
        store = cls(run_dir)
        if run_dir.exists() and any(run_dir.iterdir()) and not resume:
            raise FileExistsError(f"Run directory already exists: {run_dir}. Use --resume or --overwrite.")
        run_dir.mkdir(parents=True, exist_ok=True)
        store.logs_dir.mkdir(exist_ok=True)
        store.errors_dir.mkdir(exist_ok=True)
        current = dict(experiment)
        current.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        if store.experiment_path.exists():
            existing = read_json(store.experiment_path)
            differences = compare_experiment_configs(existing, current)
            if differences:
                raise ValueError("Resume parameter mismatch:\n  " + "\n  ".join(differences))
        else:
            write_json(store.experiment_path, current)
        store.samples_path.touch(exist_ok=True)
        return store

    def completed_sample_ids(self) -> set[str]:
        return {
            str(record["sample_id"])
            for record in iter_jsonl(self.samples_path)
            if record.get("status") == "completed" and "sample_id" in record
        }

    def completed_detection_keys(self, path: Path | None = None) -> set[tuple[str, str, str]]:
        target = path or self.detections_path
        return {
            (str(record["sample_id"]), str(record["text_class"]), str(record["input_mode"]))
            for record in iter_jsonl(target)
            if record.get("status", "completed") == "completed"
            and {"sample_id", "text_class", "input_mode"}.issubset(record)
        }
