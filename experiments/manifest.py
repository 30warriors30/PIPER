from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from experiments.config import ParetoExperimentConfig
from utils.datasets import SampleFilterError, load_samples, prepare_sample
from utils.io import append_jsonl, iter_jsonl, read_json, write_json
from utils.seeds import derive_seed, message_bits
from watermark.config import DatasetConfig
from watermark.ecc import BCHCodec


def manifest_path(config: ParetoExperimentConfig) -> Path:
    return config.experiment_dir / "manifest.jsonl"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [record for record in iter_jsonl(path) if record.get("status", "completed") == "completed"]


def load_selected_test_rows(config: ParetoExperimentConfig) -> list[dict[str, Any]]:
    """Return the manifest-ordered positive test subset for watermark evaluation."""
    rows = [
        row for row in load_manifest(manifest_path(config)) if row["split"] == "test"
    ]
    return rows[: config.watermarked_test_samples]


def selected_test_sample_ids(config: ParetoExperimentConfig) -> set[str]:
    return {str(row["sample_id"]) for row in load_selected_test_rows(config)}


def _dataset_config(config: ParetoExperimentConfig) -> DatasetConfig:
    return DatasetConfig(
        kind="c4",
        name=config.dataset_name,
        config=config.dataset_config,
        split=config.dataset_split,
        streaming=config.dataset_streaming,
        max_samples=config.total_manifest_samples,
        sample_offset=config.sample_offset,
        path=config.dataset_path,
        completion_field="natural_text",
        id_field="id",
    )


def _metadata(config: ParetoExperimentConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_type": "quality_detection_pareto",
        "config": config.to_dict(),
        "manifest": {
            "calibration_samples": config.calibration_samples,
            "test_samples": config.test_samples,
            "exact_tokens": config.exact_tokens,
        },
    }


def build_manifest(
    config: ParetoExperimentConfig,
    tokenizer: Any,
    *,
    model_max_length: int,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    root = config.experiment_dir
    path = manifest_path(config)
    metadata_path = root / "experiment.json"
    filtered_path = root / "errors" / "filtered_manifest_samples.jsonl"
    expected_metadata = _metadata(config)

    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "errors").mkdir(exist_ok=True)

    if metadata_path.exists():
        existing = read_json(metadata_path)
        if existing != expected_metadata:
            raise ValueError("Existing experiment metadata does not match the requested configuration")
    else:
        write_json(metadata_path, expected_metadata)

    existing_rows = load_manifest(path)
    if existing_rows and not resume:
        raise FileExistsError(f"Manifest exists: {path}. Use resume=True or overwrite=True.")
    if len(existing_rows) >= config.total_manifest_samples:
        return {
            "completed": len(existing_rows),
            "filtered": sum(1 for _ in iter_jsonl(filtered_path)),
        }

    completed_ids = {str(row["sample_id"]) for row in existing_rows}
    completed = len(existing_rows)
    filtered = sum(1 for _ in iter_jsonl(filtered_path))
    codec = BCHCodec(config.ecc_n, config.ecc_k, config.ecc_t)

    for candidate in load_samples(_dataset_config(config)):
        if completed >= config.total_manifest_samples:
            break
        if candidate.sample_id in completed_ids:
            continue
        try:
            prepared = prepare_sample(
                candidate,
                tokenizer,
                max_new_tokens=config.exact_tokens,
                context_width=config.context_width,
                model_max_length=model_max_length,
            )
        except SampleFilterError as exc:
            filtered += 1
            append_jsonl(
                filtered_path,
                {
                    "schema_version": 1,
                    "sample_id": candidate.sample_id,
                    "status": "filtered",
                    "reason": exc.reason,
                    "raw_token_count": exc.raw_token_count,
                    "required_token_count": exc.required_token_count,
                },
            )
            continue

        if prepared.prompt_token_ids is None or prepared.natural_token_ids is None:
            raise RuntimeError("Prepared sample is missing token IDs")
        payload = message_bits(config.message_seed, prepared.sample_id, codec.k)
        encoded = codec.encode(payload)
        split = "calibration" if completed < config.calibration_samples else "test"
        append_jsonl(
            path,
            {
                "schema_version": 1,
                "manifest_index": completed,
                "sample_id": prepared.sample_id,
                "dataset": prepared.dataset,
                "status": "completed",
                "split": split,
                "prompt": prepared.prompt,
                "prompt_token_ids": list(prepared.prompt_token_ids),
                "natural_text": prepared.natural_completion or "",
                "natural_token_ids": list(prepared.natural_token_ids),
                "message_bits": "".join(map(str, payload)),
                "encoded_bits": "".join(map(str, encoded)),
                "generation_seed": derive_seed(
                    config.global_seed,
                    prepared.dataset,
                    prepared.sample_id,
                ),
                "metadata": prepared.metadata,
            },
        )
        completed_ids.add(prepared.sample_id)
        completed += 1

    if completed != config.total_manifest_samples:
        raise RuntimeError(
            f"Dataset exhausted after {completed} valid samples; "
            f"required {config.total_manifest_samples}"
        )
    return {"completed": completed, "filtered": filtered}
