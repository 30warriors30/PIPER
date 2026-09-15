from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from experiments.config import OperatingPoint, ParetoExperimentConfig
from experiments.manifest import load_manifest, manifest_path
from utils.batched_generation import (
    GeneratedSequence,
    adaptive_batches,
    generate_exact_batch,
)
from utils.io import append_jsonl, iter_jsonl, write_json
from utils.model import excluded_token_ids, load_model_and_tokenizer
from watermark.config import ModelConfig
from watermark.logits_processor import DualLayerLogitsProcessor


def _bits(value: str) -> tuple[int, ...]:
    return tuple(int(char) for char in value)


class ExperimentGenerator:
    def __init__(
        self,
        config: ParetoExperimentConfig,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: torch.device | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.config = config
        if batch_size is None:
            batch_size = int(getattr(config, "generation_batch_size", 16))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        if model is None or tokenizer is None or device is None:
            loaded_model, loaded_tokenizer, loaded_device = load_model_and_tokenizer(
                ModelConfig(
                    path=config.model_path,
                    device=config.device,
                    dtype=config.dtype,
                    local_files_only=config.local_files_only,
                )
            )
            self.model = loaded_model if model is None else model
            self.tokenizer = loaded_tokenizer if tokenizer is None else tokenizer
            self.device = loaded_device if device is None else device
        else:
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
        self.vocab_size = int(
            getattr(self.model.config, "vocab_size", len(self.tokenizer))
        )
        self.excluded_ids = excluded_token_ids(
            self.tokenizer,
            exclude_special_tokens=True,
            exclude_eos=False,
        )

    def _generate_batch(
        self,
        rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        processor: Any | None,
    ) -> tuple[GeneratedSequence, ...]:
        return generate_exact_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            prompt_token_ids=[row["prompt_token_ids"] for row in rows],
            seeds=[int(row["generation_seed"]) for row in rows],
            exact_tokens=self.config.exact_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            processor=processor,
        )

    def generate_unwatermarked_batch(
        self,
        rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> tuple[GeneratedSequence, ...]:
        return self._generate_batch(rows, None)

    def generate_watermarked_batch(
        self,
        rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        point: OperatingPoint,
    ) -> tuple[GeneratedSequence, ...]:
        processor = DualLayerLogitsProcessor(
            secret_key=self.config.secret_key.encode("utf-8"),
            context_width=self.config.context_width,
            encoded_bits_by_row=tuple(_bits(str(row["encoded_bits"])) for row in rows),
            vocab_size=self.vocab_size,
            excluded_token_ids=self.excluded_ids,
            presence_mode=point.presence_mode,
            delta_presence=point.delta_presence,
            delta_payload=point.delta_payload,
            prf_mode=self.config.prf_mode,
            partition_engine=self.config.partition_engine,
            capture_traces=False,
            seeding_scheme=self.config.seeding_scheme,
            candidate_top_k=self.config.candidate_top_k,
        )
        return self._generate_batch(rows, processor)

    def generate_unwatermarked(
        self, row: dict[str, Any]
    ) -> tuple[tuple[int, ...], str, float]:
        result = self.generate_unwatermarked_batch((row,))[0]
        return result.token_ids, result.text, result.seconds

    def generate_watermarked(
        self,
        row: dict[str, Any],
        point: OperatingPoint,
    ) -> tuple[tuple[int, ...], str, float]:
        result = self.generate_watermarked_batch((row,), point)[0]
        return result.token_ids, result.text, result.seconds


def _completed_ids(path: Path) -> set[str]:
    return {
        str(record["sample_id"])
        for record in iter_jsonl(path)
        if record.get("status", "completed") == "completed"
    }


def _prepare_output(path: Path, *, resume: bool, overwrite: bool) -> set[str]:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if overwrite and path.exists():
        path.unlink()
    if path.exists() and path.stat().st_size and not resume and not overwrite:
        raise FileExistsError(
            f"Output exists: {path}. Use resume=True or overwrite=True."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    return _completed_ids(path) if resume else set()


def run_shared_baseline(
    config: ParetoExperimentConfig,
    generator: ExperimentGenerator,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    output = config.experiment_dir / "shared" / "baseline.jsonl"
    completed = _prepare_output(output, resume=resume, overwrite=overwrite)
    rows = load_manifest(manifest_path(config))
    pending = [row for row in rows if str(row["sample_id"]) not in completed]
    skipped = len(rows) - len(pending)
    processed = 0
    progress = tqdm(total=len(pending), desc="Shared baseline", unit="sample")
    for batch in adaptive_batches(
        pending,
        batch_size=generator.batch_size,
        run=generator.generate_unwatermarked_batch,
    ):
        for row, result in zip(batch.items, batch.results, strict=True):
            sample_id = str(row["sample_id"])
            natural_ids = tuple(int(value) for value in row["natural_token_ids"])
            if len(natural_ids) != config.exact_tokens:
                raise RuntimeError("Manifest natural continuation is not exact length")
            append_jsonl(
                output,
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "split": row["split"],
                    "status": "completed",
                    "prompt_token_ids": row["prompt_token_ids"],
                    "message_bits": row["message_bits"],
                    "encoded_bits": row["encoded_bits"],
                    "generation_seed": row["generation_seed"],
                    "generation_batch_size_configured": generator.batch_size,
                    "generation_batch_size_actual": result.batch_size,
                    "unwatermarked_text": result.text,
                    "unwatermarked_token_ids": list(result.token_ids),
                    "unwatermarked_generation_seconds": result.seconds,
                    "natural_text": row.get("natural_text", ""),
                    "natural_token_ids": list(natural_ids),
                },
            )
            processed += 1
        progress.update(batch.batch_size)
    progress.close()
    return {"processed": processed, "skipped": skipped}


def run_operating_point_generation(
    config: ParetoExperimentConfig,
    point: OperatingPoint,
    generator: ExperimentGenerator,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    run_dir = config.experiment_dir / "runs" / point.point_id
    output = run_dir / "watermarked.jsonl"
    completed = _prepare_output(output, resume=resume, overwrite=overwrite)
    write_json(run_dir / "operating_point.json", point.to_dict())
    rows = [
        row for row in load_manifest(manifest_path(config)) if row["split"] == "test"
    ]
    pending = [row for row in rows if str(row["sample_id"]) not in completed]
    skipped = len(rows) - len(pending)
    processed = 0
    progress = tqdm(total=len(pending), desc=point.point_id, unit="sample")
    for batch in adaptive_batches(
        pending,
        batch_size=generator.batch_size,
        run=lambda chunk: generator.generate_watermarked_batch(tuple(chunk), point),
    ):
        for row, result in zip(batch.items, batch.results, strict=True):
            append_jsonl(
                output,
                {
                    "schema_version": 1,
                    "sample_id": str(row["sample_id"]),
                    "split": "test",
                    "status": "completed",
                    "prompt_token_ids": row["prompt_token_ids"],
                    "message_bits": row["message_bits"],
                    "encoded_bits": row["encoded_bits"],
                    "generation_seed": row["generation_seed"],
                    "generation_batch_size_configured": generator.batch_size,
                    "generation_batch_size_actual": result.batch_size,
                    "operating_point": point.to_dict(),
                    "watermarked_text": result.text,
                    "watermarked_token_ids": list(result.token_ids),
                    "watermarked_generation_seconds": result.seconds,
                },
            )
            processed += 1
        progress.update(batch.batch_size)
    progress.close()
    return {"processed": processed, "skipped": skipped}
