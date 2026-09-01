from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from experiments.config import OperatingPoint, ParetoExperimentConfig
from experiments.manifest import load_manifest, manifest_path
from utils.generation import paired_rng
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
    ) -> None:
        self.config = config
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
        self.vocab_size = int(getattr(self.model.config, "vocab_size", len(self.tokenizer)))
        self.excluded_ids = excluded_token_ids(
            self.tokenizer,
            exclude_special_tokens=True,
            exclude_eos=False,
        )

    def _kwargs(self, processor: Any | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "min_new_tokens": self.config.exact_tokens,
            "max_new_tokens": self.config.exact_tokens,
            "do_sample": True,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        if processor is not None:
            try:
                from transformers import LogitsProcessorList

                kwargs["logits_processor"] = LogitsProcessorList([processor])
            except ImportError:
                kwargs["logits_processor"] = [processor]
        return {key: value for key, value in kwargs.items() if value is not None}

    def _generate(
        self,
        prompt_ids: list[int] | tuple[int, ...],
        seed: int,
        processor: Any | None,
    ) -> tuple[tuple[int, ...], str, float]:
        input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        kwargs = self._kwargs(processor)
        kwargs["attention_mask"] = attention_mask
        started = time.perf_counter()
        with paired_rng(seed, self.device), torch.inference_mode():
            output = self.model.generate(input_ids=input_ids, **kwargs)
        elapsed = time.perf_counter() - started
        if not isinstance(output, torch.Tensor):
            output = output.sequences
        continuation = tuple(int(value) for value in output[0, input_ids.shape[1] :].tolist())
        if len(continuation) != self.config.exact_tokens:
            raise RuntimeError(
                f"Exact-length invariant failed: got {len(continuation)}, "
                f"expected {self.config.exact_tokens}"
            )
        return continuation, self.tokenizer.decode(continuation, skip_special_tokens=True), elapsed

    def generate_unwatermarked(self, row: dict[str, Any]) -> tuple[tuple[int, ...], str, float]:
        return self._generate(row["prompt_token_ids"], int(row["generation_seed"]), None)

    def generate_watermarked(
        self,
        row: dict[str, Any],
        point: OperatingPoint,
    ) -> tuple[tuple[int, ...], str, float]:
        processor = DualLayerLogitsProcessor(
            secret_key=self.config.secret_key.encode("utf-8"),
            context_width=self.config.context_width,
            encoded_bits=_bits(str(row["encoded_bits"])),
            vocab_size=self.vocab_size,
            excluded_token_ids=self.excluded_ids,
            presence_mode=point.presence_mode,
            delta_presence=point.delta_presence,
            delta_payload=point.delta_payload,
            prf_mode=self.config.prf_mode,
        )
        return self._generate(row["prompt_token_ids"], int(row["generation_seed"]), processor)


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
        raise FileExistsError(f"Output exists: {path}. Use resume=True or overwrite=True.")
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
    processed = skipped = 0
    for row in tqdm(load_manifest(manifest_path(config)), desc="Shared baseline", unit="sample"):
        sample_id = str(row["sample_id"])
        if sample_id in completed:
            skipped += 1
            continue
        ids, text, elapsed = generator.generate_unwatermarked(row)
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
                "unwatermarked_text": text,
                "unwatermarked_token_ids": list(ids),
                "unwatermarked_generation_seconds": elapsed,
                "natural_text": row.get("natural_text", ""),
                "natural_token_ids": list(natural_ids),
            },
        )
        processed += 1
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
    processed = skipped = 0
    rows = [row for row in load_manifest(manifest_path(config)) if row["split"] == "test"]
    for row in tqdm(rows, desc=point.point_id, unit="sample"):
        sample_id = str(row["sample_id"])
        if sample_id in completed:
            skipped += 1
            continue
        ids, text, elapsed = generator.generate_watermarked(row, point)
        append_jsonl(
            output,
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "split": "test",
                "status": "completed",
                "prompt_token_ids": row["prompt_token_ids"],
                "message_bits": row["message_bits"],
                "encoded_bits": row["encoded_bits"],
                "generation_seed": row["generation_seed"],
                "operating_point": point.to_dict(),
                "watermarked_text": text,
                "watermarked_token_ids": list(ids),
                "watermarked_generation_seconds": elapsed,
            },
        )
        processed += 1
    return {"processed": processed, "skipped": skipped}
