from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm import tqdm

from experiments.config import OperatingPoint, ParetoExperimentConfig
from utils.io import append_jsonl, iter_jsonl, write_json
from utils.model import load_model_and_tokenizer
from watermark.config import ModelConfig


def continuation_nll(
    model: Any,
    prompt_ids: Sequence[int],
    continuation_ids: Sequence[int],
    *,
    device: torch.device,
) -> dict[str, float | int]:
    prompt = [int(value) for value in prompt_ids]
    continuation = [int(value) for value in continuation_ids]
    if not prompt:
        raise ValueError("prompt_ids must not be empty")
    if not continuation:
        raise ValueError("continuation_ids must not be empty")
    input_ids = torch.tensor([prompt + continuation], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits if hasattr(output, "logits") else output[0]
    start = len(prompt) - 1
    selected_logits = logits[0, start : start + len(continuation), :].float()
    targets = torch.tensor(continuation, dtype=torch.long, device=selected_logits.device)
    token_nll = torch.nn.functional.cross_entropy(selected_logits, targets, reduction="none")
    total_nll = float(token_nll.sum().item())
    count = len(continuation)
    mean_nll = total_nll / count
    return {
        "nll": total_nll,
        "token_count": count,
        "mean_nll": mean_nll,
        "ppl": float(math.exp(min(mean_nll, 80.0))),
    }


def quality_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_class.setdefault(str(record["text_class"]), []).append(record)
    classes: dict[str, Any] = {}
    for text_class, rows in sorted(by_class.items()):
        total_nll = sum(float(row["nll"]) for row in rows)
        total_tokens = sum(int(row["token_count"]) for row in rows)
        mean_nll = None if total_tokens == 0 else total_nll / total_tokens
        classes[text_class] = {
            "count": len(rows),
            "token_count": total_tokens,
            "corpus_nll": total_nll,
            "mean_nll": mean_nll,
            "corpus_ppl": None if mean_nll is None else float(math.exp(min(mean_nll, 80.0))),
        }

    unwm = {
        str(row["sample_id"]): float(row["nll"]) / int(row["token_count"])
        for row in records
        if row.get("text_class") == "unwatermarked" and int(row.get("token_count", 0)) > 0
    }
    wm = {
        str(row["sample_id"]): float(row["nll"]) / int(row["token_count"])
        for row in records
        if row.get("text_class") == "watermarked" and int(row.get("token_count", 0)) > 0
    }
    paired_ids = sorted(set(unwm) & set(wm))
    deltas = [wm[sample_id] - unwm[sample_id] for sample_id in paired_ids]
    paired_delta = None if not deltas else float(np.mean(deltas))
    return {
        "classes": classes,
        "paired_count": len(paired_ids),
        "paired_delta_nll": paired_delta,
        "ppl_ratio": None if paired_delta is None else float(math.exp(paired_delta)),
        "relative_ppl_increase": None
        if paired_delta is None
        else float(math.exp(paired_delta) - 1.0),
    }


def _load_model(
    config: ParetoExperimentConfig,
    model: Any | None,
    device: torch.device | None,
) -> tuple[Any, torch.device]:
    if model is not None and device is not None:
        return model, device
    loaded_model, _, loaded_device = load_model_and_tokenizer(
        ModelConfig(
            path=config.model_path,
            device=config.device,
            dtype=config.dtype,
            local_files_only=config.local_files_only,
        )
    )
    return loaded_model if model is None else model, loaded_device if device is None else device


def _prepare(path: Path, *, resume: bool, overwrite: bool) -> set[tuple[str, str]]:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if overwrite and path.exists():
        path.unlink()
    if path.exists() and path.stat().st_size and not resume and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Use resume=True or overwrite=True.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    if not resume:
        return set()
    return {
        (str(row["sample_id"]), str(row["text_class"]))
        for row in iter_jsonl(path)
        if row.get("status", "completed") == "completed"
    }


def score_shared_quality(
    config: ParetoExperimentConfig,
    *,
    model: Any | None = None,
    device: torch.device | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    model, device = _load_model(config, model, device)
    output = config.experiment_dir / "shared" / "quality.jsonl"
    completed = _prepare(output, resume=resume, overwrite=overwrite)
    processed = skipped = 0
    rows = [row for row in iter_jsonl(config.experiment_dir / "shared" / "baseline.jsonl") if row["split"] == "test"]
    for row in tqdm(rows, desc="Shared quality", unit="sample"):
        for text_class, field in (
            ("unwatermarked", "unwatermarked_token_ids"),
            ("natural", "natural_token_ids"),
        ):
            key = (str(row["sample_id"]), text_class)
            if key in completed:
                skipped += 1
                continue
            score = continuation_nll(
                model,
                row["prompt_token_ids"],
                row[field],
                device=device,
            )
            append_jsonl(
                output,
                {
                    "schema_version": 1,
                    "sample_id": row["sample_id"],
                    "split": "test",
                    "status": "completed",
                    "text_class": text_class,
                    **score,
                },
            )
            processed += 1
    return {"processed": processed, "skipped": skipped}


def score_operating_point_quality(
    config: ParetoExperimentConfig,
    point: OperatingPoint,
    *,
    model: Any | None = None,
    device: torch.device | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    model, device = _load_model(config, model, device)
    run_dir = config.experiment_dir / "runs" / point.point_id
    output = run_dir / "quality.jsonl"
    completed = _prepare(output, resume=resume, overwrite=overwrite)
    processed = skipped = 0
    for row in tqdm(list(iter_jsonl(run_dir / "watermarked.jsonl")), desc=f"Quality {point.point_id}", unit="sample"):
        key = (str(row["sample_id"]), "watermarked")
        if key in completed:
            skipped += 1
            continue
        score = continuation_nll(
            model,
            row["prompt_token_ids"],
            row["watermarked_token_ids"],
            device=device,
        )
        append_jsonl(
            output,
            {
                "schema_version": 1,
                "sample_id": row["sample_id"],
                "split": "test",
                "status": "completed",
                "text_class": "watermarked",
                "operating_point": point.to_dict(),
                **score,
            },
        )
        processed += 1
    shared = list(iter_jsonl(config.experiment_dir / "shared" / "quality.jsonl"))
    point_rows = list(iter_jsonl(output))
    write_json(run_dir / "quality_summary.json", quality_summary(shared + point_rows))
    return {"processed": processed, "skipped": skipped}
