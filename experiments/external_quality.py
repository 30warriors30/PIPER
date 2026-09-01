from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiments.pareto import pareto_front
from experiments.quality import quality_summary
from utils.io import append_jsonl, iter_jsonl, read_json, write_json
from utils.model import load_model_and_tokenizer, model_max_length
from watermark.config import ModelConfig


def evaluator_slug(model_path: str, evaluator_name: str | None = None) -> str:
    raw = evaluator_name or Path(model_path).expanduser().name or "evaluator"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip().lower()).strip("-._")
    slug = slug.replace(".", "_")
    return slug or "evaluator"


def verify_tokenizer_compatibility(generation_tokenizer: Any, evaluator_tokenizer: Any) -> None:
    generation_vocab = generation_tokenizer.get_vocab()
    evaluator_vocab = evaluator_tokenizer.get_vocab()
    if generation_vocab != evaluator_vocab:
        raise ValueError(
            "Generation and evaluator tokenizers are not token-ID compatible. "
            "This evaluator reuses saved token IDs directly, so use an OPT-family checkpoint "
            "with the same tokenizer mapping."
        )


def batched_continuation_nll(
    model: Any,
    examples: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    pad_token_id: int,
) -> list[dict[str, Any]]:
    if not examples:
        return []

    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    continuation_lengths: list[int] = []
    for example in examples:
        prompt = [int(value) for value in example["prompt_token_ids"]]
        continuation = [int(value) for value in example["continuation_token_ids"]]
        if not prompt:
            raise ValueError("prompt_token_ids must not be empty")
        if not continuation:
            raise ValueError("continuation_token_ids must not be empty")
        sequences.append(prompt + continuation)
        prompt_lengths.append(len(prompt))
        continuation_lengths.append(len(continuation))

    max_length = max(len(sequence) for sequence in sequences)
    configured_max = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(configured_max, int) and configured_max > 0 and max_length > configured_max:
        raise ValueError(
            f"Evaluator sequence length {max_length} exceeds model max_position_embeddings={configured_max}"
        )

    batch_size = len(sequences)
    input_ids = torch.full(
        (batch_size, max_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long, device=device)

    for index, (sequence, prompt_length) in enumerate(zip(sequences, prompt_lengths)):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, :length] = 1
        labels[index, prompt_length:length] = input_ids[index, prompt_length:length]

    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits if hasattr(output, "logits") else output[0]
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    flat_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(batch_size, -1)
    valid = shift_labels.ne(-100)

    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        token_count = int(valid[index].sum().item())
        expected = continuation_lengths[index]
        if token_count != expected:
            raise RuntimeError(f"Expected {expected} continuation targets, got {token_count}")
        total_nll = float(flat_loss[index][valid[index]].sum().item())
        mean_nll = total_nll / token_count
        rows.append(
            {
                "sample_id": str(example["sample_id"]),
                "nll": total_nll,
                "token_count": token_count,
                "mean_nll": mean_nll,
                "ppl": float(math.exp(min(mean_nll, 80.0))),
            }
        )
    return rows


def _chunks(rows: Sequence[Mapping[str, Any]], size: int) -> Iterable[Sequence[Mapping[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _completed_keys(path: Path) -> set[tuple[str, str]]:
    return {
        (str(row["sample_id"]), str(row["text_class"]))
        for row in iter_jsonl(path)
        if row.get("status", "completed") == "completed"
    }


def _score_examples(
    *,
    model: Any,
    device: torch.device,
    pad_token_id: int,
    examples: Sequence[dict[str, Any]],
    output: Path,
    batch_size: int,
    description: str,
    evaluator_model_path: str,
    evaluator_name: str,
    resume: bool,
) -> tuple[int, int]:
    completed = _completed_keys(output) if resume and output.exists() else set()
    pending = [
        row
        for row in examples
        if (str(row["sample_id"]), str(row["text_class"])) not in completed
    ]
    skipped = len(examples) - len(pending)
    processed = 0
    progress = tqdm(total=len(pending), desc=description, unit="sample", dynamic_ncols=True)
    try:
        for batch in _chunks(pending, batch_size):
            scores = batched_continuation_nll(
                model,
                batch,
                device=device,
                pad_token_id=pad_token_id,
            )
            for source, score in zip(batch, scores):
                append_jsonl(
                    output,
                    {
                        "schema_version": 1,
                        "sample_id": source["sample_id"],
                        "split": source.get("split", "test"),
                        "status": "completed",
                        "text_class": source["text_class"],
                        "evaluator_model": evaluator_model_path,
                        "evaluator_name": evaluator_name,
                        **{key: value for key, value in score.items() if key != "sample_id"},
                    },
                )
                processed += 1
                progress.update(1)
    finally:
        progress.close()
    return processed, skipped


def _load_sweep_rows(experiment_dir: Path) -> list[dict[str, Any]]:
    path = experiment_dir / "sweep_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run experiments/aggregate_results.py before external PPL evaluation."
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return [dict(row) for row in value]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _plot_external(
    rows: Sequence[dict[str, Any]],
    front: Sequence[dict[str, Any]],
    *,
    y_key: str,
    ylabel: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    valid = [
        row
        for row in rows
        if row.get("external_relative_ppl_increase") is not None and row.get(y_key) is not None
    ]
    figure, axis = plt.subplots()
    axis.scatter(
        [100.0 * float(row["external_relative_ppl_increase"]) for row in valid],
        [float(row[y_key]) for row in valid],
    )
    if front:
        axis.plot(
            [100.0 * float(row["external_relative_ppl_increase"]) for row in front],
            [float(row[y_key]) for row in front],
            marker="o",
        )
    axis.set_xlabel("External evaluator relative PPL increase (%)")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _aggregate_external(
    *,
    experiment_dir: Path,
    output_dir: Path,
    evaluator_model_path: str,
    evaluator_name: str,
    point_ids: Sequence[str],
) -> list[dict[str, Any]]:
    shared_rows = list(iter_jsonl(output_dir / "shared" / "quality.jsonl"))
    original_rows = {str(row["point_id"]): row for row in _load_sweep_rows(experiment_dir)}
    augmented: list[dict[str, Any]] = []
    for point_id in point_ids:
        if point_id not in original_rows:
            continue
        point_rows = list(iter_jsonl(output_dir / "runs" / point_id / "quality.jsonl"))
        summary = quality_summary(shared_rows + point_rows)
        write_json(output_dir / "runs" / point_id / "quality_summary.json", summary)
        row = dict(original_rows[point_id])
        row.update(
            {
                "external_evaluator_model": evaluator_model_path,
                "external_evaluator_name": evaluator_name,
                "external_paired_delta_nll": summary.get("paired_delta_nll"),
                "external_ppl_ratio": summary.get("ppl_ratio"),
                "external_relative_ppl_increase": summary.get("relative_ppl_increase"),
            }
        )
        augmented.append(row)

    _write_csv(output_dir / "sweep_results.csv", augmented)
    (output_dir / "sweep_results.json").write_text(
        json.dumps(augmented, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    presence_front = pareto_front(
        augmented,
        x_key="external_relative_ppl_increase",
        y_key="tpr",
    )
    payload_front = pareto_front(
        augmented,
        x_key="external_relative_ppl_increase",
        y_key="error_erasure_exact_recovery",
    )
    _write_csv(output_dir / "pareto_presence.csv", presence_front)
    _write_csv(output_dir / "pareto_payload.csv", payload_front)
    _plot_external(
        augmented,
        presence_front,
        y_key="tpr",
        ylabel="TPR at frozen threshold",
        path=output_dir / "figures" / "quality_vs_tpr.png",
    )
    _plot_external(
        augmented,
        payload_front,
        y_key="error_erasure_exact_recovery",
        ylabel="Exact message recovery",
        path=output_dir / "figures" / "quality_vs_recovery.png",
    )
    return augmented


def score_external_evaluator(
    *,
    experiment_dir: Path,
    evaluator_model_path: str,
    evaluator_name: str | None = None,
    batch_size: int = 1,
    dtype: str = "float16",
    device_name: str = "cuda",
    local_files_only: bool = True,
    resume: bool = False,
    overwrite: bool = False,
    only_points: Sequence[str] = (),
    model: Any | None = None,
    generation_tokenizer: Any | None = None,
    evaluator_tokenizer: Any | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    experiment_dir = Path(experiment_dir)
    metadata = read_json(experiment_dir / "experiment.json")
    generation_model_path = str(metadata.get("config", {}).get("model_path", ""))
    if not generation_model_path:
        raise ValueError("experiment.json is missing config.model_path")

    slug = evaluator_slug(evaluator_model_path, evaluator_name)
    display_name = evaluator_name or Path(evaluator_model_path).name
    output_dir = experiment_dir / "external_quality" / slug
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume and not overwrite:
        raise FileExistsError(f"External evaluator output exists: {output_dir}. Use --resume or --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    if model is None or evaluator_tokenizer is None or generation_tokenizer is None or device is None:
        from transformers import AutoTokenizer

        if generation_tokenizer is None:
            generation_tokenizer = AutoTokenizer.from_pretrained(
                generation_model_path,
                local_files_only=local_files_only,
            )
        if model is None or evaluator_tokenizer is None or device is None:
            loaded_model, loaded_tokenizer, loaded_device = load_model_and_tokenizer(
                ModelConfig(
                    path=evaluator_model_path,
                    device=device_name,
                    dtype=dtype,
                    local_files_only=local_files_only,
                )
            )
            model = loaded_model if model is None else model
            evaluator_tokenizer = loaded_tokenizer if evaluator_tokenizer is None else evaluator_tokenizer
            device = loaded_device if device is None else device

    assert model is not None
    assert evaluator_tokenizer is not None
    assert generation_tokenizer is not None
    assert device is not None
    verify_tokenizer_compatibility(generation_tokenizer, evaluator_tokenizer)
    if evaluator_tokenizer.pad_token_id is None:
        if evaluator_tokenizer.eos_token_id is None:
            raise ValueError("Evaluator tokenizer has neither pad_token_id nor eos_token_id")
        evaluator_tokenizer.pad_token = evaluator_tokenizer.eos_token
    pad_token_id = int(evaluator_tokenizer.pad_token_id)

    sweep_rows = _load_sweep_rows(experiment_dir)
    available_point_ids = [str(row["point_id"]) for row in sweep_rows]
    if only_points:
        unknown = sorted(set(only_points) - set(available_point_ids))
        if unknown:
            raise ValueError("Unknown point IDs: " + ", ".join(unknown))
        point_ids = [point_id for point_id in available_point_ids if point_id in set(only_points)]
    else:
        point_ids = available_point_ids

    write_json(
        output_dir / "metadata.json",
        {
            "schema_version": 1,
            "generation_model": generation_model_path,
            "evaluator_model": evaluator_model_path,
            "evaluator_name": display_name,
            "dtype": dtype,
            "device": str(device),
            "batch_size": batch_size,
            "tokenizer_id_compatible": True,
            "model_max_length": model_max_length(model, evaluator_tokenizer),
            "point_count": len(point_ids),
        },
    )

    baseline_rows = [
        row
        for row in iter_jsonl(experiment_dir / "shared" / "baseline.jsonl")
        if row.get("split") == "test"
    ]
    shared_examples: list[dict[str, Any]] = []
    for row in baseline_rows:
        shared_examples.extend(
            [
                {
                    "sample_id": row["sample_id"],
                    "split": "test",
                    "text_class": "unwatermarked",
                    "prompt_token_ids": row["prompt_token_ids"],
                    "continuation_token_ids": row["unwatermarked_token_ids"],
                },
                {
                    "sample_id": row["sample_id"],
                    "split": "test",
                    "text_class": "natural",
                    "prompt_token_ids": row["prompt_token_ids"],
                    "continuation_token_ids": row["natural_token_ids"],
                },
            ]
        )
    shared_processed, shared_skipped = _score_examples(
        model=model,
        device=device,
        pad_token_id=pad_token_id,
        examples=shared_examples,
        output=output_dir / "shared" / "quality.jsonl",
        batch_size=batch_size,
        description=f"External quality shared [{display_name}]",
        evaluator_model_path=evaluator_model_path,
        evaluator_name=display_name,
        resume=resume,
    )

    watermarked_processed = 0
    watermarked_skipped = 0
    for point_id in point_ids:
        source = experiment_dir / "runs" / point_id / "watermarked.jsonl"
        if not source.exists():
            raise FileNotFoundError(source)
        examples = [
            {
                "sample_id": row["sample_id"],
                "split": row.get("split", "test"),
                "text_class": "watermarked",
                "prompt_token_ids": row["prompt_token_ids"],
                "continuation_token_ids": row["watermarked_token_ids"],
            }
            for row in iter_jsonl(source)
            if row.get("split", "test") == "test"
        ]
        processed, skipped = _score_examples(
            model=model,
            device=device,
            pad_token_id=pad_token_id,
            examples=examples,
            output=output_dir / "runs" / point_id / "quality.jsonl",
            batch_size=batch_size,
            description=f"External quality {point_id} [{display_name}]",
            evaluator_model_path=evaluator_model_path,
            evaluator_name=display_name,
            resume=resume,
        )
        watermarked_processed += processed
        watermarked_skipped += skipped

    augmented = _aggregate_external(
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        evaluator_model_path=evaluator_model_path,
        evaluator_name=display_name,
        point_ids=point_ids,
    )
    result = {
        "evaluator": display_name,
        "output_dir": str(output_dir),
        "point_count": len(augmented),
        "shared_scored": shared_processed,
        "shared_skipped": shared_skipped,
        "watermarked_scored": watermarked_processed,
        "watermarked_skipped": watermarked_skipped,
    }
    write_json(output_dir / "summary.json", result)
    return result
