from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import torch

from experiments.external_quality import (
    _score_examples,
    evaluator_slug,
    verify_tokenizer_compatibility,
)
from experiments.quality import quality_summary
from utils.io import iter_jsonl, read_json, write_json
from utils.model import load_model_and_tokenizer, model_max_length
from watermark.config import ModelConfig


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_existing_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if base is not None:
        candidates.append((base / path).resolve())
    candidates.extend(
        [
            (_project_dir() / path).resolve(),
            (Path.cwd() / path).resolve(),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def _load_metadata(comparison_dir: Path) -> dict[str, Any]:
    metadata_path = comparison_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = read_json(metadata_path)
    if metadata.get("baseline") != "Segment-RSBH":
        raise ValueError(f"{comparison_dir} does not look like a Segment-RSBH comparison directory")
    return metadata


def _experiment_dir_from_metadata(comparison_dir: Path, metadata: dict[str, Any]) -> Path:
    value = metadata.get("experiment_dir")
    if not value:
        raise ValueError("Segment-RSBH metadata is missing experiment_dir")
    return _resolve_existing_path(str(value), base=comparison_dir)


def load_segment_quality_examples(
    comparison_dir: Path,
    *,
    include_shared: bool = True,
) -> list[dict[str, Any]]:
    comparison_dir = _resolve_existing_path(comparison_dir)
    metadata = _load_metadata(comparison_dir)
    examples: list[dict[str, Any]] = []

    if include_shared:
        experiment_dir = _experiment_dir_from_metadata(comparison_dir, metadata)
        for row in iter_jsonl(experiment_dir / "shared" / "baseline.jsonl"):
            if row.get("split") != "test":
                continue
            examples.extend(
                [
                    {
                        "sample_id": str(row["sample_id"]),
                        "split": "test",
                        "text_class": "unwatermarked",
                        "prompt_token_ids": row["prompt_token_ids"],
                        "continuation_token_ids": row["unwatermarked_token_ids"],
                    },
                    {
                        "sample_id": str(row["sample_id"]),
                        "split": "test",
                        "text_class": "natural",
                        "prompt_token_ids": row["prompt_token_ids"],
                        "continuation_token_ids": row["natural_token_ids"],
                    },
                ]
            )

    records_path = comparison_dir / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(records_path)
    for row in iter_jsonl(records_path):
        if row.get("split") != "test" or row.get("text_class") != "watermarked":
            continue
        examples.append(
            {
                "sample_id": str(row["sample_id"]),
                "split": "test",
                "text_class": "watermarked",
                "prompt_token_ids": row["prompt_token_ids"],
                "continuation_token_ids": row["token_ids"],
            }
        )
    return examples


def _load_runtime(
    *,
    comparison_dir: Path,
    metadata: dict[str, Any],
    evaluator_model_path: str,
    dtype: str,
    device_name: str,
    local_files_only: bool,
    model: Any | None,
    generation_tokenizer: Any | None,
    evaluator_tokenizer: Any | None,
    device: torch.device | None,
) -> tuple[Any, Any, Any, torch.device]:
    from transformers import AutoTokenizer

    generation_model = metadata.get("model_path")
    if not generation_model:
        raise ValueError("Segment-RSBH metadata is missing model_path")
    generation_model_path = _resolve_existing_path(str(generation_model), base=comparison_dir)
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
    assert device is not None
    verify_tokenizer_compatibility(generation_tokenizer, evaluator_tokenizer)
    if evaluator_tokenizer.pad_token_id is None:
        if evaluator_tokenizer.eos_token_id is None:
            raise ValueError("Evaluator tokenizer has neither pad_token_id nor eos_token_id")
        evaluator_tokenizer.pad_token = evaluator_tokenizer.eos_token
    return model, generation_tokenizer, evaluator_tokenizer, device


def score_segment_ppl(
    *,
    comparison_dir: Path,
    evaluator_model_path: str,
    evaluator_name: str | None = None,
    batch_size: int = 1,
    dtype: str = "float16",
    device_name: str = "cuda",
    local_files_only: bool = True,
    include_shared: bool = True,
    resume: bool = False,
    overwrite: bool = False,
    model: Any | None = None,
    generation_tokenizer: Any | None = None,
    evaluator_tokenizer: Any | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    comparison_dir = _resolve_existing_path(comparison_dir)
    metadata = _load_metadata(comparison_dir)
    slug = evaluator_slug(evaluator_model_path, evaluator_name)
    display_name = evaluator_name or Path(evaluator_model_path).name
    output_dir = comparison_dir / "external_quality" / slug
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume and not overwrite:
        raise FileExistsError(f"Segment-RSBH PPL output exists: {output_dir}. Use --resume or --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, generation_tokenizer, evaluator_tokenizer, device = _load_runtime(
        comparison_dir=comparison_dir,
        metadata=metadata,
        evaluator_model_path=evaluator_model_path,
        dtype=dtype,
        device_name=device_name,
        local_files_only=local_files_only,
        model=model,
        generation_tokenizer=generation_tokenizer,
        evaluator_tokenizer=evaluator_tokenizer,
        device=device,
    )
    pad_token_id = int(evaluator_tokenizer.pad_token_id)
    examples = load_segment_quality_examples(comparison_dir, include_shared=include_shared)

    write_json(
        output_dir / "metadata.json",
        {
            "schema_version": 1,
            "comparison_dir": str(comparison_dir),
            "rs_scheme": metadata.get("rs_scheme"),
            "generation_model": metadata.get("model_path"),
            "evaluator_model": evaluator_model_path,
            "evaluator_name": display_name,
            "dtype": dtype,
            "device": str(device),
            "batch_size": batch_size,
            "include_shared": include_shared,
            "tokenizer_id_compatible": True,
            "model_max_length": model_max_length(model, evaluator_tokenizer),
            "example_count": len(examples),
        },
    )
    processed, skipped = _score_examples(
        model=model,
        device=device,
        pad_token_id=pad_token_id,
        examples=examples,
        output=output_dir / "quality.jsonl",
        batch_size=batch_size,
        description=f"Segment-RSBH PPL [{display_name}]",
        evaluator_model_path=evaluator_model_path,
        evaluator_name=display_name,
        resume=resume,
    )
    rows = list(iter_jsonl(output_dir / "quality.jsonl"))
    summary = quality_summary(rows)
    result = {
        "schema_version": 1,
        "evaluator": display_name,
        "output_dir": str(output_dir),
        "scored": processed,
        "skipped": skipped,
        "total_rows": len(rows),
        "summary": summary,
    }
    write_json(output_dir / "summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute OPT-family PPL for existing Segment-RSBH comparison records."
    )
    parser.add_argument("--comparison-dir", required=True)
    parser.add_argument("--evaluator-model", "--evaluator-model-path", dest="evaluator_model", required=True)
    parser.add_argument("--evaluator-name")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only")
    parser.set_defaults(local_files_only=True)
    parser.add_argument("--include-shared", action=argparse.BooleanOptionalAction, default=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _print_plan(args: argparse.Namespace) -> None:
    comparison_dir = _resolve_existing_path(args.comparison_dir)
    slug = evaluator_slug(args.evaluator_model, args.evaluator_name)
    print("=" * 72)
    print("Segment-RSBH OPT PPL evaluator")
    print("=" * 72)
    print(f"Comparison:      {comparison_dir}")
    print(f"Evaluator model: {args.evaluator_model}")
    print(f"Evaluator name:  {args.evaluator_name or Path(args.evaluator_model).name}")
    print(f"Output:          {comparison_dir / 'external_quality' / slug}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Device / dtype:  {args.device} / {args.dtype}")
    print(f"Include shared:  {args.include_shared}")
    print("Generation:      disabled (existing Segment-RSBH texts only)")
    print("=" * 72)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _print_plan(args)
    if args.dry_run:
        print("Dry run completed. No model was loaded and no files were written.")
        return 0
    result = score_segment_ppl(
        comparison_dir=Path(args.comparison_dir),
        evaluator_model_path=args.evaluator_model,
        evaluator_name=args.evaluator_name,
        batch_size=args.batch_size,
        dtype=args.dtype,
        device_name=args.device,
        local_files_only=args.local_files_only,
        include_shared=args.include_shared,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
