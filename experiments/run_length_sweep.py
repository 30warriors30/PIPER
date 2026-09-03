from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

_process_identity = getattr(os, "getuid", os.getpid)()
_MPLCONFIGDIR = (
    Path(os.environ.get("TEMP", "/tmp")) / f"piper-matplotlib-{_process_identity}"
)
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from experiments.run_capacity_fpr import (
    DELTA_PAYLOAD,
    DELTA_PRESENCE,
    GLOBAL_SEED,
    MAX_ERASURE_ASSIGNMENTS,
    MESSAGE_SEED,
    COUNTING_MODE,
    PRESENCE_TEST,
    PRIMARY_POLICY,
    TARGET_FPR,
    TEMPERATURE,
    TOP_P,
    _bits,
    _build_detector,
    _completed_ids,
    _completed_negative_keys,
    _load_completed,
    _presence_record,
    _watermarked_detection_record,
    _format_rate,
    summarize_capacity_records,
    theoretical_threshold,
)
from utils.datasets import SampleFilterError, load_samples, prepare_sample
from utils.batched_generation import adaptive_batches, generate_exact_batch
from utils.io import append_jsonl, iter_jsonl, read_json, write_json
from utils.model import excluded_token_ids, load_model_and_tokenizer, model_max_length
from utils.seeds import derive_seed, message_bits
from watermark.config import DatasetConfig, ModelConfig
from watermark.ecc import BCHCodec
from watermark.logits_processor import DualLayerLogitsProcessor


DEFAULT_T_VALUES = (100, 200, 300, 500, 1000)
DEFAULT_TEST_SAMPLES = 200
CONTEXT_WIDTH = 4
SUPPORTED_BCH: dict[int, tuple[int, int, int]] = {8: (23, 8, 3)}


def parse_t_values(values: Sequence[str] | None) -> tuple[int, ...]:
    if values is None:
        return DEFAULT_T_VALUES
    parsed = tuple(sorted({int(value) for value in values}))
    if not parsed:
        raise ValueError("At least one T value is required")
    if any(value <= 0 for value in parsed):
        raise ValueError("T values must be positive")
    return parsed


def run_id(exact_tokens: int, capacity_bits: int) -> str:
    return f"T{int(exact_tokens)}_b{int(capacity_bits)}"


def _dataset_config(args: argparse.Namespace) -> DatasetConfig:
    return DatasetConfig(
        kind="c4",
        name=args.dataset_name,
        config=args.dataset_config,
        split=args.dataset_split,
        streaming=True,
        max_samples=args.test_samples,
        sample_offset=args.sample_offset,
        path=args.dataset_path,
        completion_field="natural_text",
        id_field="id",
    )


def _ecc_shape(args: argparse.Namespace) -> tuple[int, int, int]:
    expected = SUPPORTED_BCH.get(int(args.b))
    if expected is None:
        raise ValueError("Only --b 8 is currently supported for the length sweep")
    actual = (int(args.ecc_n), int(args.ecc_k), int(args.ecc_t))
    if actual != expected:
        raise ValueError("The current length sweep supports only BCH(23,8,3) for --b 8")
    return actual


def _experiment_metadata(
    args: argparse.Namespace, t_values: Sequence[int]
) -> dict[str, Any]:
    n, k, t = _ecc_shape(args)
    return {
        "schema_version": 1,
        "experiment_type": "length_sweep_b8",
        "model_path": args.model_path,
        "test_samples": args.test_samples,
        "t_values": list(t_values),
        "max_exact_tokens": max(t_values),
        "b": int(args.b),
        "ecc": {"n": n, "k": k, "t": t},
        "presence_mode": "soft",
        "delta_presence": float(args.delta_presence),
        "delta_payload": float(args.delta_payload),
        "target_fpr": float(args.target_fpr),
        "presence_test": PRESENCE_TEST,
        "counting_mode": COUNTING_MODE,
        "primary_policy": PRIMARY_POLICY,
        "theoretical_z_threshold": theoretical_threshold(args.target_fpr),
        "context_width": args.context_width,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "global_seed": args.global_seed,
        "message_seed": args.message_seed,
        "max_erasure_assignments": args.max_erasure_assignments,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "dataset_path": args.dataset_path,
        "sample_offset": args.sample_offset,
        "secret_key_fingerprint": __import__("hashlib")
        .sha256(args.secret_key.encode("utf-8"))
        .hexdigest()[:16],
    }


def _prepare_root(args: argparse.Namespace, t_values: Sequence[int]) -> Path:
    root = Path(args.output_dir)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.overwrite and root.exists():
        shutil.rmtree(root)
    metadata = _experiment_metadata(args, t_values)
    metadata_path = root / "experiment.json"
    if root.exists() and any(root.iterdir()):
        if not args.resume:
            raise FileExistsError(
                f"Output directory exists: {root}. Use --resume or --overwrite."
            )
        if not metadata_path.exists():
            raise ValueError(f"Cannot resume {root}: experiment.json is missing")
        if read_json(metadata_path) != metadata:
            raise ValueError(
                "Resume parameters do not match the existing experiment metadata"
            )
    root.mkdir(parents=True, exist_ok=True)
    if not metadata_path.exists():
        write_json(metadata_path, metadata)
    return root


def build_shared_manifest(
    *,
    args: argparse.Namespace,
    root: Path,
    tokenizer: Any,
    max_model_length: int,
    max_exact_tokens: int,
) -> list[dict[str, Any]]:
    path = root / "manifest.jsonl"
    existing = _load_completed(path)
    if len(existing) >= args.test_samples:
        return existing[: args.test_samples]

    completed_ids = {str(row["sample_id"]) for row in existing}
    filtered_path = root / "errors" / "filtered_manifest_samples.jsonl"
    completed = len(existing)

    for candidate in load_samples(_dataset_config(args)):
        if completed >= args.test_samples:
            break
        if candidate.sample_id in completed_ids:
            continue
        try:
            prepared = prepare_sample(
                candidate,
                tokenizer,
                max_new_tokens=max_exact_tokens,
                context_width=args.context_width,
                model_max_length=max_model_length,
            )
        except SampleFilterError as exc:
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
            raise RuntimeError("Prepared sample is missing exact token IDs")
        append_jsonl(
            path,
            {
                "schema_version": 1,
                "manifest_index": completed,
                "sample_id": prepared.sample_id,
                "dataset": prepared.dataset,
                "status": "completed",
                "prompt": prepared.prompt,
                "prompt_token_ids": list(prepared.prompt_token_ids),
                "natural_text": prepared.natural_completion or "",
                "natural_token_ids": list(prepared.natural_token_ids),
                "generation_seed": derive_seed(
                    args.global_seed, prepared.dataset, prepared.sample_id
                ),
                "metadata": prepared.metadata,
            },
        )
        completed_ids.add(prepared.sample_id)
        completed += 1

    rows = _load_completed(path)
    if len(rows) < args.test_samples:
        raise RuntimeError(
            f"Dataset exhausted after {len(rows)} valid {max_exact_tokens}-token samples; "
            f"required {args.test_samples}"
        )
    return rows[: args.test_samples]


def _generation_kwargs(
    args: argparse.Namespace,
    tokenizer: Any,
    processor: Any | None,
    *,
    exact_tokens: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "min_new_tokens": int(exact_tokens),
        "max_new_tokens": int(exact_tokens),
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    if processor is not None:
        try:
            from transformers import LogitsProcessorList

            kwargs["logits_processor"] = LogitsProcessorList([processor])
        except ImportError:
            kwargs["logits_processor"] = [processor]
    return {key: value for key, value in kwargs.items() if value is not None}


def _generate_exact(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt_ids: Sequence[int],
    seed: int,
    args: argparse.Namespace,
    processor: Any | None,
    exact_tokens: int,
) -> tuple[tuple[int, ...], str, float]:
    result = generate_exact_batch(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt_token_ids=(prompt_ids,),
        seeds=(int(seed),),
        exact_tokens=exact_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        processor=processor,
    )[0]
    return result.token_ids, result.text, result.seconds


def generate_shared_baseline(
    *,
    exact_tokens: int,
    capacity_bits: int,
    args: argparse.Namespace,
    root: Path,
    manifest: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    path = root / "runs" / run_id(exact_tokens, capacity_bits) / "baseline.jsonl"
    completed = _completed_ids(path)
    pending = [row for row in manifest if str(row["sample_id"]) not in completed]
    progress = tqdm(
        total=len(pending), desc=f"Shared baseline T={exact_tokens}", unit="sample"
    )
    for batch in adaptive_batches(
        pending,
        batch_size=args.generation_batch_size,
        run=lambda chunk: generate_exact_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_token_ids=[row["prompt_token_ids"] for row in chunk],
            seeds=[int(row["generation_seed"]) for row in chunk],
            exact_tokens=exact_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            processor=None,
        ),
    ):
        for row, result in zip(batch.items, batch.results, strict=True):
            sample_id = str(row["sample_id"])
            natural_ids = tuple(
                int(value) for value in row["natural_token_ids"][:exact_tokens]
            )
            if len(natural_ids) != int(exact_tokens):
                raise RuntimeError(
                    f"Natural continuation has fewer than {exact_tokens} tokens"
                )
            append_jsonl(
                path,
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "status": "completed",
                    "exact_tokens": int(exact_tokens),
                    "prompt_token_ids": row["prompt_token_ids"],
                    "generation_seed": int(row["generation_seed"]),
                    "generation_batch_size_configured": args.generation_batch_size,
                    "generation_batch_size_actual": result.batch_size,
                    "unwatermarked_token_ids": list(result.token_ids),
                    "unwatermarked_text": result.text,
                    "unwatermarked_generation_seconds": result.seconds,
                    "natural_token_ids": list(natural_ids),
                    "natural_text": tokenizer.decode(
                        natural_ids, skip_special_tokens=True
                    ),
                },
            )
            completed.add(sample_id)
        progress.update(batch.batch_size)
    progress.close()
    return _load_completed(path)


def detect_shared_negatives(
    *,
    exact_tokens: int,
    capacity_bits: int,
    args: argparse.Namespace,
    root: Path,
    baseline: list[dict[str, Any]],
    tokenizer: Any,
    vocab_size: int,
) -> list[dict[str, Any]]:
    path = (
        root
        / "runs"
        / run_id(exact_tokens, capacity_bits)
        / "negative_detections.jsonl"
    )
    completed = _completed_negative_keys(path)
    n, k, t = _ecc_shape(args)
    detector = _build_detector(
        args=args,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        codec=BCHCodec(n, k, t),
    )
    progress = tqdm(
        total=len(baseline) * 2,
        desc=f"Negative detection T={exact_tokens}",
        unit="result",
    )
    for row in baseline:
        sample_id = str(row["sample_id"])
        for text_class, field in (
            ("unwatermarked", "unwatermarked_token_ids"),
            ("natural", "natural_token_ids"),
        ):
            key = (sample_id, text_class)
            if key in completed:
                progress.update(1)
                continue
            token_ids = [int(value) for value in row[field]]
            started = time.perf_counter()
            result = detector.detect_token_ids(token_ids)
            elapsed = time.perf_counter() - started
            record = _presence_record(
                sample_id=sample_id,
                text_class=text_class,
                result=result,
                elapsed=elapsed,
            )
            record["exact_tokens"] = int(exact_tokens)
            append_jsonl(path, record)
            completed.add(key)
            progress.update(1)
    progress.close()
    return _load_completed(path)


def generate_length_point(
    *,
    exact_tokens: int,
    capacity_bits: int,
    args: argparse.Namespace,
    root: Path,
    manifest: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    vocab_size: int,
    excluded_ids: set[int],
) -> list[dict[str, Any]]:
    n, k, t = _ecc_shape(args)
    codec = BCHCodec(n, k, t)
    run_dir = root / "runs" / run_id(exact_tokens, capacity_bits)
    generation_path = run_dir / "watermarked.jsonl"
    completed = _completed_ids(generation_path)

    pending = []
    for row in manifest:
        sample_id = str(row["sample_id"])
        if sample_id in completed:
            continue
        message = message_bits(args.message_seed, sample_id, k)
        pending.append((row, message, codec.encode(message)))

    def generate_batch(chunk):
        processor = DualLayerLogitsProcessor(
            secret_key=args.secret_key.encode("utf-8"),
            context_width=args.context_width,
            encoded_bits_by_row=tuple(encoded for _, _, encoded in chunk),
            vocab_size=vocab_size,
            excluded_token_ids=excluded_ids,
            presence_mode="soft",
            delta_presence=args.delta_presence,
            delta_payload=args.delta_payload,
            prf_mode="paper_shared",
            # Detection and existing resumable artifacts still use the v1 partition.
            partition_engine="v1",
            capture_traces=False,
        )
        return generate_exact_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_token_ids=[row["prompt_token_ids"] for row, _, _ in chunk],
            seeds=[int(row["generation_seed"]) for row, _, _ in chunk],
            exact_tokens=exact_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            processor=processor,
        )

    progress = tqdm(
        total=len(pending), desc=f"Generate T={exact_tokens}", unit="sample"
    )
    for batch in adaptive_batches(
        pending, batch_size=args.generation_batch_size, run=generate_batch
    ):
        for item, result in zip(batch.items, batch.results, strict=True):
            row, message, encoded = item
            sample_id = str(row["sample_id"])
            append_jsonl(
                generation_path,
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "status": "completed",
                    "exact_tokens": int(exact_tokens),
                    "capacity_bits": capacity_bits,
                    "ecc": {"n": n, "k": k, "t": t},
                    "prompt_token_ids": row["prompt_token_ids"],
                    "generation_seed": int(row["generation_seed"]),
                    "generation_batch_size_configured": args.generation_batch_size,
                    "generation_batch_size_actual": result.batch_size,
                    "message_bits": "".join(map(str, message)),
                    "encoded_bits": "".join(map(str, encoded)),
                    "watermarked_token_ids": list(result.token_ids),
                    "watermarked_text": result.text,
                    "generation_seconds": result.seconds,
                },
            )
            completed.add(sample_id)
        progress.update(batch.batch_size)
    progress.close()
    return _load_completed(generation_path)


def detect_length_point(
    *,
    exact_tokens: int,
    capacity_bits: int,
    args: argparse.Namespace,
    root: Path,
    generated: list[dict[str, Any]],
    tokenizer: Any,
    vocab_size: int,
) -> list[dict[str, Any]]:
    n, k, t = _ecc_shape(args)
    codec = BCHCodec(n, k, t)
    detector = _build_detector(
        args=args,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        codec=codec,
    )
    path = root / "runs" / run_id(exact_tokens, capacity_bits) / "detections.jsonl"
    completed = _completed_ids(path)

    for row in tqdm(generated, desc=f"Detect T={exact_tokens}", unit="sample"):
        sample_id = str(row["sample_id"])
        if sample_id in completed:
            continue
        token_ids = [int(value) for value in row["watermarked_token_ids"]]
        message = _bits(row["message_bits"])
        encoded = _bits(row["encoded_bits"])
        assert message is not None and encoded is not None
        started = time.perf_counter()
        result = detector.detect_token_ids(token_ids)
        elapsed = time.perf_counter() - started
        record = _watermarked_detection_record(
            sample_id=sample_id,
            capacity_bits=capacity_bits,
            message=message,
            encoded=encoded,
            result=result,
            elapsed=elapsed,
        )
        record["exact_tokens"] = int(exact_tokens)
        append_jsonl(path, record)
        completed.add(sample_id)
    return _load_completed(path)


def _combined_fpr(metrics: dict[str, Any]) -> float | None:
    model_fpr = metrics.get("model_fpr")
    natural_fpr = metrics.get("natural_fpr")
    model_n = int(metrics.get("num_model_negatives") or 0)
    natural_n = int(metrics.get("num_natural_negatives") or 0)
    total = model_n + natural_n
    if model_fpr is None or natural_fpr is None or total == 0:
        return None
    return (float(model_fpr) * model_n + float(natural_fpr) * natural_n) / total


def write_length_metrics(
    *,
    exact_tokens: int,
    capacity_bits: int,
    root: Path,
    negative_records: list[dict[str, Any]],
    watermarked_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = summarize_capacity_records(
        capacity_bits=capacity_bits,
        negative_records=negative_records,
        watermarked_records=watermarked_records,
        exact_tokens=exact_tokens,
    )
    metrics.update(
        {
            "schema_version": 1,
            "target_fpr": args.target_fpr,
            "theoretical_z_threshold": theoretical_threshold(args.target_fpr),
            "presence_test": PRESENCE_TEST,
            "counting_mode": COUNTING_MODE,
            "primary_policy": PRIMARY_POLICY,
            "delta_presence": args.delta_presence,
            "delta_payload": args.delta_payload,
            "combined_fpr": _combined_fpr(metrics),
        }
    )
    write_json(
        root / "runs" / run_id(exact_tokens, capacity_bits) / "metrics.json", metrics
    )
    return metrics


def _plot_metric(
    rows: list[dict[str, Any]],
    *,
    fields: list[tuple[str, str]],
    ylabel: str,
    path: Path,
) -> None:
    x_values = [int(row["exact_tokens"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    for field, label in fields:
        values = [
            float(row[field]) if row.get(field) is not None else math.nan
            for row in rows
        ]
        axis.plot(x_values, values, marker="o", label=label)
    axis.set_xlabel("T")
    axis.set_ylabel(ylabel)
    axis.set_xticks(x_values)
    axis.grid(True, alpha=0.25)
    if len(fields) > 1:
        axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def aggregate_length_results(
    root: Path,
    t_values: Iterable[int],
    *,
    capacity_bits: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exact_tokens in sorted(set(int(value) for value in t_values)):
        path = root / "runs" / run_id(exact_tokens, capacity_bits) / "metrics.json"
        if path.exists():
            rows.append(read_json(path))
    if not rows:
        raise RuntimeError("No completed length metrics are available to aggregate")
    rows.sort(key=lambda row: int(row["exact_tokens"]))

    fields = [
        "exact_tokens",
        "capacity_bits",
        "ecc_n",
        "ecc_k",
        "ecc_t",
        "target_fpr",
        "theoretical_z_threshold",
        "presence_test",
        "counting_mode",
        "primary_policy",
        "model_fpr",
        "natural_fpr",
        "combined_fpr",
        "tpr",
        "strict_exact_recovery",
        "hard_fill_exact_recovery",
        "tie_zero_exact_recovery",
        "error_erasure_exact_recovery",
        "correct_attribution_rate",
        "conditional_decoding_accuracy",
        "end_to_end_exact_recovery",
        "wrong_message_rate",
        "abstention_rate",
        "mean_erasures",
        "raw_decided_bit_ber",
        "raw_erasure_rate",
        "goodput_bits_per_token",
        "num_watermarked",
        "num_model_negatives",
        "num_natural_negatives",
        "delta_presence",
        "delta_payload",
    ]
    csv_path = root / "length_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    write_json(root / "length_results.json", rows)

    figures = root / "figures"
    _plot_metric(
        rows,
        fields=[
            ("model_fpr", "Model-generated"),
            ("natural_fpr", "Natural"),
            ("combined_fpr", "Combined"),
        ],
        ylabel="Observed false positive rate",
        path=figures / "length_vs_fpr.png",
    )
    _plot_metric(
        rows,
        fields=[("tpr", "TPR")],
        ylabel="TPR@1%FPR",
        path=figures / "length_vs_tpr.png",
    )
    _plot_metric(
        rows,
        fields=[
            ("tie_zero_exact_recovery", "Tie-zero EMR"),
            ("correct_attribution_rate", "Correct attribution rate"),
        ],
        ylabel="Exact message recovery",
        path=figures / "length_vs_emr.png",
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Length sweep experiment: fixed b=8, BCH(23,8,3), "
            "delta_presence=0, delta_payload=2, and T in {100,200,300,500,1000}."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--secret-key", required=True)
    parser.add_argument(
        "--output-dir", default="outputs/experiments/length_sweep_b8_m2"
    )
    parser.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--t-values", nargs="+")
    parser.add_argument("--b", type=int, default=8, choices=sorted(SUPPORTED_BCH))
    parser.add_argument("--ecc-n", type=int, default=23)
    parser.add_argument("--ecc-k", type=int, default=8)
    parser.add_argument("--ecc-t", type=int, default=3)
    parser.add_argument("--delta-presence", type=float, default=DELTA_PRESENCE)
    parser.add_argument("--delta-payload", type=float, default=DELTA_PAYLOAD)
    parser.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    parser.add_argument("--context-width", type=int, default=CONTEXT_WIDTH)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=16,
        help="Initial generation batch size; CUDA OOM automatically halves it.",
    )
    parser.add_argument("--global-seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--message-seed", type=int, default=MESSAGE_SEED)
    parser.add_argument(
        "--max-erasure-assignments", type=int, default=MAX_ERASURE_ASSIGNMENTS
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"]
    )
    parser.add_argument("--dataset-path")
    parser.add_argument("--dataset-name", default="allenai/c4")
    parser.add_argument("--dataset-config", default="realnewslike")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, t_values: Sequence[int]) -> None:
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.test_samples <= 0:
        raise ValueError("--test-samples must be positive")
    if args.context_width <= 0:
        raise ValueError("--context-width must be positive")
    if args.generation_batch_size <= 0:
        raise ValueError("--generation-batch-size must be positive")
    if args.max_erasure_assignments <= 0:
        raise ValueError("--max-erasure-assignments must be positive")
    if float(args.delta_presence) != 0.0:
        raise ValueError("This length sweep fixes --delta-presence 0.0")
    if float(args.delta_payload) != 2.0:
        raise ValueError("This length sweep fixes --delta-payload 2.0")
    theoretical_threshold(args.target_fpr)
    _ecc_shape(args)
    BCHCodec(args.ecc_n, args.ecc_k, args.ecc_t)
    if max(t_values) <= 0:
        raise ValueError("T values must be positive")


def _print_plan(args: argparse.Namespace, t_values: Sequence[int], root: Path) -> None:
    print("=" * 72)
    print("Dual-layer length sweep")
    print(f"T values:         {list(t_values)}")
    print(f"b:                {args.b}")
    print(f"ECC:              BCH({args.ecc_n},{args.ecc_k},{args.ecc_t})")
    print(f"delta_presence:   {args.delta_presence}")
    print(f"delta_payload:    {args.delta_payload}")
    print(f"target FPR:       {args.target_fpr}")
    print(f"presence test:    {PRESENCE_TEST}")
    print(f"counting mode:    {COUNTING_MODE}")
    print(f"decoder:          {PRIMARY_POLICY}")
    print(f"test samples:     {args.test_samples}")
    print(
        f"generation batch: {args.generation_batch_size} (automatic CUDA OOM backoff)"
    )
    print(f"output:           {root}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t_values = parse_t_values(args.t_values)
    _validate_args(args, t_values)
    root = Path(args.output_dir)
    _print_plan(args, t_values, root)
    if args.dry_run:
        print("Dry run completed. No model was loaded and no files were written.")
        return 0

    root = _prepare_root(args, t_values)
    model, tokenizer, device = load_model_and_tokenizer(
        ModelConfig(
            path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            local_files_only=not args.allow_download,
        )
    )
    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))
    excluded_ids = excluded_token_ids(
        tokenizer,
        exclude_special_tokens=True,
        exclude_eos=False,
    )
    manifest = build_shared_manifest(
        args=args,
        root=root,
        tokenizer=tokenizer,
        max_model_length=model_max_length(model, tokenizer),
        max_exact_tokens=max(t_values),
    )

    for exact_tokens in t_values:
        baseline = generate_shared_baseline(
            exact_tokens=exact_tokens,
            capacity_bits=args.b,
            args=args,
            root=root,
            manifest=manifest,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        negative_records = detect_shared_negatives(
            exact_tokens=exact_tokens,
            capacity_bits=args.b,
            args=args,
            root=root,
            baseline=baseline,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
        )
        generated = generate_length_point(
            exact_tokens=exact_tokens,
            capacity_bits=args.b,
            args=args,
            root=root,
            manifest=manifest,
            model=model,
            tokenizer=tokenizer,
            device=device,
            vocab_size=vocab_size,
            excluded_ids=excluded_ids,
        )
        detections = detect_length_point(
            exact_tokens=exact_tokens,
            capacity_bits=args.b,
            args=args,
            root=root,
            generated=generated,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
        )
        metrics = write_length_metrics(
            exact_tokens=exact_tokens,
            capacity_bits=args.b,
            root=root,
            negative_records=negative_records,
            watermarked_records=detections,
            args=args,
        )
        print(
            f"T={exact_tokens:>4}: FPR(model)={metrics['model_fpr']:.4f}, "
            f"FPR(natural)={metrics['natural_fpr']:.4f}, "
            f"TPR={metrics['tpr']:.4f}, "
            f"CAR={_format_rate(metrics['correct_attribution_rate'])}, "
            f"CDA={_format_rate(metrics['conditional_decoding_accuracy'])}"
        )

    aggregate_length_results(root, t_values, capacity_bits=args.b)
    print(f"Results written to: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
