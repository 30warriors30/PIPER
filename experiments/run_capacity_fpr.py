from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

# Allow direct execution from the project root without installing the package.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from tqdm import tqdm

from utils.datasets import SampleFilterError, load_samples, prepare_sample
from utils.batched_generation import adaptive_batches, generate_exact_batch
from utils.io import append_jsonl, iter_jsonl, read_json, write_json
from utils.model import excluded_token_ids, load_model_and_tokenizer, model_max_length
from utils.seeds import derive_seed, message_bits
from watermark.config import DatasetConfig, ModelConfig
from watermark.detector import DualLayerDetector
from watermark.ecc import BCHCodec
from watermark.logits_processor import DualLayerLogitsProcessor


# ---------------------------------------------------------------------------
# Fixed paper experiment settings
# ---------------------------------------------------------------------------
EXACT_TOKENS = 500
DELTA_PRESENCE = 0.0
DELTA_PAYLOAD = 2.0
TARGET_FPR = 0.01
DEFAULT_TEST_SAMPLES = 1000
CONTEXT_WIDTH = 4
TEMPERATURE = 1.0
TOP_P = 0.95
GLOBAL_SEED = 42
MESSAGE_SEED = 42
MAX_ERASURE_ASSIGNMENTS = 64
PRESENCE_TEST = "exact_binomial"
COUNTING_MODE = "unique_context"
PRIMARY_POLICY = "tie_zero"

# b is the RAW payload length before ECC.
# All three codes have t=3 (designed minimum distance >= 7).
CAPACITY_SPECS: dict[int, tuple[int, int, int]] = {
    8: (23, 8, 3),
    16: (31, 16, 3),
    32: (50, 32, 3),
}


def theoretical_threshold(target_fpr: float) -> float:
    """One-sided standard-normal threshold for the requested nominal FPR."""
    if not 0.0 < float(target_fpr) < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    return float(norm.ppf(1.0 - float(target_fpr)))


def _bits(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(int(char) for char in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(bit) for bit in value)
    return None


def _decode_bits(payload: Any) -> tuple[int, ...] | None:
    if not isinstance(payload, dict) or str(payload.get("status")) != "decoded":
        return None
    return _bits(payload.get("message_bits"))


def _rate(values: Sequence[bool]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _format_rate(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def summarize_capacity_records(
    *,
    capacity_bits: int,
    negative_records: list[dict[str, Any]],
    watermarked_records: list[dict[str, Any]],
    exact_tokens: int = EXACT_TOKENS,
) -> dict[str, Any]:
    """Aggregate one capacity point using the fixed presence gate.

    The paper headline uses tie-to-zero ECC decoding after the presence gate.
    Legacy decoder metrics are retained as ablations.
    """

    model_negatives = [
        bool(row.get("detected", False))
        for row in negative_records
        if row.get("text_class") == "unwatermarked"
    ]
    natural_negatives = [
        bool(row.get("detected", False))
        for row in negative_records
        if row.get("text_class") == "natural"
    ]
    detected = [bool(row.get("detected", False)) for row in watermarked_records]

    strict_exact = 0
    hard_fill_exact = 0
    tie_zero_exact = 0
    error_erasure_exact = 0
    correct_attribution = 0
    detected_count = 0
    error_erasure_wrong = 0
    error_erasure_abstain = 0
    tie_zero_wrong = 0
    tie_zero_abstain = 0
    erasure_counts: list[int] = []
    raw_errors = 0
    raw_decided = 0
    raw_erased = 0
    total_code_bits = 0

    for row in watermarked_records:
        expected_message = _bits(row.get("message_bits"))
        strict_bits = _decode_bits(row.get("strict"))
        hard_fill_bits = _decode_bits(row.get("hard_fill"))
        tie_zero_bits = _decode_bits(row.get("tie_zero"))
        error_erasure_bits = _decode_bits(row.get("error_erasure"))

        strict_exact += int(
            expected_message is not None and strict_bits == expected_message
        )
        hard_fill_exact += int(
            expected_message is not None and hard_fill_bits == expected_message
        )
        tie_is_exact = (
            expected_message is not None and tie_zero_bits == expected_message
        )
        tie_zero_exact += int(tie_is_exact)
        ee_is_exact = (
            expected_message is not None and error_erasure_bits == expected_message
        )
        error_erasure_exact += int(ee_is_exact)
        detected_count += int(bool(row.get("detected", False)))
        correct_attribution += int(bool(row.get("detected", False)) and tie_is_exact)

        if tie_zero_bits is None:
            tie_zero_abstain += 1
        elif expected_message is not None and tie_zero_bits != expected_message:
            tie_zero_wrong += 1

        if error_erasure_bits is None:
            error_erasure_abstain += 1
        elif expected_message is not None and error_erasure_bits != expected_message:
            error_erasure_wrong += 1

        expected_code = _bits(row.get("encoded_bits"))
        n0 = [int(value) for value in (row.get("code_bit_counts_0") or [])]
        n1 = [int(value) for value in (row.get("code_bit_counts_1") or [])]
        erased = {int(index) for index in (row.get("erasure_positions") or [])}
        erasure_counts.append(len(erased))

        if (
            expected_code is None
            or len(n0) != len(expected_code)
            or len(n1) != len(expected_code)
        ):
            continue
        total_code_bits += len(expected_code)
        raw_erased += len(erased)
        for index, expected in enumerate(expected_code):
            if index in erased:
                continue
            decision = 1 if n1[index] > n0[index] else 0
            raw_decided += 1
            raw_errors += int(decision != expected)

    count = len(watermarked_records)
    tie_zero_rate = None if count == 0 else tie_zero_exact / count
    ee_rate = None if count == 0 else error_erasure_exact / count
    n, k, t = CAPACITY_SPECS[int(capacity_bits)]
    return {
        "capacity_bits": int(capacity_bits),
        "ecc_n": n,
        "ecc_k": k,
        "ecc_t": t,
        "exact_tokens": int(exact_tokens),
        "model_fpr": _rate(model_negatives),
        "natural_fpr": _rate(natural_negatives),
        "tpr": _rate(detected),
        "strict_exact_recovery": None if count == 0 else strict_exact / count,
        "hard_fill_exact_recovery": None if count == 0 else hard_fill_exact / count,
        "tie_zero_exact_recovery": tie_zero_rate,
        "error_erasure_exact_recovery": ee_rate,
        "correct_attribution_rate": None if count == 0 else correct_attribution / count,
        "conditional_decoding_accuracy": None
        if detected_count == 0
        else correct_attribution / detected_count,
        "end_to_end_exact_recovery": None
        if count == 0
        else correct_attribution / count,
        "wrong_message_rate": None if count == 0 else tie_zero_wrong / count,
        "abstention_rate": None if count == 0 else tie_zero_abstain / count,
        "error_erasure_wrong_message_rate": None
        if count == 0
        else error_erasure_wrong / count,
        "error_erasure_abstention_rate": None
        if count == 0
        else error_erasure_abstain / count,
        "mean_erasures": None if not erasure_counts else float(np.mean(erasure_counts)),
        "raw_decided_bit_ber": None if raw_decided == 0 else raw_errors / raw_decided,
        "raw_erasure_rate": None
        if total_code_bits == 0
        else raw_erased / total_code_bits,
        "goodput_bits_per_token": None
        if tie_zero_rate is None
        else float(capacity_bits) * tie_zero_rate / float(exact_tokens),
        "num_watermarked": count,
        "num_model_negatives": len(model_negatives),
        "num_natural_negatives": len(natural_negatives),
    }


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
        id_field="id",
    )


def _experiment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_type": "capacity_fpr_500",
        "model_path": args.model_path,
        "test_samples": args.test_samples,
        "exact_tokens": EXACT_TOKENS,
        "capacity_specs": {
            str(bits): {"n": n, "k": k, "t": t}
            for bits, (n, k, t) in CAPACITY_SPECS.items()
        },
        "presence_mode": "soft",
        "delta_presence": DELTA_PRESENCE,
        "delta_payload": DELTA_PAYLOAD,
        "target_fpr": args.target_fpr,
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


def _prepare_root(args: argparse.Namespace) -> Path:
    root = Path(args.output_dir)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.overwrite and root.exists():
        shutil.rmtree(root)
    metadata = _experiment_metadata(args)
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


def _completed_ids(path: Path) -> set[str]:
    return {
        str(row["sample_id"])
        for row in iter_jsonl(path)
        if row.get("status", "completed") == "completed"
    }


def _completed_negative_keys(path: Path) -> set[tuple[str, str]]:
    return {
        (str(row["sample_id"]), str(row["text_class"]))
        for row in iter_jsonl(path)
        if row.get("status", "completed") == "completed"
    }


def _load_completed(path: Path) -> list[dict[str, Any]]:
    return [
        row for row in iter_jsonl(path) if row.get("status", "completed") == "completed"
    ]


def build_shared_manifest(
    *,
    args: argparse.Namespace,
    root: Path,
    tokenizer: Any,
    max_model_length: int,
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
                max_new_tokens=EXACT_TOKENS,
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
            f"Dataset exhausted after {len(rows)} valid 500-token samples; "
            f"required {args.test_samples}"
        )
    return rows[: args.test_samples]


def _generation_kwargs(
    args: argparse.Namespace, tokenizer: Any, processor: Any | None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "min_new_tokens": EXACT_TOKENS,
        "max_new_tokens": EXACT_TOKENS,
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
) -> tuple[tuple[int, ...], str, float]:
    result = generate_exact_batch(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt_token_ids=(prompt_ids,),
        seeds=(int(seed),),
        exact_tokens=EXACT_TOKENS,
        temperature=args.temperature,
        top_p=args.top_p,
        processor=processor,
    )[0]
    return result.token_ids, result.text, result.seconds


def generate_shared_baseline(
    *,
    args: argparse.Namespace,
    root: Path,
    manifest: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    path = root / "shared" / "baseline.jsonl"
    completed = _completed_ids(path)
    pending = [row for row in manifest if str(row["sample_id"]) not in completed]
    progress = tqdm(total=len(pending), desc="Shared 500-token baseline", unit="sample")
    for batch in adaptive_batches(
        pending,
        batch_size=args.generation_batch_size,
        run=lambda chunk: generate_exact_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_token_ids=[row["prompt_token_ids"] for row in chunk],
            seeds=[int(row["generation_seed"]) for row in chunk],
            exact_tokens=EXACT_TOKENS,
            temperature=args.temperature,
            top_p=args.top_p,
            processor=None,
        ),
    ):
        for row, result in zip(batch.items, batch.results, strict=True):
            sample_id = str(row["sample_id"])
            natural_ids = tuple(int(value) for value in row["natural_token_ids"])
            if len(natural_ids) != EXACT_TOKENS:
                raise RuntimeError("Natural continuation is not exactly 500 tokens")
            append_jsonl(
                path,
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "status": "completed",
                    "prompt_token_ids": row["prompt_token_ids"],
                    "generation_seed": int(row["generation_seed"]),
                    "generation_batch_size_configured": args.generation_batch_size,
                    "generation_batch_size_actual": result.batch_size,
                    "unwatermarked_token_ids": list(result.token_ids),
                    "unwatermarked_text": result.text,
                    "unwatermarked_generation_seconds": result.seconds,
                    "natural_token_ids": list(natural_ids),
                    "natural_text": row["natural_text"],
                },
            )
            completed.add(sample_id)
        progress.update(batch.batch_size)
    progress.close()
    return _load_completed(path)


def _build_detector(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    vocab_size: int,
    codec: BCHCodec,
) -> DualLayerDetector:
    return DualLayerDetector(
        secret_key=args.secret_key.encode("utf-8"),
        context_width=args.context_width,
        vocab_size=int(vocab_size),
        excluded_token_ids=excluded_token_ids(
            tokenizer,
            exclude_special_tokens=True,
            exclude_eos=False,
        ),
        prf_mode="paper_shared",
        ecc_codec=codec,
        presence_test=PRESENCE_TEST,
        threshold_mode="theoretical",
        target_fpr=args.target_fpr,
        fixed_z_threshold=theoretical_threshold(args.target_fpr),
        calibrated_threshold=None,
        primary_counting_mode=COUNTING_MODE,
        unique_ngram_width=4,
        min_tokens_per_code_bit=1,
        hard_fill_value=0,
        primary_policy=PRIMARY_POLICY,
        max_erasure_assignments=args.max_erasure_assignments,
        evaluate_all_policies=True,
    )


def _presence_record(
    *,
    sample_id: str,
    text_class: str,
    result: Any,
    elapsed: float,
) -> dict[str, Any]:
    primary = result.counting[result.primary_counting_mode]
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "status": "completed",
        "text_class": text_class,
        "input_mode": result.input_mode,
        "num_scored_tokens": primary.scored_tokens,
        "upper_count": primary.upper_hits,
        "z_score": primary.z_score,
        "binomial_p_value": primary.exact_p_value,
        "null_probability": primary.null_probability,
        "presence_test": result.presence_test,
        "alpha": result.alpha,
        "threshold": result.threshold,
        "detected": result.detected,
        "decoder_invoked": result.decoder_invoked,
        "detection_time_seconds": elapsed,
    }


def detect_shared_negatives(
    *,
    args: argparse.Namespace,
    root: Path,
    baseline: list[dict[str, Any]],
    tokenizer: Any,
    vocab_size: int,
) -> list[dict[str, Any]]:
    """Detect shared negatives once.

    The coarse presence statistic does not depend on payload length b. We use the
    b=8 codec only to instantiate the existing detector; code-bit allocation does
    not affect upper/lower membership or the Z score.
    """

    path = root / "shared" / "negative_detections.jsonl"
    completed = _completed_negative_keys(path)
    n, k, t = CAPACITY_SPECS[8]
    detector = _build_detector(
        args=args,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        codec=BCHCodec(n, k, t),
    )
    progress = tqdm(
        total=len(baseline) * 2, desc="Shared negative detection", unit="result"
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
            append_jsonl(
                path,
                _presence_record(
                    sample_id=sample_id,
                    text_class=text_class,
                    result=result,
                    elapsed=elapsed,
                ),
            )
            completed.add(key)
            progress.update(1)
    progress.close()
    return _load_completed(path)


def _watermarked_detection_record(
    *,
    sample_id: str,
    capacity_bits: int,
    message: tuple[int, ...],
    encoded: tuple[int, ...],
    result: Any,
    elapsed: float,
) -> dict[str, Any]:
    primary = result.counting[result.primary_counting_mode]
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "status": "completed",
        "capacity_bits": capacity_bits,
        "text_class": "watermarked",
        "input_mode": result.input_mode,
        "num_scored_tokens": primary.scored_tokens,
        "upper_count": primary.upper_hits,
        "z_score": primary.z_score,
        "binomial_p_value": primary.exact_p_value,
        "null_probability": primary.null_probability,
        "presence_test": result.presence_test,
        "alpha": result.alpha,
        "threshold": result.threshold,
        "detected": result.detected,
        "decoder_invoked": result.decoder_invoked,
        "message_bits": "".join(map(str, message)),
        "encoded_bits": "".join(map(str, encoded)),
        "strict": asdict(result.strict_decode),
        "hard_fill": asdict(result.hard_fill_decode),
        "tie_zero": None
        if result.tie_zero_decode is None
        else asdict(result.tie_zero_decode),
        "error_erasure": asdict(result.error_erasure_decode),
        "gated_message_bits": None
        if result.gated_message_bits is None
        else "".join(map(str, result.gated_message_bits)),
        "code_bit_counts_0": list(primary.n0),
        "code_bit_counts_1": list(primary.n1),
        "erasure_positions": list(primary.erasures),
        "detection_time_seconds": elapsed,
    }


def generate_capacity_point(
    *,
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
    n, k, t = CAPACITY_SPECS[capacity_bits]
    codec = BCHCodec(n, k, t)
    run_dir = root / "runs" / f"b{capacity_bits}"
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
            delta_presence=DELTA_PRESENCE,
            delta_payload=DELTA_PAYLOAD,
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
            exact_tokens=EXACT_TOKENS,
            temperature=args.temperature,
            top_p=args.top_p,
            processor=processor,
        )

    progress = tqdm(
        total=len(pending), desc=f"Generate b={capacity_bits}", unit="sample"
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


def detect_capacity_point(
    *,
    capacity_bits: int,
    args: argparse.Namespace,
    root: Path,
    generated: list[dict[str, Any]],
    tokenizer: Any,
    vocab_size: int,
) -> list[dict[str, Any]]:
    n, k, t = CAPACITY_SPECS[capacity_bits]
    codec = BCHCodec(n, k, t)
    detector = _build_detector(
        args=args,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        codec=codec,
    )
    path = root / "runs" / f"b{capacity_bits}" / "detections.jsonl"
    completed = _completed_ids(path)

    for row in tqdm(generated, desc=f"Detect b={capacity_bits}", unit="sample"):
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
        append_jsonl(
            path,
            _watermarked_detection_record(
                sample_id=sample_id,
                capacity_bits=capacity_bits,
                message=message,
                encoded=encoded,
                result=result,
                elapsed=elapsed,
            ),
        )
        completed.add(sample_id)
    return _load_completed(path)


def write_capacity_metrics(
    *,
    capacity_bits: int,
    root: Path,
    negative_records: list[dict[str, Any]],
    watermarked_records: list[dict[str, Any]],
    target_fpr: float,
) -> dict[str, Any]:
    metrics = summarize_capacity_records(
        capacity_bits=capacity_bits,
        negative_records=negative_records,
        watermarked_records=watermarked_records,
        exact_tokens=EXACT_TOKENS,
    )
    metrics.update(
        {
            "schema_version": 1,
            "target_fpr": target_fpr,
            "theoretical_z_threshold": theoretical_threshold(target_fpr),
            "presence_test": PRESENCE_TEST,
            "counting_mode": COUNTING_MODE,
            "primary_policy": PRIMARY_POLICY,
            "delta_presence": DELTA_PRESENCE,
            "delta_payload": DELTA_PAYLOAD,
        }
    )
    write_json(root / "runs" / f"b{capacity_bits}" / "metrics.json", metrics)
    return metrics


def _plot_metric(
    rows: list[dict[str, Any]],
    *,
    fields: list[tuple[str, str]],
    ylabel: str,
    path: Path,
) -> None:
    capacities = [int(row["capacity_bits"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    for field, label in fields:
        values = [
            float(row[field]) if row.get(field) is not None else math.nan
            for row in rows
        ]
        axis.plot(capacities, values, marker="o", label=label)
    axis.set_xlabel("Raw payload capacity b (bits)")
    axis.set_ylabel(ylabel)
    axis.set_xticks(capacities)
    axis.grid(True, alpha=0.25)
    if len(fields) > 1:
        axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def aggregate_results(root: Path, capacities: Iterable[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bits in sorted(set(int(value) for value in capacities)):
        path = root / "runs" / f"b{bits}" / "metrics.json"
        if path.exists():
            rows.append(read_json(path))
    if not rows:
        raise RuntimeError("No completed capacity metrics are available to aggregate")

    fields = [
        "capacity_bits",
        "ecc_n",
        "ecc_k",
        "ecc_t",
        "exact_tokens",
        "target_fpr",
        "theoretical_z_threshold",
        "presence_test",
        "counting_mode",
        "primary_policy",
        "model_fpr",
        "natural_fpr",
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
    csv_path = root / "capacity_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    write_json(root / "capacity_results.json", rows)

    figures = root / "figures"
    _plot_metric(
        rows,
        fields=[
            ("model_fpr", "Model-generated negatives"),
            ("natural_fpr", "Natural negatives"),
        ],
        ylabel="Observed false positive rate",
        path=figures / "capacity_vs_fpr.png",
    )
    _plot_metric(
        rows,
        fields=[("tpr", "TPR")],
        ylabel="True positive rate",
        path=figures / "capacity_vs_tpr.png",
    )
    _plot_metric(
        rows,
        fields=[
            ("tie_zero_exact_recovery", "Tie-zero EMR"),
            ("correct_attribution_rate", "Correct attribution rate"),
        ],
        ylabel="Exact message recovery",
        path=figures / "capacity_vs_emr.png",
    )
    _plot_metric(
        rows,
        fields=[("goodput_bits_per_token", "Effective goodput")],
        ylabel="Recovered payload bits / token",
        path=figures / "capacity_vs_goodput.png",
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed capacity/FPR experiment: T=500, b in {8,16,32}, "
            "delta_presence=0, delta_payload=2, unique-context exact-binomial gate."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--secret-key", required=True)
    parser.add_argument("--output-dir", default="outputs/capacity_fpr_500")
    parser.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    parser.add_argument("--only-b", type=int, choices=sorted(CAPACITY_SPECS))
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
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32", "auto"],
    )
    parser.add_argument("--dataset-path")
    parser.add_argument("--dataset-name", default="allenai/c4")
    parser.add_argument("--dataset-config", default="realnewslike")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
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
    theoretical_threshold(args.target_fpr)
    # Fail immediately if any declared BCH shape is invalid in this codebase.
    for n, k, t in CAPACITY_SPECS.values():
        BCHCodec(n, k, t)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    root = _prepare_root(args)
    capacities = [args.only_b] if args.only_b is not None else sorted(CAPACITY_SPECS)

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

    print("=" * 72)
    print("Capacity / FPR experiment")
    print(f"T:               {EXACT_TOKENS}")
    print(f"b:               {capacities}")
    print(f"delta_presence:  {DELTA_PRESENCE}")
    print(f"delta_payload:   {DELTA_PAYLOAD}")
    print(f"target FPR:      {args.target_fpr}")
    print(f"presence test:   {PRESENCE_TEST}")
    print(f"counting mode:   {COUNTING_MODE}")
    print(f"decoder:         {PRIMARY_POLICY}")
    print(f"test samples:    {args.test_samples}")
    print(f"generation batch:{args.generation_batch_size} (automatic CUDA OOM backoff)")
    print(f"output:          {root}")
    print("=" * 72)

    manifest = build_shared_manifest(
        args=args,
        root=root,
        tokenizer=tokenizer,
        max_model_length=model_max_length(model, tokenizer),
    )
    baseline = generate_shared_baseline(
        args=args,
        root=root,
        manifest=manifest,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    negative_records = detect_shared_negatives(
        args=args,
        root=root,
        baseline=baseline,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
    )

    for bits in capacities:
        generated = generate_capacity_point(
            capacity_bits=bits,
            args=args,
            root=root,
            manifest=manifest,
            model=model,
            tokenizer=tokenizer,
            device=device,
            vocab_size=vocab_size,
            excluded_ids=excluded_ids,
        )
        detections = detect_capacity_point(
            capacity_bits=bits,
            args=args,
            root=root,
            generated=generated,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
        )
        metrics = write_capacity_metrics(
            capacity_bits=bits,
            root=root,
            negative_records=negative_records,
            watermarked_records=detections,
            target_fpr=args.target_fpr,
        )
        print(
            f"b={bits:>2}: FPR(model)={metrics['model_fpr']:.4f}, "
            f"FPR(natural)={metrics['natural_fpr']:.4f}, "
            f"TPR={metrics['tpr']:.4f}, "
            f"CAR={_format_rate(metrics['correct_attribution_rate'])}, "
            f"CDA={_format_rate(metrics['conditional_decoding_accuracy'])}, "
            f"goodput={metrics['goodput_bits_per_token']:.6f} bits/token"
        )

    completed_capacities = [
        bits
        for bits in sorted(CAPACITY_SPECS)
        if (root / "runs" / f"b{bits}" / "metrics.json").exists()
    ]
    aggregate_results(root, completed_capacities)
    print(f"Results written to: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
