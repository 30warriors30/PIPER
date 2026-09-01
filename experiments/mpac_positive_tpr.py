from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mpac_comparison import (
    append_jsonl,
    embedded_message_length,
    evaluate_payload_prediction,
    generate_mpac_watermarked,
    import_mpac,
    iter_jsonl,
    load_model_and_tokenizer,
    make_detector,
    score_text,
    validate_runtime_args,
    write_json,
)


def _bits(value: str, *, expected_length: int | None = None) -> str:
    bits = str(value)
    if not bits or any(bit not in "01" for bit in bits):
        raise ValueError("message_bits must contain only 0 and 1")
    if expected_length is not None and len(bits) != int(expected_length):
        raise ValueError(f"message_bits must contain exactly {expected_length} bits")
    return bits


def _deterministic_extra_bits(
    *,
    needed: int,
    sample_id: str,
    generation_seed: int,
    message_seed: int,
    prefix_bits: str,
) -> str:
    chunks: list[str] = []
    counter = 0
    while sum(len(chunk) for chunk in chunks) < int(needed):
        material = f"{message_seed}:{sample_id}:{generation_seed}:{prefix_bits}:{counter}".encode("utf-8")
        digest = hashlib.sha256(material).digest()
        chunks.append("".join(f"{byte:08b}" for byte in digest))
        counter += 1
    return "".join(chunks)[: int(needed)]


def _message_for_width(
    source_bits: str,
    *,
    width: int,
    sample_id: str,
    generation_seed: int,
    message_seed: int,
) -> str:
    source_bits = _bits(source_bits)
    if len(source_bits) == int(width):
        return source_bits
    if len(source_bits) > int(width):
        raise ValueError(f"source message has {len(source_bits)} bits, cannot shrink to {width}")
    extra = _deterministic_extra_bits(
        needed=int(width) - len(source_bits),
        sample_id=sample_id,
        generation_seed=int(generation_seed),
        message_seed=int(message_seed),
        prefix_bits=source_bits,
    )
    return source_bits + extra


def build_positive_row(
    source: Mapping[str, Any],
    *,
    message_length: int,
    mpac_ecc: str,
    message_seed: int,
) -> dict[str, Any]:
    sample_id = str(source["sample_id"])
    generation_seed = int(source["generation_seed"])
    source_bits = _bits(str(source["message_bits"]), expected_length=8)

    if mpac_ecc == "bch23":
        if int(message_length) != 8:
            raise ValueError("BCH(23,8,3) MPAC positives require message_length=8")
        message_bits = source_bits
    elif mpac_ecc == "none":
        message_bits = _message_for_width(
            source_bits,
            width=int(message_length),
            sample_id=sample_id,
            generation_seed=generation_seed,
            message_seed=int(message_seed),
        )
    else:
        raise ValueError(f"Unsupported MPAC ECC mode: {mpac_ecc}")

    return {
        "sample_id": sample_id,
        "split": "test",
        "message_bits": message_bits,
        "source_message_bits": source_bits,
        "generation_seed": generation_seed,
        "prompt_token_ids": [int(value) for value in source["prompt_token_ids"]],
    }


def load_empirical_thresholds(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("empirical_threshold_results")
    if not isinstance(thresholds, dict):
        raise ValueError(f"{path} does not contain empirical_threshold_results")
    result: dict[str, float] = {}
    for key, value in thresholds.items():
        if not isinstance(value, dict) or "threshold" not in value:
            raise ValueError(f"Missing threshold for {key} in {path}")
        result[str(key)] = float(value["threshold"])
    if not result:
        raise ValueError(f"No empirical thresholds found in {path}")
    return result


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def summarize_positive_records(
    records: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    completed = [row for row in records if row.get("status", "completed") == "completed"]
    if not completed:
        raise ValueError("No completed positive records to summarize")

    threshold_results: dict[str, Any] = {}
    for key, threshold in thresholds.items():
        detected = [float(row["z_score"]) >= float(threshold) for row in completed]
        recovered = [bool(row.get("message_recovered", row.get("bit_match", False))) for row in completed]
        detected_and_recovered = [
            bool(is_detected and is_recovered)
            for is_detected, is_recovered in zip(detected, recovered, strict=True)
        ]
        threshold_results[str(key)] = {
            "threshold": float(threshold),
            "true_positives": int(sum(detected)),
            "tpr": float(sum(detected) / len(completed)),
            "end_to_end_exact_recovered": int(sum(detected_and_recovered)),
            "end_to_end_exact_recovery": float(sum(detected_and_recovered) / len(completed)),
        }

    recovered_values = [
        float(bool(row.get("message_recovered", row.get("bit_match", False))))
        for row in completed
    ]
    original_bit_acc = [
        float(row["original_bit_acc"])
        for row in completed
        if isinstance(row.get("original_bit_acc"), (float, int))
    ]
    embedded_bit_acc = [
        float(row["embedded_bit_acc"])
        for row in completed
        if isinstance(row.get("embedded_bit_acc"), (float, int))
    ]
    z_scores = [float(row["z_score"]) for row in completed]
    token_counts = [
        int(row["num_tokens_scored"])
        for row in completed
        if isinstance(row.get("num_tokens_scored"), (float, int))
    ]

    return {
        "num_watermarked": len(completed),
        "threshold_results": threshold_results,
        "exact_message_recovery": _mean(recovered_values),
        "mean_original_bit_accuracy": _mean(original_bit_acc),
        "mean_embedded_bit_accuracy": _mean(embedded_bit_acc),
        "mean_z_score": _mean(z_scores),
        "min_z_score": float(min(z_scores)),
        "max_z_score": float(max(z_scores)),
        "num_scored_tokens_values": sorted(set(token_counts)),
    }


def _load_positive_sources(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows = [row for row in iter_jsonl(path) if row.get("status", "completed") == "completed"]
    if limit <= 0:
        raise ValueError("--limit must be positive")
    rows = rows[: int(limit)]
    if not rows:
        raise RuntimeError(f"No completed rows found in {path}")
    return rows


def repair_jsonl_tail(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"valid_records": 0, "removed_invalid_records": 0}

    valid_lines: list[str] = []
    removed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                removed += 1
                continue
            valid_lines.append(line if line.endswith("\n") else line + "\n")

    if removed:
        with path.open("w", encoding="utf-8") as handle:
            handle.writelines(valid_lines)
    return {"valid_records": len(valid_lines), "removed_invalid_records": removed}


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["sample_id"])
        for row in iter_jsonl(path)
        if row.get("status", "completed") == "completed"
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [row for row in iter_jsonl(path) if row.get("status", "completed") == "completed"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_runtime_args(args)
    if args.mpac_ecc == "none" and int(args.message_length) not in {8, 23}:
        raise ValueError("This experiment expects no-BCH MPAC message_length to be 8 or 23")

    positive_source = Path(args.positive_source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    metadata_path = output_dir / "metadata.json"
    if records_path.exists() and args.overwrite:
        records_path.unlink()
    repair_info = repair_jsonl_tail(records_path)

    thresholds = load_empirical_thresholds(Path(args.threshold_summary))
    sources = _load_positive_sources(positive_source, limit=int(args.limit))

    metadata = {
        "schema_version": 1,
        "experiment_type": "mpac_positive_tpr",
        "positive_source": str(positive_source),
        "threshold_summary": str(args.threshold_summary),
        "output_dir": str(output_dir),
        "model_path": args.model_path,
        "num_requested": int(args.limit),
        "exact_tokens": int(args.exact_tokens),
        "message_length": int(args.message_length),
        "mpac_ecc": args.mpac_ecc,
        "embedded_message_length": embedded_message_length(args),
        "base": int(args.base),
        "gamma": float(args.gamma),
        "delta": float(args.delta),
        "seeding_scheme": args.seeding_scheme,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "message_seed": int(args.message_seed),
        "thresholds": thresholds,
        "records_repair": repair_info,
    }
    if args.mpac_ecc == "bch23":
        metadata.update({"ecc_n": 23, "ecc_k": 8, "ecc_t": 3})
    write_json(metadata_path, metadata)

    completed = _completed_ids(records_path)
    remaining = [row for row in sources if str(row["sample_id"]) not in completed]
    if remaining:
        processor_cls, detector_cls = import_mpac(Path(args.mb_repo))
        model, tokenizer, device = load_model_and_tokenizer(args)
        detector = make_detector(args, tokenizer, detector_cls, device)
        for source in tqdm(remaining, desc="Generate and score MPAC positives", unit="sample"):
            started = time.perf_counter()
            prepared = build_positive_row(
                source,
                message_length=int(args.message_length),
                mpac_ecc=args.mpac_ecc,
                message_seed=int(args.message_seed),
            )
            generated = generate_mpac_watermarked(
                prepared,
                args=args,
                model=model,
                tokenizer=tokenizer,
                device=device,
                processor_cls=processor_cls,
            )
            score = score_text(
                detector,
                token_ids=generated["token_ids"],
                message_bits=generated["embedded_message_bits"],
                sampled_positions=generated["sampled_positions"],
                device=device,
            )
            payload_eval = evaluate_payload_prediction(
                pred_message=score["pred_message"],
                original_message_bits=generated["message_bits"],
                embedded_message_bits=generated["embedded_message_bits"],
                args=args,
            )
            append_jsonl(
                records_path,
                {
                    "schema_version": 1,
                    "status": "completed",
                    "source_message_bits": prepared["source_message_bits"],
                    "total_seconds": float(time.perf_counter() - started),
                    **generated,
                    **score,
                    **payload_eval,
                },
            )

    records = _load_records(records_path)
    summary = {
        "schema_version": 1,
        "metadata": metadata,
        "mpac": summarize_positive_records(records, thresholds=thresholds),
        "records_path": str(records_path),
    }
    write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paired MPAC positives from existing BREW positives and report TPR at calibrated thresholds."
    )
    parser.add_argument("--positive-source", required=True)
    parser.add_argument("--threshold-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mb-repo", default="/tmp/mb-lm-watermarking")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--exact-tokens", type=int, default=500)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--message-length", type=int, default=8)
    parser.add_argument("--mpac-ecc", choices=["none", "bch23"], default="none")
    parser.add_argument("--message-seed", type=int, default=42)
    parser.add_argument("--base", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--seeding-scheme", default="lefthash")
    parser.add_argument("--use-position-prf", action="store_true")
    parser.add_argument("--use-fixed-position", action="store_true")
    parser.add_argument("--ignore-repeated-ngrams", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
