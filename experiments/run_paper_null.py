from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Sequence

from scipy.stats import beta

from utils.io import append_jsonl, iter_jsonl, write_json
from watermark.detector import DualLayerDetector
from watermark.ecc import BCHCodec


def extract_token_ids(record: dict[str, Any], text_class: str) -> tuple[int, ...]:
    candidates = (
        record.get("token_ids"),
        record.get(f"{text_class}_token_ids"),
        (record.get(text_class) or {}).get("token_ids")
        if isinstance(record.get(text_class), dict)
        else None,
    )
    for value in candidates:
        if isinstance(value, (list, tuple)):
            return tuple(int(token_id) for token_id in value)
    raise KeyError(f"No token IDs found for text class {text_class!r}")


def _clopper_pearson(false_positives: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    low = 0.0 if false_positives == 0 else float(beta.ppf(0.025, false_positives, total - false_positives + 1))
    high = 1.0 if false_positives == total else float(beta.ppf(0.975, false_positives + 1, total - false_positives))
    return low, high


def summarize_p_values(
    p_values: Sequence[float],
    alphas: Sequence[float],
) -> dict[str, dict[str, float | int]]:
    values = [float(value) for value in p_values]
    result: dict[str, dict[str, float | int]] = {}
    for raw_alpha in alphas:
        alpha = float(raw_alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alphas must be in (0, 1)")
        false_positives = sum(value <= alpha for value in values)
        low, high = _clopper_pearson(false_positives, len(values))
        result[f"{alpha:g}"] = {
            "alpha": alpha,
            "trials": len(values),
            "false_positives": false_positives,
            "empirical_fpr": 0.0 if not values else false_positives / len(values),
            "ci95_low": low,
            "ci95_high": high,
        }
    return result


def _key_id(secret_key: str) -> str:
    return hashlib.sha256(secret_key.encode("utf-8")).hexdigest()[:16]


def _load_keys(args: argparse.Namespace) -> list[str]:
    keys = list(args.secret_key or [])
    if args.keys_file:
        keys.extend(
            line.strip()
            for line in Path(args.keys_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    unique = list(dict.fromkeys(keys))
    if not unique:
        raise ValueError("Provide --secret-key or --keys-file")
    return unique


def _build_detector(args: argparse.Namespace, secret_key: str) -> DualLayerDetector:
    return DualLayerDetector(
        secret_key=secret_key.encode("utf-8"),
        context_width=args.context_width,
        vocab_size=args.vocab_size,
        excluded_token_ids=set(args.excluded_token_ids),
        prf_mode="paper_shared",
        ecc_codec=BCHCodec(args.ecc_n, args.ecc_k, args.ecc_t),
        presence_test="exact_binomial",
        threshold_mode="theoretical",
        target_fpr=min(args.alphas),
        fixed_z_threshold=2.326347874,
        calibrated_threshold=None,
        primary_counting_mode="unique_context",
        unique_ngram_width=args.context_width,
        min_tokens_per_code_bit=1,
        hard_fill_value=0,
        primary_policy="tie_zero",
        max_erasure_assignments=64,
        evaluate_all_policies=False,
        seeding_scheme=args.seeding_scheme,
        partition_engine=args.partition_engine,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate PIPER exact-binomial null FPR from saved token-ID JSONL records."
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--text-class",
        default="natural",
        choices=["natural", "unwatermarked", "watermarked"],
    )
    parser.add_argument("--secret-key", action="append")
    parser.add_argument("--keys-file")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--excluded-token-ids", nargs="*", type=int, default=[])
    parser.add_argument("--context-width", type=int, default=4)
    parser.add_argument(
        "--seeding-scheme",
        default="selfhash",
        choices=["history", "selfhash"],
    )
    parser.add_argument(
        "--partition-engine",
        default="v2",
        choices=["v1", "v2"],
    )
    parser.add_argument("--ecc-n", type=int, default=23)
    parser.add_argument("--ecc-k", type=int, default=8)
    parser.add_argument("--ecc-t", type=int, default=3)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.01, 0.001])
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive")
    if args.context_width <= 0:
        raise ValueError("--context-width must be positive")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("--max-records must be positive")
    summarize_p_values([], args.alphas)

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_path = output_dir / "null_detections.jsonl"
    if detections_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {detections_path}. Use --overwrite.")
    if detections_path.exists():
        detections_path.unlink()

    keys = _load_keys(args)
    detectors = [(secret_key, _build_detector(args, secret_key)) for secret_key in keys]
    per_key: dict[str, list[float]] = {_key_id(key): [] for key in keys}
    all_p_values: list[float] = []
    processed_texts = 0

    for index, record in enumerate(iter_jsonl(input_file)):
        if args.max_records is not None and processed_texts >= args.max_records:
            break
        token_ids = extract_token_ids(record, args.text_class)
        sample_id = str(record.get("sample_id", index))
        for secret_key, detector in detectors:
            result = detector.detect_token_ids(token_ids, decode_payload=False)
            primary = result.counting[result.primary_counting_mode]
            key_id = _key_id(secret_key)
            per_key[key_id].append(primary.exact_p_value)
            all_p_values.append(primary.exact_p_value)
            append_jsonl(
                detections_path,
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "text_class": args.text_class,
                    "input_mode": "blind_text",
                    "key_id": key_id,
                    "num_scored_tokens": primary.scored_tokens,
                    "upper_count": primary.upper_hits,
                    "null_probability": primary.null_probability,
                    "binomial_p_value": primary.exact_p_value,
                    "z_score": primary.z_score,
                },
            )
        processed_texts += 1

    summary = {
        "schema_version": 1,
        "input_file": str(input_file),
        "text_class": args.text_class,
        "input_mode": "blind_text",
        "presence_test": "exact_binomial",
        "counting_mode": "unique_context",
        "seeding_scheme": args.seeding_scheme,
        "partition_engine": args.partition_engine,
        "processed_texts": processed_texts,
        "num_keys": len(keys),
        "combined": summarize_p_values(all_p_values, args.alphas),
        "per_key": {
            key_id: summarize_p_values(values, args.alphas)
            for key_id, values in per_key.items()
        },
    }
    write_json(output_dir / "null_summary.json", summary)
    print(f"Processed {processed_texts} texts across {len(keys)} keys")
    print(f"Results written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
