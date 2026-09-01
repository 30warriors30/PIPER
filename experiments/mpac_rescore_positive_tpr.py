from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mpac_comparison import (
    append_jsonl,
    embedded_message_length,
    evaluate_payload_prediction,
    import_mpac,
    iter_jsonl,
    make_detector,
    score_text,
    write_json,
)
from experiments.mpac_positive_tpr import load_empirical_thresholds, summarize_positive_records


MPAC_RESCORE_ARGS: argparse.Namespace | None = None
MPAC_RESCORE_DETECTOR: Any | None = None
MPAC_RESCORE_DEVICE: torch.device | None = None


def _bit_string(value: Any, *, field: str) -> str:
    bits = str(value)
    if not bits or any(bit not in "01" for bit in bits):
        raise ValueError(f"{field} must contain only 0 and 1")
    return bits


def validate_record_shape(
    record: Mapping[str, Any],
    *,
    message_length: int,
    mpac_ecc: str,
) -> None:
    for field in ("sample_id", "token_ids", "sampled_positions", "message_bits", "embedded_message_bits"):
        if field not in record:
            raise ValueError(f"record is missing {field}")
    original_bits = _bit_string(record["message_bits"], field="message_bits")
    embedded_bits = _bit_string(record["embedded_message_bits"], field="embedded_message_bits")
    expected_embedded = 23 if mpac_ecc == "bch23" else int(message_length)
    if len(embedded_bits) != expected_embedded:
        raise ValueError(
            f"record {record['sample_id']} has {len(embedded_bits)} embedded bits, "
            f"expected {expected_embedded}"
        )
    if mpac_ecc == "bch23" and len(original_bits) != 8:
        raise ValueError("BCH23 records must carry an 8-bit original message")
    if mpac_ecc == "none" and len(original_bits) != int(message_length):
        raise ValueError(
            f"no-BCH records must carry a {message_length}-bit original message"
        )


def _load_records(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    rows = [
        row for row in iter_jsonl(path)
        if row.get("status", "completed") == "completed"
    ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: int(limit)]
    if not rows:
        raise RuntimeError(f"No completed positive records found in {path}")
    return rows


def _effective_workers(args: argparse.Namespace) -> int:
    requested = int(getattr(args, "workers", 1))
    if requested <= 0:
        raise ValueError("--workers must be positive")
    return min(requested, os.cpu_count() or 1)


def _load_detector(args: argparse.Namespace) -> tuple[Any, torch.device]:
    from transformers import AutoTokenizer

    _, detector_cls = import_mpac(Path(args.mb_repo))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    device = torch.device(args.device)
    detector = make_detector(args, tokenizer, detector_cls, device)
    return detector, device


def _rescore_one_record(
    record: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    detector: Any,
    device: torch.device,
) -> dict[str, Any]:
    validate_record_shape(
        record,
        message_length=int(args.message_length),
        mpac_ecc=args.mpac_ecc,
    )
    started = time.perf_counter()
    score = score_text(
        detector,
        token_ids=[int(value) for value in record["token_ids"]],
        message_bits=str(record["embedded_message_bits"]),
        sampled_positions=str(record["sampled_positions"]),
        device=device,
    )
    payload_eval = evaluate_payload_prediction(
        pred_message=score["pred_message"],
        original_message_bits=str(record["message_bits"]),
        embedded_message_bits=str(record["embedded_message_bits"]),
        args=args,
    )
    return {
        **record,
        "status": "completed",
        "rescore_seconds": float(time.perf_counter() - started),
        **score,
        **payload_eval,
    }


def _init_mpac_rescore_worker(
    model_path: str,
    mb_repo: str,
    local_files_only: bool,
    message_length: int,
    mpac_ecc: str,
    base: int,
    gamma: float,
    seeding_scheme: str,
    ignore_repeated_ngrams: bool,
    use_position_prf: bool,
    use_fixed_position: bool,
    device: str,
) -> None:
    global MPAC_RESCORE_ARGS, MPAC_RESCORE_DETECTOR, MPAC_RESCORE_DEVICE
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(1)
    MPAC_RESCORE_ARGS = argparse.Namespace(
        model_path=model_path,
        mb_repo=mb_repo,
        local_files_only=bool(local_files_only),
        message_length=int(message_length),
        mpac_ecc=mpac_ecc,
        base=int(base),
        gamma=float(gamma),
        seeding_scheme=seeding_scheme,
        ignore_repeated_ngrams=bool(ignore_repeated_ngrams),
        use_position_prf=bool(use_position_prf),
        use_fixed_position=bool(use_fixed_position),
        device=device,
    )
    MPAC_RESCORE_DETECTOR, MPAC_RESCORE_DEVICE = _load_detector(MPAC_RESCORE_ARGS)


def _rescore_one_record_in_worker(record: dict[str, Any]) -> dict[str, Any]:
    assert MPAC_RESCORE_ARGS is not None
    assert MPAC_RESCORE_DETECTOR is not None
    assert MPAC_RESCORE_DEVICE is not None
    return _rescore_one_record(
        record,
        args=MPAC_RESCORE_ARGS,
        detector=MPAC_RESCORE_DETECTOR,
        device=MPAC_RESCORE_DEVICE,
    )


def _rescore_records(args: argparse.Namespace, records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    workers = _effective_workers(args)
    if workers == 1:
        detector, device = _load_detector(args)
        return [
            _rescore_one_record(record, args=args, detector=detector, device=device)
            for record in tqdm(records, desc="Rescore MPAC positives", unit="sample")
        ]

    initargs = (
        args.model_path,
        args.mb_repo,
        args.local_files_only,
        args.message_length,
        args.mpac_ecc,
        args.base,
        args.gamma,
        args.seeding_scheme,
        args.ignore_repeated_ngrams,
        args.use_position_prf,
        args.use_fixed_position,
        args.device,
    )
    indexed_results: list[tuple[int, dict[str, Any]]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_mpac_rescore_worker,
        initargs=initargs,
    ) as pool:
        futures = {
            pool.submit(_rescore_one_record_in_worker, dict(record)): index
            for index, record in enumerate(records)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Rescore MPAC positives",
            unit="sample",
        ):
            indexed_results.append((futures[future], future.result()))
    indexed_results.sort(key=lambda item: item[0])
    return [row for _, row in indexed_results]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mpac_ecc == "bch23" and int(args.message_length) != 8:
        raise ValueError("--mpac-ecc bch23 requires --message-length 8")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_out = output_dir / "records.jsonl"
    if records_out.exists() and not args.overwrite:
        raise FileExistsError(f"{records_out} exists; pass --overwrite to replace it")
    if records_out.exists():
        records_out.unlink()

    thresholds = load_empirical_thresholds(Path(args.threshold_summary))
    source_records = _load_records(Path(args.records_path), limit=args.limit)
    rescored = _rescore_records(args, source_records)
    for row in rescored:
        append_jsonl(records_out, row)

    metadata = {
        "schema_version": 1,
        "experiment_type": "mpac_rescore_positive_tpr",
        "records_path": str(args.records_path),
        "threshold_summary": str(args.threshold_summary),
        "output_dir": str(output_dir),
        "model_path": args.model_path,
        "message_length": int(args.message_length),
        "mpac_ecc": args.mpac_ecc,
        "embedded_message_length": embedded_message_length(args),
        "base": int(args.base),
        "gamma": float(args.gamma),
        "seeding_scheme": args.seeding_scheme,
        "ignore_repeated_ngrams": bool(args.ignore_repeated_ngrams),
        "thresholds": thresholds,
        "num_source_records": len(source_records),
        "workers": _effective_workers(args),
    }
    if args.mpac_ecc == "bch23":
        metadata.update({"ecc_n": 23, "ecc_k": 8, "ecc_t": 3})
    summary = {
        "schema_version": 1,
        "metadata": metadata,
        "mpac": summarize_positive_records(rescored, thresholds=thresholds),
        "records_path": str(records_out),
    }
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rescore existing MPAC positives and report TPR at calibrated thresholds."
    )
    parser.add_argument("--records-path", required=True)
    parser.add_argument("--threshold-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mb-repo", default="/tmp/mb-lm-watermarking")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--message-length", type=int, default=8)
    parser.add_argument("--mpac-ecc", choices=["none", "bch23"], default="none")
    parser.add_argument("--base", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--seeding-scheme", default="lefthash")
    parser.add_argument("--use-position-prf", action="store_true")
    parser.add_argument("--use-fixed-position", action="store_true")
    parser.add_argument("--ignore-repeated-ngrams", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
