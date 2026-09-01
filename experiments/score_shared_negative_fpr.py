from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

from scipy.stats import norm
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


BREW_DETECTOR: Any | None = None
MPAC_DETECTOR: Any | None = None
MPAC_DEVICE: Any | None = None
MPAC_MESSAGE_BITS = ""


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _target_key(target_fpr: float) -> str:
    text = f"{float(target_fpr):g}".replace(".", "p").replace("-", "neg")
    return f"fpr_{text}"


def empirical_threshold(scores: Sequence[float], target_fpr: float) -> dict[str, float | int]:
    if not scores:
        raise ValueError("scores must not be empty")
    if not 0.0 <= float(target_fpr) < 1.0:
        raise ValueError("target_fpr must be in [0, 1)")
    values = [float(score) for score in scores]
    count = len(values)
    for candidate in sorted(set(values)):
        false_positives = sum(score >= candidate for score in values)
        false_positive_rate = false_positives / count
        if false_positive_rate <= float(target_fpr):
            return {
                "threshold": float(candidate),
                "false_positives": int(false_positives),
                "empirical_fpr": float(false_positive_rate),
            }
    return {
        "threshold": float(math.nextafter(max(values), math.inf)),
        "false_positives": 0,
        "empirical_fpr": 0.0,
    }


def summarize_fixed_thresholds(
    scores: Sequence[float],
    *,
    target_fprs: Sequence[float],
) -> dict[str, dict[str, float | int | bool | list[float]]]:
    if not scores:
        raise ValueError("scores must not be empty")
    values = [float(score) for score in scores]
    count = len(values)
    mean = sum(values) / count
    std = math.sqrt(sum((score - mean) ** 2 for score in values) / (count - 1)) if count > 1 else 0.0
    summaries: dict[str, dict[str, float | int | bool | list[float]]] = {}
    for target in target_fprs:
        target = float(target)
        threshold = float(norm.ppf(1.0 - target))
        false_positives = sum(score >= threshold for score in values)
        empirical_fpr = false_positives / count
        se_target = math.sqrt(target * (1.0 - target) / count)
        interval = [target - 1.96 * se_target, target + 1.96 * se_target]
        summaries[_target_key(target)] = {
            "target_fpr": target,
            "fixed_z_threshold": threshold,
            "num_samples": count,
            "false_positives": int(false_positives),
            "empirical_fpr": float(empirical_fpr),
            "expected_false_positives": float(count * target),
            "target_fpr_approx_95pct_interval_for_n": interval,
            "matches_target_interval": bool(interval[0] <= empirical_fpr <= interval[1]),
            "mean_z_score": float(mean),
            "std_z_score": float(std),
            "min_z_score": float(min(values)),
            "max_z_score": float(max(values)),
        }
    return summaries


def summarize_empirical_thresholds(
    scores: Sequence[float],
    *,
    target_fprs: Sequence[float],
) -> dict[str, dict[str, float | int]]:
    return {
        _target_key(float(target)): empirical_threshold(scores, float(target))
        for target in target_fprs
    }


def build_output_prefix(
    *,
    algorithm: str,
    text_class: str,
    output_tag: str | None,
    max_samples: int | None,
) -> str:
    prefix = f"{algorithm}_{text_class}"
    if output_tag:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_tag):
            raise ValueError("--output-tag may contain only letters, digits, '_', '.', and '-'")
        prefix += f"_{output_tag}"
    if max_samples is not None:
        prefix += f"_{int(max_samples)}"
    return prefix


def _baseline_path(root: Path, exact_tokens: int, b: int) -> Path:
    return root / "runs" / f"T{int(exact_tokens)}_b{int(b)}" / "baseline.jsonl"


def _load_rows(
    *,
    root: Path,
    text_class: str,
    exact_tokens: int,
    b: int,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    if text_class == "natural":
        baseline = _baseline_path(root, exact_tokens, b)
        path = baseline if baseline.exists() else root / "manifest.jsonl"
    elif text_class == "unwatermarked":
        path = _baseline_path(root, exact_tokens, b)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Generate the LLM unwatermarked baseline first."
            )
    else:
        raise ValueError("--text-class must be natural or unwatermarked")

    rows = [row for row in iter_jsonl(path) if row.get("status", "completed") == "completed"]
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        rows = rows[:max_samples]
    if not rows:
        raise RuntimeError(f"No completed rows found in {path}")
    return rows


def _ids(row: dict[str, Any], field: str) -> list[int]:
    return [int(value) for value in row[field]]


def _init_brew_worker(
    model_path: str,
    secret_key: str,
    context_width: int,
    target_fpr: float,
    local_files_only: bool,
) -> None:
    global BREW_DETECTOR
    from transformers import AutoTokenizer

    from utils.model import excluded_token_ids
    from watermark.detector import DualLayerDetector
    from watermark.ecc import BCHCodec

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    BREW_DETECTOR = DualLayerDetector(
        secret_key=secret_key.encode("utf-8"),
        context_width=int(context_width),
        vocab_size=vocab_size,
        excluded_token_ids=excluded_token_ids(
            tokenizer,
            exclude_special_tokens=True,
            exclude_eos=False,
        ),
        prf_mode="paper_shared",
        ecc_codec=BCHCodec(23, 8, 3),
        presence_test="exact_binomial",
        threshold_mode="theoretical",
        target_fpr=float(target_fpr),
        fixed_z_threshold=float(norm.ppf(1.0 - float(target_fpr))),
        calibrated_threshold=None,
        primary_counting_mode="unique_context",
        unique_ngram_width=4,
        min_tokens_per_code_bit=1,
        hard_fill_value=0,
        primary_policy="tie_zero",
        max_erasure_assignments=64,
        evaluate_all_policies=False,
    )


def _score_brew_one(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
    row, text_class = payload
    assert BREW_DETECTOR is not None
    token_field = "natural_token_ids" if text_class == "natural" else "unwatermarked_token_ids"
    started = time.perf_counter()
    result = BREW_DETECTOR.detect_token_ids(_ids(row, token_field), decode_payload=False)
    elapsed = time.perf_counter() - started
    primary = result.counting[result.primary_counting_mode]
    return {
        "schema_version": 1,
        "sample_id": str(row["sample_id"]),
        "text_class": text_class,
        "status": "completed",
        "algorithm": "brew",
        "input_mode": result.input_mode,
        "num_scored_tokens": int(primary.scored_tokens),
        "upper_count": int(primary.upper_hits),
        "z_score": float(primary.z_score),
        "p_value": float(primary.exact_p_value),
        "detection_time_seconds": float(elapsed),
    }


def _init_mpac_worker(
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
) -> None:
    global MPAC_DETECTOR, MPAC_DEVICE, MPAC_MESSAGE_BITS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoTokenizer

    from experiments.mpac_comparison import embedded_message_length, import_mpac, make_detector

    torch.set_num_threads(1)
    MPAC_DEVICE = torch.device("cpu")
    args = argparse.Namespace(
        message_length=int(message_length),
        mpac_ecc=mpac_ecc,
        base=int(base),
        gamma=float(gamma),
        seeding_scheme=seeding_scheme,
        ignore_repeated_ngrams=bool(ignore_repeated_ngrams),
        use_position_prf=bool(use_position_prf),
        use_fixed_position=bool(use_fixed_position),
    )
    _, detector_cls = import_mpac(Path(mb_repo))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    MPAC_DETECTOR = make_detector(args, tokenizer, detector_cls, MPAC_DEVICE)
    MPAC_MESSAGE_BITS = "0" * embedded_message_length(args)


def _score_mpac_one(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
    row, text_class = payload
    assert MPAC_DETECTOR is not None
    assert MPAC_DEVICE is not None
    import torch

    from experiments.mpac_comparison import _dummy_positions

    token_field = "natural_token_ids" if text_class == "natural" else "unwatermarked_token_ids"
    token_ids = _ids(row, token_field)
    MPAC_DETECTOR.position_increment = 0
    started = time.perf_counter()
    score = MPAC_DETECTOR.detect(
        tokenized_text=torch.tensor(token_ids, dtype=torch.long, device=MPAC_DEVICE),
        return_prediction=False,
        convert_to_float=True,
        return_z_at_T=False,
        return_bit_match=False,
        return_p_value=True,
        message=MPAC_MESSAGE_BITS,
        position=_dummy_positions(
            token_ids,
            int(MPAC_DETECTOR.context_width),
            bool(MPAC_DETECTOR.self_salt),
        ),
    )
    elapsed = time.perf_counter() - started
    return {
        "schema_version": 1,
        "sample_id": str(row["sample_id"]),
        "text_class": text_class,
        "status": "completed",
        "algorithm": "mpac",
        "z_score": float(score["z_score"]),
        "p_value": float(score["p_value"]),
        "num_scored_tokens": int(score["num_tokens_scored"]),
        "num_green_tokens": int(score["num_green_tokens"]),
        "green_fraction": float(score["green_fraction"]),
        "position_acc": float(score.get("position_acc", float("nan"))),
        "detection_time_seconds": float(elapsed),
    }


def _score_rows(
    *,
    args: argparse.Namespace,
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    workers = min(int(args.workers), os.cpu_count() or 1)
    payloads = [(row, args.text_class) for row in rows]
    if args.algorithm == "brew":
        initializer = _init_brew_worker
        initargs = (
            args.model_path,
            args.secret_key,
            args.context_width,
            args.target_fprs[0],
            args.local_files_only,
        )
        scorer = _score_brew_one
    elif args.algorithm == "mpac":
        initializer = _init_mpac_worker
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
        )
        scorer = _score_mpac_one
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs) as pool:
        futures = [pool.submit(scorer, payload) for payload in payloads]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Score {args.algorithm} {args.text_class}",
            unit="sample",
        ):
            results.append(future.result())
    order = {str(row["sample_id"]): index for index, row in enumerate(rows)}
    results.sort(key=lambda row: order.get(str(row["sample_id"]), len(order)))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score shared negative FPR with fixed theoretical z thresholds.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--algorithm", choices=["brew", "mpac"], required=True)
    parser.add_argument("--text-class", choices=["natural", "unwatermarked"], required=True)
    parser.add_argument("--model-path", default="/data/yanlu/BREW/models/facebook/opt-1.3b")
    parser.add_argument("--exact-tokens", type=int, default=500)
    parser.add_argument("--b", type=int, default=8)
    parser.add_argument("--target-fprs", type=float, nargs="+", default=[0.01, 0.001])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-tag")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--secret-key", default="dual-layer-key-2026")
    parser.add_argument("--context-width", type=int, default=4)

    parser.add_argument("--mb-repo", default="/tmp/mb-lm-watermarking")
    parser.add_argument("--message-length", type=int, default=8)
    parser.add_argument("--mpac-ecc", choices=["none", "bch23"], default="none")
    parser.add_argument("--base", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--seeding-scheme", default="lefthash")
    parser.add_argument("--ignore-repeated-ngrams", action="store_true")
    parser.add_argument("--use-position-prf", action="store_true")
    parser.add_argument("--use-fixed-position", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.algorithm == "mpac" and args.mpac_ecc == "bch23" and args.message_length != 8:
        raise ValueError("--mpac-ecc bch23 requires --message-length 8")

    root = Path(args.root)
    rows = _load_rows(
        root=root,
        text_class=args.text_class,
        exact_tokens=int(args.exact_tokens),
        b=int(args.b),
        max_samples=args.max_samples,
    )
    output_dir = root / "runs" / f"T{int(args.exact_tokens)}_b{int(args.b)}" / args.algorithm
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = build_output_prefix(
        algorithm=args.algorithm,
        text_class=args.text_class,
        output_tag=args.output_tag,
        max_samples=args.max_samples,
    )
    scores_path = output_dir / f"{prefix}_scores.jsonl"
    summary_path = output_dir / f"{prefix}_fixed_threshold_summary.json"
    empirical_path = output_dir / f"{prefix}_empirical_thresholds.json"
    if scores_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {scores_path}. Use --overwrite to replace it.")

    started = time.perf_counter()
    results = _score_rows(args=args, rows=rows)
    elapsed = time.perf_counter() - started
    _write_jsonl(scores_path, results)
    scores = [float(row["z_score"]) for row in results]
    fixed = summarize_fixed_thresholds(scores, target_fprs=args.target_fprs)
    empirical = summarize_empirical_thresholds(scores, target_fprs=args.target_fprs)
    summary = {
        "root": str(root),
        "algorithm": args.algorithm,
        "text_class": args.text_class,
        "exact_tokens": int(args.exact_tokens),
        "b": int(args.b),
        "num_samples": len(results),
        "elapsed_seconds": elapsed,
        "workers": min(int(args.workers), os.cpu_count() or 1),
        "scores_path": str(scores_path),
        "fixed_threshold_results": fixed,
        "empirical_threshold_results": empirical,
        "num_scored_tokens_values": sorted({int(row["num_scored_tokens"]) for row in results}),
    }
    _write_json(summary_path, summary)
    _write_json(empirical_path, {"empirical_threshold_results": empirical})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
