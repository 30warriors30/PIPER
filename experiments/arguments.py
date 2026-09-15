from __future__ import annotations

import argparse
from pathlib import Path

from experiments.config import ParetoExperimentConfig
from experiments.presets import preset_points
from utils.arguments import (
    add_batch_execution_arguments,
    batch_execution_config_from_args,
)


def experiment_parser(description: str = "Run dual-layer watermark Pareto experiments.") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--secret-key", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-root", default="outputs/experiments")
    parser.add_argument("--preset", default="paper", choices=["paper", "smoke", "pilot", "full"])
    parser.add_argument("--stage", default="all", choices=["all", "manifest", "baseline", "sweep", "aggregate"])
    parser.add_argument("--only-point", action="append", default=[])
    parser.add_argument("--max-points", type=int)

    parser.add_argument("--calibration-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument(
        "--limit-test",
        type=int,
        help=(
            "Use only the first N test manifest rows for watermarked generation, "
            "detection, quality, and TPR/payload metrics. Shared negative FPR "
            "evaluation still uses all --test-samples rows."
        ),
    )
    parser.add_argument("--exact-tokens", type=int, default=200)
    parser.add_argument("--context-width", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--message-seed", type=int, default=42)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--presence-test", default="exact_binomial", choices=["exact_binomial", "z_score"])

    parser.add_argument("--ecc-n", type=int, default=23)
    parser.add_argument("--ecc-k", type=int, default=8)
    parser.add_argument("--ecc-t", type=int, default=3)
    parser.add_argument("--prf-mode", default="paper_shared", choices=["paper_shared", "domain_separated"])
    parser.add_argument("--candidate-top-k", type=int, default=50)
    parser.add_argument("--seeding-scheme", default="selfhash", choices=["history", "selfhash"])
    parser.add_argument("--partition-engine", default="v2", choices=["v1", "v2"])
    parser.add_argument("--allocation-mode", default="hash_mod", choices=["hash_mod", "balanced_permutation"])
    parser.add_argument("--counting-mode", default="unique_context", choices=["all_tokens", "unique_context", "unique_ngram"])
    parser.add_argument("--max-erasure-assignments", type=int, default=64)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only")
    parser.set_defaults(local_files_only=True)

    parser.add_argument("--dataset-name", default="allenai/c4")
    parser.add_argument("--dataset-config", default="realnewslike")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--dataset-path")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--no-streaming", action="store_false", dest="dataset_streaming")
    parser.set_defaults(dataset_streaming=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--overwrite", action="store_true")
    add_batch_execution_arguments(parser)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ParetoExperimentConfig:
    execution = batch_execution_config_from_args(args)
    smoke = args.preset == "smoke"
    calibration_samples = args.calibration_samples if args.calibration_samples is not None else (20 if smoke else 200)
    test_samples = args.test_samples if args.test_samples is not None else (20 if smoke else 200)
    return ParetoExperimentConfig(
        model_path=args.model_path,
        secret_key=args.secret_key,
        output_root=Path(args.output_root).expanduser(),
        experiment_id=args.experiment_id,
        calibration_samples=calibration_samples,
        test_samples=test_samples,
        limit_test=args.limit_test,
        exact_tokens=args.exact_tokens,
        context_width=args.context_width,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        global_seed=args.global_seed,
        message_seed=args.message_seed,
        target_fpr=args.target_fpr,
        presence_test=args.presence_test,
        ecc_n=args.ecc_n,
        ecc_k=args.ecc_k,
        ecc_t=args.ecc_t,
        prf_mode=args.prf_mode,
        candidate_top_k=args.candidate_top_k,
        seeding_scheme=args.seeding_scheme,
        partition_engine=args.partition_engine,
        allocation_mode=args.allocation_mode,
        counting_mode=args.counting_mode,
        max_erasure_assignments=args.max_erasure_assignments,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_streaming=args.dataset_streaming,
        dataset_path=args.dataset_path,
        sample_offset=args.sample_offset,
        generation_batch_size=execution.generation_batch_size,
        detection_batch_size=execution.detection_batch_size,
        detection_workers=execution.detection_workers,
    )


def points_from_args(args: argparse.Namespace):
    points = list(preset_points(args.preset))
    if args.only_point:
        wanted = set(args.only_point)
        unknown = wanted - {point.point_id for point in points}
        if unknown:
            raise ValueError("Unknown point IDs: " + ", ".join(sorted(unknown)))
        points = [point for point in points if point.point_id in wanted]
    if args.max_points is not None:
        if args.max_points <= 0:
            raise ValueError("max_points must be positive")
        points = points[: args.max_points]
    return tuple(points)


def print_plan(config: ParetoExperimentConfig, args: argparse.Namespace, points) -> None:
    print("=" * 72)
    print("Dual-layer watermark quality/detection Pareto experiment")
    print("=" * 72)
    print(f"Stage:              {args.stage}")
    print(f"Experiment dir:     {config.experiment_dir}")
    print(f"Model:              {config.model_path}")
    print(f"Manifest samples:   {config.total_manifest_samples}")
    print(f"Calibration/test:   {config.calibration_samples} / {config.test_samples}")
    print(f"Watermarked test:   {config.watermarked_test_samples}")
    print(f"Exact tokens:       {config.exact_tokens}")
    print(f"Partition seed:     {config.seeding_scheme}")
    print(f"Partition engine:   {config.partition_engine}")
    print(f"Raw candidate top-k: {config.candidate_top_k}")
    print(f"Sampling top-k/p:   {config.top_k} / {config.top_p}")
    print(f"Target FPR:         {config.target_fpr}")
    print(f"Presence test:      {config.presence_test}")
    print(f"Counting mode:      {config.counting_mode}")
    print(f"Operating points: {len(points)}")
    for point in points:
        print(f"  - {point.point_id}")
    print("=" * 72)
