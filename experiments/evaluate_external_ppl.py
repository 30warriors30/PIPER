from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.external_quality import evaluator_slug, score_external_evaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score an existing Pareto experiment with an independent OPT-family "
            "causal language model. Existing generated texts are reused; no generation is run."
        )
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--evaluator-model", required=True)
    parser.add_argument("--evaluator-name")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32", "auto"],
    )
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only")
    parser.set_defaults(local_files_only=True)
    parser.add_argument("--only-point", action="append", default=[])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _print_plan(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment_dir).expanduser()
    sweep_path = experiment_dir / "sweep_results.json"
    point_count = None
    if sweep_path.exists():
        value = json.loads(sweep_path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            point_count = len(value)
    slug = evaluator_slug(args.evaluator_model, args.evaluator_name)
    print("=" * 72)
    print("Independent OPT PPL evaluator")
    print("=" * 72)
    print(f"Experiment:       {experiment_dir}")
    print(f"Evaluator model:  {args.evaluator_model}")
    print(f"Evaluator name:   {args.evaluator_name or Path(args.evaluator_model).name}")
    print(f"Output:           {experiment_dir / 'external_quality' / slug}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Device / dtype:   {args.device} / {args.dtype}")
    print(f"Available points: {point_count if point_count is not None else 'unknown'}")
    print(f"Selected points:  {', '.join(args.only_point) if args.only_point else 'all'}")
    print("Generation:       disabled (existing texts only)")
    print("=" * 72)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    _print_plan(args)
    if args.dry_run:
        print("Dry run completed. No model was loaded and no files were written.")
        return 0
    result = score_external_evaluator(
        experiment_dir=Path(args.experiment_dir).expanduser(),
        evaluator_model_path=args.evaluator_model,
        evaluator_name=args.evaluator_name,
        batch_size=args.batch_size,
        dtype=args.dtype,
        device_name=args.device,
        local_files_only=args.local_files_only,
        resume=args.resume,
        overwrite=args.overwrite,
        only_points=tuple(args.only_point),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
