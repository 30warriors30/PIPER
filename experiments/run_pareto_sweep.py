from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.arguments import config_from_args, experiment_parser, points_from_args, print_plan
from experiments.detection import (
    calibrate_shared_z_threshold,
    load_shared_z_threshold,
    run_operating_point_detection,
    run_shared_negative_detection,
)
from experiments.generation import ExperimentGenerator, run_operating_point_generation, run_shared_baseline
from experiments.manifest import build_manifest
from experiments.metrics import compute_operating_point_metrics
from experiments.pareto import aggregate_experiment
from experiments.quality import score_operating_point_quality, score_shared_quality
from utils.model import model_max_length


def _run_manifest(config, generator, *, resume: bool, overwrite: bool):
    return build_manifest(
        config,
        generator.tokenizer,
        model_max_length=model_max_length(generator.model, generator.tokenizer),
        resume=resume,
        overwrite=overwrite,
    )


def _run_baseline(config, generator, *, resume: bool, overwrite: bool):
    baseline = run_shared_baseline(config, generator, resume=resume, overwrite=overwrite)
    negatives = run_shared_negative_detection(
        config,
        tokenizer=generator.tokenizer,
        vocab_size=generator.vocab_size,
        resume=resume,
        overwrite=overwrite,
    )
    calibration = calibrate_shared_z_threshold(config)
    quality = score_shared_quality(
        config,
        model=generator.model,
        device=generator.device,
        resume=resume,
        overwrite=overwrite,
    )
    return {"baseline": baseline, "negative_detection": negatives, "calibration": calibration, "quality": quality}


def _run_sweep(config, points, generator, *, resume: bool, overwrite: bool):
    threshold = load_shared_z_threshold(config)
    results = {}
    for point in points:
        generated = run_operating_point_generation(
            config,
            point,
            generator,
            resume=resume,
            overwrite=overwrite,
        )
        detected = run_operating_point_detection(
            config,
            point,
            calibrated_threshold=threshold,
            tokenizer=generator.tokenizer,
            vocab_size=generator.vocab_size,
            resume=resume,
            overwrite=overwrite,
        )
        quality = score_operating_point_quality(
            config,
            point,
            model=generator.model,
            device=generator.device,
            resume=resume,
            overwrite=overwrite,
        )
        metrics = compute_operating_point_metrics(config, point)
        results[point.point_id] = {
            "generation": generated,
            "detection": detected,
            "quality": quality,
            "tpr": metrics["presence"].get("tpr"),
            "correct_attribution_rate": metrics["payload"].get("correct_attribution_rate"),
            "conditional_decoding_accuracy": metrics["payload"].get("conditional_decoding_accuracy"),
            "relative_ppl_increase": metrics["quality"].get("relative_ppl_increase"),
        }
    return results


def main() -> int:
    parser = experiment_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    points = points_from_args(args)
    print_plan(config, args, points)
    if args.dry_run:
        print("Dry run completed. No model was loaded and no files were written.")
        return 0

    generator = None
    if args.stage != "aggregate":
        generator = ExperimentGenerator(config)

    results = {}
    if args.stage in {"all", "manifest"}:
        assert generator is not None
        results["manifest"] = _run_manifest(
            config,
            generator,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    if args.stage in {"all", "baseline"}:
        assert generator is not None
        results["baseline"] = _run_baseline(
            config,
            generator,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    if args.stage in {"all", "sweep"}:
        assert generator is not None
        results["sweep"] = _run_sweep(
            config,
            points,
            generator,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    if args.stage in {"all", "aggregate"}:
        results["aggregate"] = aggregate_experiment(config, points)

    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
