from __future__ import annotations

import json
import sys
from pathlib import Path

from utils.arguments import batch_execution_config_from_args, detection_parser
from utils.detection import run_samples_detection, run_text_detection


def _calibrated_threshold(path: str | None) -> float | None:
    if not path:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(value["calibrated_threshold"])


def main() -> int:
    args = detection_parser().parse_args()
    execution = batch_execution_config_from_args(args)
    hard_fill: int | str = int(args.hard_fill_value) if args.hard_fill_value in {"0", "1"} else "prf"
    calibrated = _calibrated_threshold(args.calibration_file)
    if args.threshold_mode == "calibrated" and calibrated is None:
        raise ValueError("--threshold-mode calibrated requires --calibration-file")
    if args.input_type == "samples":
        if not args.run_dir:
            raise ValueError("--run-dir is required for --input-type samples")
        print(f"Detecting project samples in {args.run_dir}")
        if args.dry_run:
            print("Dry run completed.")
            return 0
        result = run_samples_detection(
            Path(args.run_dir),
            presence_test=args.presence_test,
            threshold_mode=args.threshold_mode,
            target_fpr=args.target_fpr,
            fixed_threshold=args.fixed_threshold,
            calibrated_threshold=calibrated,
            counting_mode=args.counting_mode,
            unique_ngram_width=args.unique_ngram_width,
            primary_policy=args.primary_decoding_policy,
            min_tokens_per_code_bit=args.min_tokens_per_code_bit,
            hard_fill_value=hard_fill,
            max_erasure_assignments=args.max_erasure_assignments,
            output_file=args.output_file or "detections.jsonl",
            resume=args.resume,
            overwrite=args.overwrite,
            execution=execution,
        )
    else:
        required = {
            "input_file": args.input_file,
            "output_dir": args.output_dir,
            "model_path": args.model_path,
            "secret_key": args.secret_key,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("Missing required text-mode arguments: " + ", ".join(missing))
        print(f"Detecting external text from {args.input_file}")
        if args.dry_run:
            print("Dry run completed.")
            return 0
        result = run_text_detection(
            input_file=Path(args.input_file),
            output_dir=Path(args.output_dir),
            output_file=args.output_file or "detections.jsonl",
            text_field=args.text_field,
            prompt_field=args.prompt_field,
            continuation_field=args.continuation_field,
            model_path=args.model_path,
            secret_key=args.secret_key,
            context_width=args.context_width,
            prf_mode=args.prf_mode,
            partition_mode=args.partition_mode,
            allocation_mode=args.allocation_mode,
            seeding_scheme=args.seeding_scheme,
            partition_engine=args.partition_engine,
            ecc_n=args.ecc_n,
            ecc_k=args.ecc_k,
            ecc_t=args.ecc_t,
            presence_test=args.presence_test,
            threshold_mode=args.threshold_mode,
            target_fpr=args.target_fpr,
            fixed_threshold=args.fixed_threshold,
            calibrated_threshold=calibrated,
            counting_mode=args.counting_mode,
            unique_ngram_width=args.unique_ngram_width,
            primary_policy=args.primary_decoding_policy,
            min_tokens_per_code_bit=args.min_tokens_per_code_bit,
            hard_fill_value=hard_fill,
            max_erasure_assignments=args.max_erasure_assignments,
            overwrite=args.overwrite,
            execution=execution,
        )
    print(result)
    return 0 if result.get("failed", 0) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
