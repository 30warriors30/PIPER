from __future__ import annotations

import json
import sys
from pathlib import Path

from evaluation.calibration import calibrate_threshold
from utils.arguments import calibration_parser
from utils.io import iter_jsonl, write_json


def main() -> int:
    args = calibration_parser().parse_args()
    run_dir = Path(args.run_dir)
    path = run_dir / args.detections_file
    records = [record for record in iter_jsonl(path) if record.get("status", "completed") == "completed"]
    classes = {"unwatermarked"} if args.negative_source == "unwatermarked" else ({"natural"} if args.negative_source == "natural" else {"unwatermarked", "natural"})
    scores = [
        float(record["z_score"])
        for record in records
        if record.get("input_mode") == args.input_mode and record.get("text_class") in classes
    ]
    print(f"Calibration negatives: {len(scores)}")
    if args.dry_run:
        print("Dry run completed.")
        return 0
    threshold = calibrate_threshold(scores, args.target_fpr)
    empirical = sum(score >= threshold for score in scores) / len(scores)
    output = {
        "target_fpr": args.target_fpr,
        "negative_source": args.negative_source,
        "input_mode": args.input_mode,
        "num_negative_samples": len(scores),
        "calibrated_threshold": threshold,
        "empirical_fpr": empirical,
    }
    write_json(run_dir / args.output_file, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
