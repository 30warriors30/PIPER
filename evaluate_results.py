from __future__ import annotations

import sys
from pathlib import Path

from evaluation.metrics import evaluate_records
from evaluation.report import write_reports
from utils.arguments import evaluation_parser
from utils.io import iter_jsonl


def main() -> int:
    args = evaluation_parser().parse_args()
    run_dir = Path(args.run_dir)
    detection_path = run_dir / args.detections_file
    if not detection_path.exists():
        raise FileNotFoundError(detection_path)
    print(f"Evaluating {detection_path}")
    if args.dry_run:
        print("Dry run completed.")
        return 0
    metrics = evaluate_records(list(iter_jsonl(detection_path)), input_mode=args.input_mode)
    write_reports(metrics, run_dir)
    print(run_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
