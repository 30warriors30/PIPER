from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.arguments import config_from_args, experiment_parser, points_from_args
from experiments.pareto import aggregate_experiment


def main() -> int:
    parser = experiment_parser("Aggregate existing Pareto experiment outputs.")
    args = parser.parse_args()
    config = config_from_args(args)
    points = points_from_args(args)
    if args.dry_run:
        print(f"Would aggregate {len(points)} operating points from {config.experiment_dir}")
        return 0
    print(aggregate_experiment(config, points))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
