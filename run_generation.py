from __future__ import annotations

import sys

from utils.arguments import generation_config_from_args, generation_parser
from utils.generation import generation_summary, run_generation
from utils.validation import validate_experiment_config


def main() -> int:
    parser = generation_parser()
    args = parser.parse_args()
    config = generation_config_from_args(args)
    validate_experiment_config(config, check_model_path=True, check_cuda=True)
    print(generation_summary(config), flush=True)
    if args.dry_run:
        print("Dry run completed. No model was loaded and no output was written.")
        return 0
    result = run_generation(config, resume=args.resume, overwrite=args.overwrite)
    print(result)
    return 0 if result["failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
