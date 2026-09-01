from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_length_sweep import _load_completed, generate_shared_baseline, run_id
from utils.model import load_model_and_tokenizer
from watermark.config import ModelConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate shared unwatermarked LLM negatives from an existing fixed-length manifest."
    )
    parser.add_argument("--root", required=True, help="Experiment root containing manifest.jsonl.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--exact-tokens", type=int, default=500)
    parser.add_argument("--b", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _count_completed(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for row in _load_completed(path) if row.get("status") == "completed")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    run_dir = root / "runs" / run_id(args.exact_tokens, args.b)
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = run_dir / "baseline.jsonl"
    if args.overwrite and baseline_path.exists():
        baseline_path.unlink()

    manifest = _load_completed(manifest_path)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        manifest = manifest[: args.max_samples]
    if not manifest:
        raise RuntimeError(f"No completed rows found in {manifest_path}")

    before = _count_completed(baseline_path)
    model, tokenizer, device = load_model_and_tokenizer(
        ModelConfig(
            path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
        )
    )
    runtime_args = SimpleNamespace(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
    )
    baseline = generate_shared_baseline(
        exact_tokens=int(args.exact_tokens),
        capacity_bits=int(args.b),
        args=runtime_args,
        root=root,
        manifest=manifest,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    after = _count_completed(baseline_path)
    print(
        {
            "root": str(root),
            "baseline_path": str(baseline_path),
            "manifest_rows": len(manifest),
            "completed_before": before,
            "completed_after": after,
            "processed_this_run": max(0, after - before),
            "loaded_baseline_rows": len(baseline),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
