from __future__ import annotations

import argparse
from pathlib import Path

from watermark.config import (
    DatasetConfig,
    DecodingConfig,
    DetectionConfig,
    ECCConfig,
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
    OutputConfig,
    WatermarkConfig,
)


def _bool_action() -> type[argparse.Action]:
    return argparse.BooleanOptionalAction


def generation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paired watermarked, unwatermarked, and natural continuations."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--local-files-only", action=_bool_action(), default=True)

    parser.add_argument("--dataset", default="c4", choices=["c4", "opengen", "jsonl", "json"])
    parser.add_argument("--dataset-name", default="allenai/c4")
    parser.add_argument("--dataset-config", default="realnewslike")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--streaming", action=_bool_action(), default=True)
    parser.add_argument("--dataset-path")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--completion-field", default="completion")
    parser.add_argument("--id-field")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--sample-offset", type=int, default=0)

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Exact continuation length for watermarked, unwatermarked, and natural texts.",
    )
    parser.add_argument("--do-sample", action=_bool_action(), default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--message-seed", type=int, default=42)

    parser.add_argument("--secret-key", required=True)
    parser.add_argument("--context-width", type=int, default=4)
    parser.add_argument("--presence-mode", default="soft", choices=["hard", "soft"])
    parser.add_argument("--delta-presence", type=float, default=0.0)
    parser.add_argument("--delta-payload", type=float, default=2.0)
    parser.add_argument("--prf-mode", default="paper_shared", choices=["paper_shared", "domain_separated"])
    parser.add_argument("--partition-mode", default="exact_permutation", choices=["exact_permutation"])
    parser.add_argument("--allocation-mode", default="hash_mod", choices=["hash_mod", "balanced_permutation"])
    parser.add_argument("--exclude-special-tokens", action=_bool_action(), default=True)
    parser.add_argument("--exclude-eos", action=_bool_action(), default=False)

    parser.add_argument("--ecc-n", type=int, default=23)
    parser.add_argument("--ecc-k", type=int, default=8)
    parser.add_argument("--ecc-t", type=int, default=3)

    parser.add_argument("--threshold-mode", default="theoretical", choices=["theoretical", "fixed", "calibrated"])
    parser.add_argument("--presence-test", default="exact_binomial", choices=["exact_binomial", "z_score"])
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--fixed-threshold", type=float, default=2.326347874)
    parser.add_argument("--calibrated-threshold", type=float)
    parser.add_argument("--counting-mode", default="unique_context", choices=["all_tokens", "unique_context", "unique_ngram"])
    parser.add_argument("--unique-ngram-width", type=int, default=4)
    parser.add_argument(
        "--primary-decoding-policy",
        default="tie_zero",
        choices=["tie_zero", "strict", "hard_fill", "error_erasure"],
    )
    parser.add_argument("--min-tokens-per-code-bit", type=int, default=1)
    parser.add_argument("--hard-fill-value", default="0", choices=["0", "1", "prf"])
    parser.add_argument("--max-erasure-assignments", type=int, default=64)

    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-id", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def generation_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    hard_fill: int | str = args.hard_fill_value
    if hard_fill in {"0", "1"}:
        hard_fill = int(hard_fill)
    completion_field = args.completion_field
    if completion_field in {"", "none", "None"}:
        completion_field = None
    dataset_config = args.dataset_config
    if dataset_config in {"", "none", "None"}:
        dataset_config = None
    return ExperimentConfig(
        model=ModelConfig(
            path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
        ),
        dataset=DatasetConfig(
            kind=args.dataset,
            name=args.dataset_name,
            config=dataset_config,
            split=args.dataset_split,
            streaming=args.streaming,
            max_samples=args.max_samples,
            sample_offset=args.sample_offset,
            path=args.dataset_path,
            prompt_field=args.prompt_field,
            completion_field=completion_field,
            id_field=args.id_field,
        ),
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            global_seed=args.global_seed,
            message_seed=args.message_seed,
        ),
        watermark=WatermarkConfig(
            secret_key=args.secret_key,
            context_width=args.context_width,
            presence_mode=args.presence_mode,
            delta_presence=args.delta_presence,
            delta_payload=args.delta_payload,
            prf_mode=args.prf_mode,
            partition_mode=args.partition_mode,
            allocation_mode=args.allocation_mode,
            exclude_special_tokens=args.exclude_special_tokens,
            exclude_eos=args.exclude_eos,
        ),
        ecc=ECCConfig(n=args.ecc_n, k=args.ecc_k, t=args.ecc_t),
        detection=DetectionConfig(
            presence_test=args.presence_test,
            threshold_mode=args.threshold_mode,
            target_fpr=args.target_fpr,
            fixed_threshold=args.fixed_threshold,
            calibrated_threshold=args.calibrated_threshold,
            counting_mode=args.counting_mode,
            unique_ngram_width=args.unique_ngram_width,
        ),
        decoding=DecodingConfig(
            primary_policy=args.primary_decoding_policy,
            min_tokens_per_code_bit=args.min_tokens_per_code_bit,
            hard_fill_value=hard_fill,
            max_erasure_assignments=args.max_erasure_assignments,
        ),
        output=OutputConfig(root=args.output_root, run_id=args.run_id),
    )


def detection_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect dual-layer watermarks.")
    parser.add_argument("--input-type", required=True, choices=["samples", "text"])
    parser.add_argument("--run-dir")
    parser.add_argument("--input-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-file")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--prompt-field")
    parser.add_argument("--continuation-field")
    parser.add_argument("--model-path")
    parser.add_argument("--secret-key")
    parser.add_argument("--context-width", type=int, default=4)
    parser.add_argument("--prf-mode", default="paper_shared", choices=["paper_shared", "domain_separated"])
    parser.add_argument("--partition-mode", default="exact_permutation", choices=["exact_permutation"])
    parser.add_argument("--allocation-mode", default="hash_mod", choices=["hash_mod", "balanced_permutation"])
    parser.add_argument("--ecc-n", type=int, default=23)
    parser.add_argument("--ecc-k", type=int, default=8)
    parser.add_argument("--ecc-t", type=int, default=3)
    parser.add_argument("--threshold-mode", default="theoretical", choices=["theoretical", "fixed", "calibrated"])
    parser.add_argument("--presence-test", default="exact_binomial", choices=["exact_binomial", "z_score"])
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--fixed-threshold", type=float, default=2.326347874)
    parser.add_argument("--calibration-file")
    parser.add_argument("--counting-mode", default="unique_context", choices=["all_tokens", "unique_context", "unique_ngram"])
    parser.add_argument("--unique-ngram-width", type=int, default=4)
    parser.add_argument(
        "--primary-decoding-policy",
        default="tie_zero",
        choices=["tie_zero", "strict", "hard_fill", "error_erasure"],
    )
    parser.add_argument("--min-tokens-per-code-bit", type=int, default=1)
    parser.add_argument("--hard-fill-value", default="0", choices=["0", "1", "prf"])
    parser.add_argument("--max-erasure-assignments", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def evaluation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate detection results.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--detections-file", default="detections.jsonl")
    parser.add_argument("--input-mode", default="all", choices=["all", "known_boundary", "blind_text"])
    parser.add_argument(
        "--decoding-policy",
        default="tie_zero",
        choices=["tie_zero", "strict", "hard_fill", "error_erasure"],
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def calibration_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate the Z-score threshold from negative samples.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--detections-file", default="detections.jsonl")
    parser.add_argument("--negative-source", default="unwatermarked", choices=["unwatermarked", "natural", "combined"])
    parser.add_argument("--input-mode", default="known_boundary", choices=["known_boundary", "blind_text"])
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--output-file", default="calibration.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def absolute_run_dir(value: str) -> Path:
    return Path(value).expanduser().resolve()
