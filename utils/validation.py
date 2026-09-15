from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from watermark.config import ExperimentConfig
from watermark.ecc import BCHCodec


def validate_batch_execution_args(args: Namespace) -> None:
    if getattr(args, "import_v1_completed", False) and not getattr(
        args, "resume", False
    ):
        raise ValueError("--import-v1-completed requires --resume")


def validate_experiment_config(
    config: ExperimentConfig,
    *,
    check_model_path: bool = True,
    check_cuda: bool = False,
) -> BCHCodec:
    if not config.output.run_id.strip():
        raise ValueError("run_id must not be empty")
    if not config.watermark.secret_key:
        raise ValueError("secret_key must not be empty")
    if config.watermark.context_width <= 0:
        raise ValueError("context_width must be positive")
    if config.watermark.partition_mode != "exact_permutation":
        raise ValueError("Only exact_permutation is implemented")
    if config.watermark.allocation_mode == "balanced_permutation":
        raise NotImplementedError("balanced_permutation is reserved for Phase 2")
    if config.watermark.allocation_mode != "hash_mod":
        raise ValueError("Only hash_mod is implemented")
    if config.generation.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if config.dataset.max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if config.dataset.sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    if config.generation.do_sample and config.generation.temperature <= 0:
        raise ValueError("temperature must be positive when sampling")
    if not 0.0 < config.generation.top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if config.generation.top_k is not None and config.generation.top_k <= 0:
        raise ValueError("top_k must be positive")
    if (
        config.watermark.candidate_top_k is not None
        and config.watermark.candidate_top_k <= 0
    ):
        raise ValueError("candidate_top_k must be positive")
    if config.watermark.seeding_scheme not in {"history", "selfhash"}:
        raise ValueError("seeding_scheme must be history or selfhash")
    if config.watermark.partition_engine not in {"v1", "v2"}:
        raise ValueError("partition_engine must be v1 or v2")
    if (
        config.watermark.seeding_scheme == "selfhash"
        and config.watermark.partition_engine != "v2"
    ):
        raise ValueError("selfhash requires partition_engine='v2'")
    if (
        config.watermark.seeding_scheme == "selfhash"
        and config.watermark.candidate_top_k is None
    ):
        raise ValueError("selfhash requires candidate_top_k")
    if not 0.0 < config.detection.target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    if config.detection.presence_test not in {"exact_binomial", "z_score"}:
        raise ValueError("presence_test must be exact_binomial or z_score")
    if config.detection.threshold_mode == "calibrated" and config.detection.calibrated_threshold is None:
        raise ValueError("calibrated threshold mode requires calibrated_threshold")
    if config.decoding.primary_policy not in {"tie_zero", "strict", "hard_fill", "error_erasure"}:
        raise ValueError(
            "primary_policy must be tie_zero, strict, hard_fill, or error_erasure"
        )
    if config.decoding.min_tokens_per_code_bit <= 0:
        raise ValueError("min_tokens_per_code_bit must be positive")
    if config.decoding.max_erasure_assignments <= 0:
        raise ValueError("max_erasure_assignments must be positive")
    if config.decoding.hard_fill_value not in {0, 1, "prf"}:
        raise ValueError("hard_fill_value must be 0, 1, or prf")
    if check_model_path:
        path = Path(config.model.path).expanduser()
        if config.model.local_files_only and not path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {path}")
        if path.exists() and path.is_dir() and not (path / "config.json").exists():
            raise FileNotFoundError(f"Model directory has no config.json: {path}")
    if check_cuda and config.model.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        capability = torch.cuda.get_device_capability(0)
        if capability == (12, 0) and "sm_120" not in torch.cuda.get_arch_list():
            raise RuntimeError(
                "Blackwell sm_120 GPU detected, but this PyTorch build does not contain sm_120 kernels. "
                "Install the CUDA 12.8 PyTorch wheel."
            )
    return BCHCodec(config.ecc.n, config.ecc.k, config.ecc.t)


def compare_experiment_configs(existing: dict[str, Any], current: dict[str, Any]) -> list[str]:
    ignored = {"created_at"}
    differences: list[str] = []

    def walk(left: Any, right: Any, prefix: str = "") -> None:
        if prefix.split(".")[-1] in ignored:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                path = f"{prefix}.{key}" if prefix else key
                if key not in left:
                    differences.append(f"{path}: existing=<missing>, current={right[key]!r}")
                elif key not in right:
                    differences.append(f"{path}: existing={left[key]!r}, current=<missing>")
                else:
                    walk(left[key], right[key], path)
            return
        if left != right:
            differences.append(f"{prefix}: existing={left!r}, current={right!r}")

    walk(existing, current)
    return differences


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value
