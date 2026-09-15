from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from watermark.execution import BatchExecutionConfig


def _number_slug(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "neg").replace(".", "p")


@dataclass(frozen=True, order=True)
class OperatingPoint:
    presence_mode: str
    delta_presence: float
    delta_payload: float

    def __post_init__(self) -> None:
        if self.presence_mode not in {"soft", "hard"}:
            raise ValueError("presence_mode must be 'soft' or 'hard'")
        if self.delta_presence < 0 or self.delta_payload < 0:
            raise ValueError("watermark deltas must be non-negative")
        if self.presence_mode == "hard" and self.delta_presence != 0:
            raise ValueError("hard operating points must use delta_presence=0")

    @property
    def point_id(self) -> str:
        payload = _number_slug(self.delta_payload)
        if self.presence_mode == "hard":
            return f"hard_m{payload}"
        presence = _number_slug(self.delta_presence)
        return f"soft_p{presence}_m{payload}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"point_id": self.point_id}


@dataclass(frozen=True)
class ParetoExperimentConfig:
    model_path: str
    secret_key: str
    output_root: Path
    experiment_id: str
    calibration_samples: int = 200
    test_samples: int = 200
    limit_test: int | None = None
    exact_tokens: int = 200
    context_width: int = 4
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = None
    global_seed: int = 42
    message_seed: int = 42
    target_fpr: float = 0.01
    presence_test: str = "exact_binomial"
    ecc_n: int = 23
    ecc_k: int = 8
    ecc_t: int = 3
    prf_mode: str = "paper_shared"
    candidate_top_k: int | None = None
    seeding_scheme: str = "history"
    partition_engine: str = "v1"
    allocation_mode: str = "hash_mod"
    counting_mode: str = "unique_context"
    max_erasure_assignments: int = 64
    device: str = "cuda"
    dtype: str = "float16"
    local_files_only: bool = True
    dataset_name: str = "allenai/c4"
    dataset_config: str = "realnewslike"
    dataset_split: str = "validation"
    dataset_streaming: bool = True
    dataset_path: str | None = None
    sample_offset: int = 0
    generation_batch_size: int = 16
    detection_batch_size: int = 64
    detection_workers: int = 8

    def __post_init__(self) -> None:
        if self.calibration_samples <= 0:
            raise ValueError("calibration_samples must be positive")
        if self.test_samples <= 0:
            raise ValueError("test_samples must be positive")
        if self.limit_test is not None:
            if self.limit_test <= 0:
                raise ValueError("limit_test must be positive")
            if self.limit_test > self.test_samples:
                raise ValueError("limit_test must not exceed test_samples")
        if self.exact_tokens <= 0:
            raise ValueError("exact_tokens must be positive")
        if self.context_width <= 0:
            raise ValueError("context_width must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.candidate_top_k is not None and self.candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be positive")
        if self.seeding_scheme not in {"history", "selfhash"}:
            raise ValueError("seeding_scheme must be history or selfhash")
        if self.partition_engine not in {"v1", "v2"}:
            raise ValueError("partition_engine must be v1 or v2")
        if self.seeding_scheme == "selfhash" and self.partition_engine != "v2":
            raise ValueError("selfhash requires partition_engine='v2'")
        if self.seeding_scheme == "selfhash" and self.candidate_top_k is None:
            raise ValueError("selfhash requires candidate_top_k")
        if not 0 < self.target_fpr < 1:
            raise ValueError("target_fpr must be in (0, 1)")
        if self.presence_test not in {"exact_binomial", "z_score"}:
            raise ValueError("presence_test must be exact_binomial or z_score")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        BatchExecutionConfig(
            generation_batch_size=self.generation_batch_size,
            detection_batch_size=self.detection_batch_size,
            detection_workers=self.detection_workers,
        )

    @property
    def experiment_dir(self) -> Path:
        return Path(self.output_root) / self.experiment_id

    @property
    def total_manifest_samples(self) -> int:
        return self.calibration_samples + self.test_samples

    @property
    def watermarked_test_samples(self) -> int:
        return self.test_samples if self.limit_test is None else self.limit_test

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["output_root"] = str(self.output_root)
        value["experiment_dir"] = str(self.experiment_dir)
        return value
