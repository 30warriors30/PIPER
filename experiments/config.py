from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
    exact_tokens: int = 200
    context_width: int = 4
    temperature: float = 1.0
    top_p: float = 0.95
    global_seed: int = 42
    message_seed: int = 42
    target_fpr: float = 0.01
    presence_test: str = "exact_binomial"
    ecc_n: int = 23
    ecc_k: int = 8
    ecc_t: int = 3
    prf_mode: str = "paper_shared"
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

    def __post_init__(self) -> None:
        if self.calibration_samples <= 0:
            raise ValueError("calibration_samples must be positive")
        if self.test_samples <= 0:
            raise ValueError("test_samples must be positive")
        if self.exact_tokens <= 0:
            raise ValueError("exact_tokens must be positive")
        if self.context_width <= 0:
            raise ValueError("context_width must be positive")
        if not 0 < self.target_fpr < 1:
            raise ValueError("target_fpr must be in (0, 1)")
        if self.presence_test not in {"exact_binomial", "z_score"}:
            raise ValueError("presence_test must be exact_binomial or z_score")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")

    @property
    def experiment_dir(self) -> Path:
        return Path(self.output_root) / self.experiment_id

    @property
    def total_manifest_samples(self) -> int:
        return self.calibration_samples + self.test_samples

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["output_root"] = str(self.output_root)
        value["experiment_dir"] = str(self.experiment_dir)
        return value
