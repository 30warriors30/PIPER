from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from watermark.execution import BatchExecutionConfig


@dataclass(frozen=True)
class ModelConfig:
    path: str
    device: str = "cuda"
    dtype: str = "float16"
    local_files_only: bool = True


@dataclass(frozen=True)
class DatasetConfig:
    kind: str = "c4"
    name: str = "allenai/c4"
    config: str | None = "realnewslike"
    split: str = "validation"
    streaming: bool = True
    max_samples: int = 200
    sample_offset: int = 0
    path: str | None = None
    prompt_field: str = "prompt"
    completion_field: str | None = "completion"
    id_field: str | None = None


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 200
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = None
    global_seed: int = 42
    message_seed: int = 42
    save_token_ids: bool = True


@dataclass(frozen=True)
class WatermarkConfig:
    secret_key: str
    context_width: int = 4
    presence_mode: str = "soft"
    delta_presence: float = 0.0
    delta_payload: float = 2.0
    prf_mode: str = "paper_shared"
    partition_mode: str = "exact_permutation"
    allocation_mode: str = "hash_mod"
    exclude_special_tokens: bool = True
    exclude_eos: bool = False
    candidate_top_k: int | None = None
    seeding_scheme: str = "history"
    partition_engine: str = "v1"


@dataclass(frozen=True)
class ECCConfig:
    n: int = 23
    k: int = 8
    t: int = 3


@dataclass(frozen=True)
class DetectionConfig:
    presence_test: str = "exact_binomial"
    threshold_mode: str = "theoretical"
    target_fpr: float = 0.01
    fixed_threshold: float = 2.326347874
    calibrated_threshold: float | None = None
    counting_mode: str = "unique_context"
    unique_ngram_width: int = 4


@dataclass(frozen=True)
class DecodingConfig:
    primary_policy: str = "tie_zero"
    min_tokens_per_code_bit: int = 1
    hard_fill_value: int | str = 0
    max_erasure_assignments: int = 64
    evaluate_all_policies: bool = False


@dataclass(frozen=True)
class OutputConfig:
    root: str = "outputs"
    run_id: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    dataset: DatasetConfig
    generation: GenerationConfig
    watermark: WatermarkConfig
    ecc: ECCConfig
    detection: DetectionConfig
    decoding: DecodingConfig
    output: OutputConfig
    execution: BatchExecutionConfig = field(default_factory=BatchExecutionConfig)
    schema_version: int = 1

    @property
    def run_dir(self) -> Path:
        return Path(self.output.root) / self.output.run_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        if not isinstance(data, dict):
            raise TypeError("experiment data must be an object")
        return cls(
            model=ModelConfig(**data["model"]),
            dataset=DatasetConfig(**data["dataset"]),
            generation=GenerationConfig(**data["generation"]),
            watermark=WatermarkConfig(**data["watermark"]),
            ecc=ECCConfig(**data["ecc"]),
            detection=DetectionConfig(**data.get("detection", {})),
            decoding=DecodingConfig(**data.get("decoding", {})),
            output=OutputConfig(**data["output"]),
            execution=BatchExecutionConfig(**data["execution"]),
            schema_version=int(data.get("schema_version", 1)),
        )
