from __future__ import annotations

from dataclasses import dataclass


ENGINE_VERSION = 2


@dataclass(frozen=True)
class BatchExecutionConfig:
    generation_batch_size: int = 16
    detection_batch_size: int = 64
    detection_workers: int = 8
    engine_version: int = ENGINE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "generation_batch_size",
            "detection_batch_size",
            "detection_workers",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.engine_version) != ENGINE_VERSION:
            raise ValueError(f"engine_version must be {ENGINE_VERSION}")
