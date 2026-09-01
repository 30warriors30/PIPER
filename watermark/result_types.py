from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DecodeResult:
    status: str
    message_bits: tuple[int, ...] | None
    corrected_errors: int | None
    error: str | None = None
    codeword_bits: tuple[int, ...] | None = None
    erasure_count: int | None = None
    known_position_errors: int | None = None
    distance_cost: int | None = None
    candidate_count: int | None = None
    assignments_tested: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CountingResult:
    mode: str
    scored_tokens: int
    upper_hits: int
    z_score: float
    exact_p_value: float
    n0: tuple[int, ...]
    n1: tuple[int, ...]
    erasures: tuple[int, ...]
    null_probability: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionResult:
    input_mode: str
    threshold: float
    detected: bool
    primary_counting_mode: str
    counting: dict[str, CountingResult]
    strict_decode: DecodeResult
    hard_fill_decode: DecodeResult
    error_erasure_decode: DecodeResult
    primary_policy: str
    gated_message_bits: tuple[int, ...] | None
    tie_zero_decode: DecodeResult | None = None
    presence_test: str = "z_score"
    alpha: float = 0.01
    decoder_invoked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
