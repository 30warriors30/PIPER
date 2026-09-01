from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet

import numpy as np


@dataclass(frozen=True)
class Partition:
    bit0: tuple[int, ...]
    bit1: tuple[int, ...]
    lower_a: tuple[int, ...]
    lower_b: tuple[int, ...]

    @property
    def upper(self) -> tuple[int, ...]:
        return self.bit0 + self.bit1

    @property
    def lower(self) -> tuple[int, ...]:
        return self.lower_a + self.lower_b


class ExactPermutationPartitioner:
    """Create an exact deterministic four-way vocabulary partition on CPU."""

    def __init__(self) -> None:
        self._allowed_cache: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}

    def _allowed_tokens(
        self,
        *,
        vocab_size: int,
        excluded_ids: AbstractSet[int] | None,
    ) -> np.ndarray:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        excluded_key = tuple(sorted(int(token_id) for token_id in (excluded_ids or set())))
        if any(token_id < 0 or token_id >= vocab_size for token_id in excluded_key):
            raise ValueError("excluded token ID is outside the vocabulary")
        cache_key = (int(vocab_size), excluded_key)
        cached = self._allowed_cache.get(cache_key)
        if cached is not None:
            return cached
        excluded = set(excluded_key)
        allowed = np.array(
            [token_id for token_id in range(vocab_size) if token_id not in excluded],
            dtype=np.int64,
        )
        if len(allowed) < 4:
            raise ValueError("At least four allowed vocabulary tokens are required")
        self._allowed_cache[cache_key] = allowed
        return allowed

    def partition(
        self,
        *,
        seed: int,
        vocab_size: int,
        excluded_ids: AbstractSet[int] | None = None,
    ) -> Partition:
        allowed = self._allowed_tokens(vocab_size=vocab_size, excluded_ids=excluded_ids)
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        permuted = rng.permutation(allowed).tolist()
        quarter = len(permuted) // 4
        bit0 = tuple(int(x) for x in permuted[:quarter])
        bit1 = tuple(int(x) for x in permuted[quarter : 2 * quarter])
        remaining = permuted[2 * quarter :]
        split = len(remaining) // 2
        lower_a = tuple(int(x) for x in remaining[:split])
        lower_b = tuple(int(x) for x in remaining[split:])
        return Partition(bit0=bit0, bit1=bit1, lower_a=lower_a, lower_b=lower_b)

    def region_for_token(
        self,
        *,
        seed: int,
        vocab_size: int,
        excluded_ids: AbstractSet[int] | None = None,
        token_id: int,
    ) -> int | None:
        token = int(token_id)
        if token < 0 or token >= vocab_size:
            return None
        excluded = {int(value) for value in (excluded_ids or set())}
        if token in excluded:
            return None
        allowed = self._allowed_tokens(vocab_size=vocab_size, excluded_ids=excluded_ids)
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        permuted = rng.permutation(allowed)
        positions = np.nonzero(permuted == token)[0]
        if len(positions) == 0:
            return None
        position = int(positions[0])
        quarter = len(permuted) // 4
        if position < quarter:
            return 0
        if position < 2 * quarter:
            return 1
        return None
