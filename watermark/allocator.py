from __future__ import annotations

from typing import Protocol


class BitAllocator(Protocol):
    def allocate(self, *, seed: int, code_length: int) -> int: ...


class HashModuloAllocator:
    def allocate(self, *, seed: int, code_length: int) -> int:
        if code_length <= 0:
            raise ValueError("code_length must be positive")
        return int(seed) % int(code_length)


def build_allocator(mode: str) -> BitAllocator:
    if mode == "hash_mod":
        return HashModuloAllocator()
    if mode == "balanced_permutation":
        raise NotImplementedError("balanced_permutation is reserved for Phase 2")
    raise ValueError(f"Unsupported allocation mode: {mode}")
