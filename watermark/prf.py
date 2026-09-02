from __future__ import annotations

import hashlib
from collections.abc import Sequence

MASK64 = (1 << 64) - 1
ROUND_DOMAINS = (
    0x243F6A8885A308D3,
    0x13198A2E03707344,
    0xA4093822299F31D0,
    0x082EFA98EC4E6C89,
    0x452821E638D01377,
    0xBE5466CF34E90C6C,
)


def _avalanche64(value: int) -> int:
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & MASK64
    return value ^ (value >> 31)


def partition_round_keys(base_seed: int) -> tuple[int, ...]:
    return tuple(
        _avalanche64((int(base_seed) ^ domain) & MASK64)
        for domain in ROUND_DOMAINS
    )


class KeyedPRF:
    """Deterministic keyed PRF based on BLAKE2b."""

    def __init__(self, key: bytes) -> None:
        if not key:
            raise ValueError("PRF key must not be empty")
        self.key = bytes(key)

    def seed(self, context_ids: Sequence[int], domain: str) -> int:
        if not domain:
            raise ValueError("PRF domain must not be empty")
        encoded_ids: list[bytes] = []
        for token_id in context_ids:
            value = int(token_id)
            if value < 0:
                raise ValueError("token IDs must be non-negative")
            encoded_ids.append(value.to_bytes(8, "big", signed=False))
        payload = domain.encode("utf-8") + b"\x00" + b"".join(encoded_ids)
        digest = hashlib.blake2b(payload, key=self.key, digest_size=16).digest()
        return int.from_bytes(digest[:8], "big", signed=False)
