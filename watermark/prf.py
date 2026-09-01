from __future__ import annotations

import hashlib
from collections.abc import Sequence


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
