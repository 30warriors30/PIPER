from __future__ import annotations

import hashlib
import random


def derive_seed(base_seed: int, namespace: str, sample_id: str) -> int:
    payload = f"{int(base_seed)}\x00{namespace}\x00{sample_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def message_bits(message_seed: int, sample_id: str, length: int) -> tuple[int, ...]:
    if length <= 0:
        raise ValueError("message length must be positive")
    seed = derive_seed(message_seed, "payload", sample_id)
    value = random.Random(seed).getrandbits(length)
    return tuple((value >> shift) & 1 for shift in range(length - 1, -1, -1))
