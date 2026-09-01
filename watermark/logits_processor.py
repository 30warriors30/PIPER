from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet

import torch

try:  # Keep unit tests runnable before the optional HF dependency is installed.
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover - exercised only in restricted test environments
    class LogitsProcessor:  # type: ignore[no-redef]
        pass

from watermark.allocator import HashModuloAllocator
from watermark.partition import ExactPermutationPartitioner
from watermark.prf import KeyedPRF


@dataclass(frozen=True)
class EmbeddingTrace:
    context_ids: tuple[int, ...]
    partition_seed: int
    position_seed: int
    code_bit_index: int
    embedded_bit: int
    target_ids: tuple[int, ...]
    non_target_upper_ids: tuple[int, ...]
    lower_ids: tuple[int, ...]


class DualLayerLogitsProcessor(LogitsProcessor):
    """Apply the paper's presence layer and payload-subpartition bias."""

    def __init__(
        self,
        *,
        secret_key: bytes,
        context_width: int,
        encoded_bits: tuple[int, ...],
        vocab_size: int,
        excluded_token_ids: AbstractSet[int] | None,
        presence_mode: str,
        delta_presence: float,
        delta_payload: float,
        prf_mode: str,
    ) -> None:
        if context_width <= 0:
            raise ValueError("context_width must be positive")
        if not encoded_bits or any(bit not in (0, 1) for bit in encoded_bits):
            raise ValueError("encoded_bits must be a non-empty binary sequence")
        if presence_mode not in {"hard", "soft"}:
            raise ValueError("presence_mode must be hard or soft")
        if prf_mode not in {"paper_shared", "domain_separated"}:
            raise ValueError("prf_mode must be paper_shared or domain_separated")
        self.context_width = int(context_width)
        self.encoded_bits = tuple(int(bit) for bit in encoded_bits)
        self.vocab_size = int(vocab_size)
        self.excluded_token_ids = set(int(token_id) for token_id in (excluded_token_ids or set()))
        self.presence_mode = presence_mode
        self.delta_presence = float(delta_presence)
        self.delta_payload = float(delta_payload)
        self.prf_mode = prf_mode
        self.prf = KeyedPRF(secret_key)
        self.partitioner = ExactPermutationPartitioner()
        self.allocator = HashModuloAllocator()
        self.last_trace: EmbeddingTrace | None = None

    def _seeds(self, context: tuple[int, ...]) -> tuple[int, int]:
        if self.prf_mode == "paper_shared":
            shared = self.prf.seed(context, "shared")
            return shared, shared
        return self.prf.seed(context, "partition"), self.prf.seed(context, "position")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.ndim != 2 or scores.ndim != 2:
            raise ValueError("input_ids and scores must have shape [batch, sequence/vocabulary]")
        if input_ids.shape[0] != 1 or scores.shape[0] != 1:
            raise ValueError(
                "MVP generation supports batch_size=1 because payload state is sample-specific"
            )
        if scores.shape[1] != self.vocab_size:
            raise ValueError(
                f"scores vocabulary size {scores.shape[1]} does not match configured "
                f"{self.vocab_size}"
            )
        context = tuple(int(value) for value in input_ids[0, -self.context_width :].tolist())
        partition_seed, position_seed = self._seeds(context)
        partition = self.partitioner.partition(
            seed=partition_seed,
            vocab_size=self.vocab_size,
            excluded_ids=self.excluded_token_ids,
        )
        code_bit_index = self.allocator.allocate(
            seed=position_seed,
            code_length=len(self.encoded_bits),
        )
        embedded_bit = self.encoded_bits[code_bit_index]
        target_ids = partition.bit1 if embedded_bit == 1 else partition.bit0
        non_target_upper_ids = partition.bit0 if embedded_bit == 1 else partition.bit1
        lower_ids = partition.lower

        output = scores.clone()
        if self.presence_mode == "hard":
            blocked_ids = tuple(lower_ids) + tuple(sorted(self.excluded_token_ids))
            output[0, list(blocked_ids)] = -torch.inf
            output[0, list(target_ids)] += self.delta_payload
        else:
            output[0, list(partition.upper)] += self.delta_presence
            output[0, list(target_ids)] += self.delta_payload

        self.last_trace = EmbeddingTrace(
            context_ids=context,
            partition_seed=partition_seed,
            position_seed=position_seed,
            code_bit_index=code_bit_index,
            embedded_bit=embedded_bit,
            target_ids=target_ids,
            non_target_upper_ids=non_target_upper_ids,
            lower_ids=lower_ids,
        )
        return output
