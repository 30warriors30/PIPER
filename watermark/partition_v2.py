from __future__ import annotations

from collections.abc import Sequence
from typing import AbstractSet

import torch

from watermark.partition import Partition
from watermark.prf import MASK64, _avalanche64, partition_round_keys

_AVALANCHE_MULTIPLIER_1_SIGNED = -4658895280553007687
_AVALANCHE_MULTIPLIER_2_SIGNED = -7723592293110705685


def _logical_right_shift_tensor(values: torch.Tensor, shift: int) -> torch.Tensor:
    mask = (1 << (64 - shift)) - 1
    return torch.bitwise_right_shift(values, shift) & mask


def _avalanche64_tensor(values: torch.Tensor) -> torch.Tensor:
    values = (values ^ _logical_right_shift_tensor(values, 30)) * (
        _AVALANCHE_MULTIPLIER_1_SIGNED
    )
    values = (values ^ _logical_right_shift_tensor(values, 27)) * (
        _AVALANCHE_MULTIPLIER_2_SIGNED
    )
    return values ^ _logical_right_shift_tensor(values, 31)


class StatelessExactPartitioner:
    """Map eligible token IDs through an exact stateless four-way partition."""

    _mapping_cache: dict[
        tuple[int, tuple[int, ...]],
        tuple[tuple[int, ...], tuple[int, ...]],
    ] = {}

    def __init__(
        self,
        vocab_size: int,
        excluded_ids: AbstractSet[int] | None,
    ) -> None:
        self.vocab_size = int(vocab_size)
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        excluded_key = tuple(sorted(int(token_id) for token_id in (excluded_ids or set())))
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in excluded_key):
            raise ValueError("excluded token ID is outside the vocabulary")

        eligible_count = self.vocab_size - len(excluded_key)
        if eligible_count < 4:
            raise ValueError("At least four eligible vocabulary tokens are required")

        cache_key = (self.vocab_size, excluded_key)
        cached = self._mapping_cache.get(cache_key)
        if cached is None:
            excluded = set(excluded_key)
            dense_to_token = tuple(
                token_id
                for token_id in range(self.vocab_size)
                if token_id not in excluded
            )
            token_to_dense_values = [-1] * self.vocab_size
            for dense_index, token_id in enumerate(dense_to_token):
                token_to_dense_values[token_id] = dense_index
            cached = (tuple(token_to_dense_values), dense_to_token)
            self._mapping_cache[cache_key] = cached

        self._token_to_dense, self._dense_to_token = cached
        self._eligible_count = eligible_count
        domain_bits = max(2, (eligible_count - 1).bit_length())
        self._domain_bits = domain_bits + domain_bits % 2
        self._half_bits = self._domain_bits // 2
        self._half_mask = (1 << self._half_bits) - 1
        self._eligible_ids_by_device: dict[torch.device, torch.Tensor] = {}
        self._token_to_dense_by_device: dict[torch.device, torch.Tensor] = {}

    @staticmethod
    def _normalize_round_keys(round_keys: Sequence[int]) -> tuple[int, ...]:
        if len(round_keys) != 6:
            raise ValueError("exactly six round keys are required")
        return tuple(int(round_key) & MASK64 for round_key in round_keys)

    def _permute(self, value: int, round_keys: tuple[int, ...]) -> int:
        left = int(value) >> self._half_bits
        right = int(value) & self._half_mask
        for round_key in round_keys:
            round_output = _avalanche64((right ^ round_key) & MASK64) & self._half_mask
            left, right = right, (left ^ round_output) & self._half_mask
        return (left << self._half_bits) | right

    def _rank_for_dense(self, dense_index: int, round_keys: tuple[int, ...]) -> int:
        rank = self._permute(dense_index, round_keys)
        while rank >= self._eligible_count:
            rank = self._permute(rank, round_keys)
        return rank

    @staticmethod
    def _as_signed_int64(value: int) -> int:
        normalized = int(value) & MASK64
        if normalized >= 1 << 63:
            return normalized - (1 << 64)
        return normalized

    def _permute_tensor(
        self,
        values: torch.Tensor,
        round_keys: torch.Tensor,
    ) -> torch.Tensor:
        left = torch.bitwise_right_shift(values, self._half_bits)
        right = values & self._half_mask
        for round_index in range(6):
            round_input = right ^ round_keys[:, round_index].unsqueeze(1)
            round_output = _avalanche64_tensor(round_input) & self._half_mask
            left, right = right & self._half_mask, (left ^ round_output) & self._half_mask
        return (left << self._half_bits) | right

    def _eligible_ids_for_device(self, device: torch.device) -> torch.Tensor:
        normalized_device = torch.device(device)
        cached = self._eligible_ids_by_device.get(normalized_device)
        if cached is None:
            cached = torch.tensor(
                self._dense_to_token,
                dtype=torch.int64,
                device=normalized_device,
            )
            self._eligible_ids_by_device[normalized_device] = cached
        return cached

    def _token_to_dense_for_device(self, device: torch.device) -> torch.Tensor:
        normalized_device = torch.device(device)
        cached = self._token_to_dense_by_device.get(normalized_device)
        if cached is None:
            cached = torch.tensor(
                self._token_to_dense,
                dtype=torch.int64,
                device=normalized_device,
            )
            self._token_to_dense_by_device[normalized_device] = cached
        return cached

    def rank_for_token(
        self,
        *,
        round_keys: Sequence[int],
        token_id: int,
    ) -> int | None:
        keys = self._normalize_round_keys(round_keys)
        token = int(token_id)
        if token < 0 or token >= self.vocab_size:
            return None
        dense_index = self._token_to_dense[token]
        if dense_index < 0:
            return None
        return self._rank_for_dense(dense_index, keys)

    def region_for_token(
        self,
        *,
        round_keys: Sequence[int],
        token_id: int,
    ) -> int | None:
        rank = self.rank_for_token(round_keys=round_keys, token_id=token_id)
        if rank is None:
            return None
        return self._region_for_rank(rank)

    def regions_for_vocab(
        self,
        *,
        round_keys_by_row: Sequence[Sequence[int]],
        device: torch.device,
    ) -> torch.Tensor:
        normalized_keys = tuple(
            self._normalize_round_keys(round_keys) for round_keys in round_keys_by_row
        )
        normalized_device = torch.device(device)
        batch_size = len(normalized_keys)
        eligible_ids = self._eligible_ids_for_device(normalized_device)
        dense_indices = torch.arange(
            self._eligible_count,
            dtype=torch.int64,
            device=normalized_device,
        ).unsqueeze(0).expand(batch_size, -1)
        tensor_keys = torch.tensor(
            [
                [self._as_signed_int64(round_key) for round_key in row_keys]
                for row_keys in normalized_keys
            ],
            dtype=torch.int64,
            device=normalized_device,
        ).reshape(batch_size, 6)

        ranks = self._permute_tensor(dense_indices, tensor_keys)
        outside_domain = ranks >= self._eligible_count
        while bool(torch.any(outside_domain)):
            walked_ranks = self._permute_tensor(ranks, tensor_keys)
            ranks = torch.where(outside_domain, walked_ranks, ranks)
            outside_domain = ranks >= self._eligible_count

        quarter = self._eligible_count // 4
        lower_a_end = 2 * quarter + (self._eligible_count - 2 * quarter) // 2
        eligible_regions = torch.where(ranks < lower_a_end, 2, 3)
        eligible_regions = torch.where(ranks < 2 * quarter, 1, eligible_regions)
        eligible_regions = torch.where(ranks < quarter, 0, eligible_regions)

        regions = torch.full(
            (batch_size, self.vocab_size),
            -1,
            dtype=torch.int64,
            device=normalized_device,
        )
        regions[:, eligible_ids] = eligible_regions
        return regions

    def regions_for_tokens(
        self,
        *,
        round_keys_by_token: Sequence[Sequence[Sequence[int]]],
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, candidates]")
        batch_size, candidate_count = map(int, token_ids.shape)
        normalized_keys = tuple(
            tuple(self._normalize_round_keys(round_keys) for round_keys in row)
            for row in round_keys_by_token
        )
        if len(normalized_keys) != batch_size or any(
            len(row) != candidate_count for row in normalized_keys
        ):
            raise ValueError(
                "round_keys_by_token must match token_ids batch and candidate dimensions"
            )

        device = token_ids.device
        flat_tokens = token_ids.to(dtype=torch.int64).reshape(-1)
        in_range = (flat_tokens >= 0) & (flat_tokens < self.vocab_size)
        safe_tokens = flat_tokens.clamp(min=0, max=self.vocab_size - 1)
        dense_mapping = self._token_to_dense_for_device(device)
        dense_indices = dense_mapping[safe_tokens]
        eligible = in_range & (dense_indices >= 0)
        permutation_inputs = torch.where(
            eligible,
            dense_indices,
            torch.zeros_like(dense_indices),
        ).unsqueeze(1)
        tensor_keys = torch.tensor(
            [
                [self._as_signed_int64(round_key) for round_key in keys]
                for row in normalized_keys
                for keys in row
            ],
            dtype=torch.int64,
            device=device,
        ).reshape(batch_size * candidate_count, 6)

        ranks = self._permute_tensor(permutation_inputs, tensor_keys)
        outside_domain = ranks >= self._eligible_count
        while bool(torch.any(outside_domain)):
            walked_ranks = self._permute_tensor(ranks, tensor_keys)
            ranks = torch.where(outside_domain, walked_ranks, ranks)
            outside_domain = ranks >= self._eligible_count
        ranks = ranks.squeeze(1)

        quarter = self._eligible_count // 4
        lower_a_end = 2 * quarter + (self._eligible_count - 2 * quarter) // 2
        regions = torch.where(ranks < lower_a_end, 2, 3)
        regions = torch.where(ranks < 2 * quarter, 1, regions)
        regions = torch.where(ranks < quarter, 0, regions)
        regions = torch.where(eligible, regions, -1)
        return regions.reshape(batch_size, candidate_count)

    def _region_for_rank(self, rank: int) -> int:
        quarter = self._eligible_count // 4
        if rank < quarter:
            return 0
        if rank < 2 * quarter:
            return 1
        lower_a_end = 2 * quarter + (self._eligible_count - 2 * quarter) // 2
        if rank < lower_a_end:
            return 2
        return 3

    def partition_for_context(self, *, round_keys: Sequence[int]) -> Partition:
        keys = self._normalize_round_keys(round_keys)
        ranked_tokens = [-1] * self._eligible_count
        for dense_index, token_id in enumerate(self._dense_to_token):
            rank = self._rank_for_dense(dense_index, keys)
            ranked_tokens[rank] = token_id

        quarter = self._eligible_count // 4
        lower_a_end = 2 * quarter + (self._eligible_count - 2 * quarter) // 2
        return Partition(
            bit0=tuple(ranked_tokens[:quarter]),
            bit1=tuple(ranked_tokens[quarter : 2 * quarter]),
            lower_a=tuple(ranked_tokens[2 * quarter : lower_a_end]),
            lower_b=tuple(ranked_tokens[lower_a_end:]),
        )
