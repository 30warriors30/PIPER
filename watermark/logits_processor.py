from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import AbstractSet, Literal

import torch

try:  # Keep unit tests runnable before the optional HF dependency is installed.
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover - exercised only in restricted test environments
    class LogitsProcessor:  # type: ignore[no-redef]
        pass

from watermark.allocator import HashModuloAllocator
from watermark.partition import ExactPermutationPartitioner
from watermark.partition_v2 import StatelessExactPartitioner
from watermark.prf import KeyedPRF, partition_round_keys


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
        encoded_bits: Sequence[int] | None = None,
        vocab_size: int,
        excluded_token_ids: AbstractSet[int] | None,
        presence_mode: str,
        delta_presence: float,
        delta_payload: float,
        prf_mode: str,
        encoded_bits_by_row: Sequence[Sequence[int]] | None = None,
        partition_engine: Literal["v1", "v2"] = "v1",
        capture_traces: bool = True,
    ) -> None:
        if context_width <= 0:
            raise ValueError("context_width must be positive")
        if presence_mode not in {"hard", "soft"}:
            raise ValueError("presence_mode must be hard or soft")
        if prf_mode not in {"paper_shared", "domain_separated"}:
            raise ValueError("prf_mode must be paper_shared or domain_separated")
        if partition_engine not in {"v1", "v2"}:
            raise ValueError("partition_engine must be v1 or v2")
        self.context_width = int(context_width)
        self.encoded_bits_by_row = self._normalize_codewords(
            encoded_bits=encoded_bits,
            encoded_bits_by_row=encoded_bits_by_row,
        )
        self.encoded_bits = (
            self.encoded_bits_by_row[0]
            if len(self.encoded_bits_by_row) == 1
            else None
        )
        self.vocab_size = int(vocab_size)
        self.excluded_token_ids = set(
            int(token_id) for token_id in (excluded_token_ids or set())
        )
        self.presence_mode = presence_mode
        self.delta_presence = float(delta_presence)
        self.delta_payload = float(delta_payload)
        self.prf_mode = prf_mode
        self.partition_engine: Literal["v1", "v2"] = partition_engine
        self.capture_traces = bool(capture_traces)
        self.prf = KeyedPRF(secret_key)
        if self.partition_engine == "v1":
            self.partitioner = ExactPermutationPartitioner()
        else:
            self.partitioner = StatelessExactPartitioner(
                self.vocab_size,
                self.excluded_token_ids,
            )
        self.allocator = HashModuloAllocator()
        self.last_traces: tuple[EmbeddingTrace, ...] = ()

    @staticmethod
    def _normalize_codewords(
        *,
        encoded_bits: Sequence[int] | None,
        encoded_bits_by_row: Sequence[Sequence[int]] | None,
    ) -> tuple[tuple[int, ...], ...]:
        if (encoded_bits is None) == (encoded_bits_by_row is None):
            raise ValueError(
                "exactly one of encoded_bits and encoded_bits_by_row is required"
            )
        if encoded_bits is not None:
            codeword = tuple(encoded_bits)
            if not codeword or any(bit not in (0, 1) for bit in codeword):
                raise ValueError("encoded_bits must be a non-empty binary sequence")
            return (tuple(int(bit) for bit in codeword),)

        assert encoded_bits_by_row is not None
        codewords = tuple(tuple(codeword) for codeword in encoded_bits_by_row)
        if not codewords:
            raise ValueError("at least one encoded codeword is required")
        if any(not codeword for codeword in codewords):
            raise ValueError("encoded codewords must be non-empty")
        code_length = len(codewords[0])
        if any(len(codeword) != code_length for codeword in codewords[1:]):
            raise ValueError("encoded codewords must have equal length")
        if any(bit not in (0, 1) for codeword in codewords for bit in codeword):
            raise ValueError("encoded codewords must be binary")
        return tuple(tuple(int(bit) for bit in codeword) for codeword in codewords)

    @property
    def last_trace(self) -> EmbeddingTrace | None:
        if len(self.last_traces) == 1:
            return self.last_traces[0]
        return None

    def _seeds(self, context: tuple[int, ...]) -> tuple[int, int]:
        if self.prf_mode == "paper_shared":
            shared = self.prf.seed(context, "shared")
            return shared, shared
        return self.prf.seed(context, "partition"), self.prf.seed(context, "position")

    @staticmethod
    def _row_label(count: int) -> str:
        return "row" if count == 1 else "rows"

    def _validate_call_shapes(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> None:
        if input_ids.ndim != 2 or scores.ndim != 2:
            raise ValueError(
                "input_ids and scores must have shape [batch, sequence/vocabulary]"
            )
        input_rows = int(input_ids.shape[0])
        score_rows = int(scores.shape[0])
        if input_rows != score_rows:
            raise ValueError(
                f"input_ids has {input_rows} {self._row_label(input_rows)} but scores "
                f"has {score_rows} {self._row_label(score_rows)}"
            )
        codeword_rows = len(self.encoded_bits_by_row)
        if score_rows != codeword_rows:
            codeword_label = (
                "encoded codeword" if codeword_rows == 1 else "encoded codewords"
            )
            raise ValueError(
                f"{score_rows} score {self._row_label(score_rows)} but "
                f"{codeword_rows} {codeword_label}"
            )
        if scores.shape[1] != self.vocab_size:
            raise ValueError(
                f"scores vocabulary size {scores.shape[1]} does not match configured "
                f"{self.vocab_size}"
            )

    def _call_v1(
        self,
        contexts: tuple[tuple[int, ...], ...],
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        output = scores.clone()
        traces: list[EmbeddingTrace] = []
        for row, (context, codeword) in enumerate(
            zip(contexts, self.encoded_bits_by_row)
        ):
            partition_seed, position_seed = self._seeds(context)
            partition = self.partitioner.partition(
                seed=partition_seed,
                vocab_size=self.vocab_size,
                excluded_ids=self.excluded_token_ids,
            )
            code_bit_index = self.allocator.allocate(
                seed=position_seed,
                code_length=len(codeword),
            )
            embedded_bit = codeword[code_bit_index]
            target_ids = partition.bit1 if embedded_bit == 1 else partition.bit0
            non_target_upper_ids = (
                partition.bit0 if embedded_bit == 1 else partition.bit1
            )
            lower_ids = partition.lower

            if self.presence_mode == "hard":
                blocked_ids = tuple(lower_ids) + tuple(sorted(self.excluded_token_ids))
                output[row, list(blocked_ids)] = -torch.inf
                output[row, list(target_ids)] += self.delta_payload
            else:
                output[row, list(partition.upper)] += self.delta_presence
                output[row, list(target_ids)] += self.delta_payload

            if self.capture_traces:
                traces.append(
                    EmbeddingTrace(
                        context_ids=context,
                        partition_seed=partition_seed,
                        position_seed=position_seed,
                        code_bit_index=code_bit_index,
                        embedded_bit=embedded_bit,
                        target_ids=target_ids,
                        non_target_upper_ids=non_target_upper_ids,
                        lower_ids=lower_ids,
                    )
                )
        self.last_traces = tuple(traces)
        return output

    def _call_v2(
        self,
        contexts: tuple[tuple[int, ...], ...],
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        seeds_by_row = tuple(self._seeds(context) for context in contexts)
        partition_seeds = tuple(seeds[0] for seeds in seeds_by_row)
        position_seeds = tuple(seeds[1] for seeds in seeds_by_row)
        round_keys_by_row = tuple(
            partition_round_keys(partition_seed)
            for partition_seed in partition_seeds
        )
        regions = self.partitioner.regions_for_vocab(
            round_keys_by_row=round_keys_by_row,
            device=scores.device,
        )
        code_bit_indices = tuple(
            self.allocator.allocate(seed=position_seed, code_length=len(codeword))
            for position_seed, codeword in zip(
                position_seeds,
                self.encoded_bits_by_row,
            )
        )
        embedded_bit_values = tuple(
            codeword[code_bit_index]
            for codeword, code_bit_index in zip(
                self.encoded_bits_by_row,
                code_bit_indices,
            )
        )
        embedded_bits = torch.tensor(
            embedded_bit_values,
            dtype=torch.int64,
            device=scores.device,
        )

        output = scores.clone()
        upper_mask = (regions == 0) | (regions == 1)
        target_mask = regions == embedded_bits[:, None]
        if self.presence_mode == "hard":
            output.masked_fill_(~upper_mask, -torch.inf)
        else:
            output.add_(upper_mask.to(output.dtype) * self.delta_presence)
        output.add_(target_mask.to(output.dtype) * self.delta_payload)

        if self.capture_traces:
            trace_regions = regions.detach().cpu()
            traces: list[EmbeddingTrace] = []
            for row, context in enumerate(contexts):
                row_regions = trace_regions[row]
                embedded_bit = embedded_bit_values[row]
                target_ids = tuple(
                    int(token_id)
                    for token_id in torch.nonzero(
                        row_regions == embedded_bit,
                        as_tuple=False,
                    ).flatten().tolist()
                )
                non_target_upper_ids = tuple(
                    int(token_id)
                    for token_id in torch.nonzero(
                        row_regions == 1 - embedded_bit,
                        as_tuple=False,
                    ).flatten().tolist()
                )
                lower_ids = tuple(
                    int(token_id)
                    for token_id in torch.nonzero(
                        (row_regions == 2) | (row_regions == 3),
                        as_tuple=False,
                    ).flatten().tolist()
                )
                traces.append(
                    EmbeddingTrace(
                        context_ids=context,
                        partition_seed=partition_seeds[row],
                        position_seed=position_seeds[row],
                        code_bit_index=code_bit_indices[row],
                        embedded_bit=embedded_bit,
                        target_ids=target_ids,
                        non_target_upper_ids=non_target_upper_ids,
                        lower_ids=lower_ids,
                    )
                )
            self.last_traces = tuple(traces)
        return output

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        self._validate_call_shapes(input_ids, scores)
        context_values = input_ids[:, -self.context_width :].detach().cpu().tolist()
        contexts = tuple(
            tuple(int(value) for value in row) for row in context_values
        )
        self.last_traces = ()
        if self.partition_engine == "v1":
            return self._call_v1(contexts, scores)
        return self._call_v2(contexts, scores)
