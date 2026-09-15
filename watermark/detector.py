from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import AbstractSet, Literal

from scipy.stats import binom, norm

from watermark.result_types import CountingResult, DecodeResult, DetectionResult
from watermark.allocator import HashModuloAllocator
from watermark.ecc import BCHCodec
from watermark.partition import ExactPermutationPartitioner, Partition
from watermark.partition_v2 import StatelessExactPartitioner
from watermark.prf import KeyedPRF, partition_round_keys


@dataclass(frozen=True)
class _TokenEvent:
    context: tuple[int, ...]
    token_id: int
    code_bit_index: int
    region: int | None
    upper_hit: bool
    eligible: bool


class DualLayerDetector:
    """Stateless detector that reconstructs every partition from observed token IDs."""

    def __init__(
        self,
        *,
        secret_key: bytes,
        context_width: int,
        vocab_size: int,
        excluded_token_ids: AbstractSet[int] | None,
        prf_mode: str,
        ecc_codec: BCHCodec,
        threshold_mode: str,
        target_fpr: float,
        fixed_z_threshold: float,
        calibrated_threshold: float | None,
        primary_counting_mode: str,
        unique_ngram_width: int,
        min_tokens_per_code_bit: int,
        hard_fill_value: int | str,
        primary_policy: str,
        max_erasure_assignments: int,
        evaluate_all_policies: bool = True,
        presence_test: str = "z_score",
        seeding_scheme: Literal["history", "selfhash"] = "history",
        partition_engine: Literal["v1", "v2"] = "v1",
    ) -> None:
        if context_width <= 0:
            raise ValueError("context_width must be positive")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if prf_mode not in {"paper_shared", "domain_separated"}:
            raise ValueError("prf_mode must be paper_shared or domain_separated")
        if primary_counting_mode not in {"all_tokens", "unique_context", "unique_ngram"}:
            raise ValueError("Unsupported primary_counting_mode")
        if unique_ngram_width <= 0:
            raise ValueError("unique_ngram_width must be positive")
        if min_tokens_per_code_bit <= 0:
            raise ValueError("min_tokens_per_code_bit must be positive")
        if hard_fill_value not in {0, 1, "prf"}:
            raise ValueError("hard_fill_value must be 0, 1, or prf")
        if primary_policy not in {"tie_zero", "strict", "hard_fill", "error_erasure"}:
            raise ValueError(
                "primary_policy must be tie_zero, strict, hard_fill, or error_erasure"
            )
        if presence_test not in {"exact_binomial", "z_score"}:
            raise ValueError("presence_test must be exact_binomial or z_score")
        if seeding_scheme not in {"history", "selfhash"}:
            raise ValueError("seeding_scheme must be history or selfhash")
        if partition_engine not in {"v1", "v2"}:
            raise ValueError("partition_engine must be v1 or v2")
        if seeding_scheme == "selfhash" and partition_engine != "v2":
            raise ValueError("selfhash requires partition_engine='v2'")
        if max_erasure_assignments <= 0:
            raise ValueError("max_erasure_assignments must be positive")
        self.context_width = int(context_width)
        self.vocab_size = int(vocab_size)
        self.excluded_token_ids = set(int(x) for x in (excluded_token_ids or set()))
        self.prf_mode = prf_mode
        self.ecc_codec = ecc_codec
        self.primary_counting_mode = primary_counting_mode
        self.unique_ngram_width = int(unique_ngram_width)
        self.min_tokens_per_code_bit = int(min_tokens_per_code_bit)
        self.hard_fill_value = hard_fill_value
        self.primary_policy = primary_policy
        self.max_erasure_assignments = int(max_erasure_assignments)
        self.evaluate_all_policies = bool(evaluate_all_policies)
        self.presence_test = presence_test
        self.seeding_scheme: Literal["history", "selfhash"] = seeding_scheme
        self.partition_engine: Literal["v1", "v2"] = partition_engine
        self.alpha = float(target_fpr)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("target_fpr must be in (0, 1)")
        self.prf = KeyedPRF(secret_key)
        self.partitioner = (
            ExactPermutationPartitioner()
            if self.partition_engine == "v1"
            else StatelessExactPartitioner(
                self.vocab_size,
                self.excluded_token_ids,
            )
        )
        self.allocator = HashModuloAllocator()
        self.z_threshold = self._resolve_threshold(
            threshold_mode=threshold_mode,
            target_fpr=target_fpr,
            fixed_z_threshold=fixed_z_threshold,
            calibrated_threshold=calibrated_threshold,
        )
        self.threshold = self.alpha if self.presence_test == "exact_binomial" else self.z_threshold
        eligible_vocab_size = self.vocab_size - len(self.excluded_token_ids)
        if eligible_vocab_size < 4:
            raise ValueError("At least four eligible vocabulary tokens are required")
        self.null_probability = 2 * (eligible_vocab_size // 4) / eligible_vocab_size

    @staticmethod
    def _resolve_threshold(
        *,
        threshold_mode: str,
        target_fpr: float,
        fixed_z_threshold: float,
        calibrated_threshold: float | None,
    ) -> float:
        if threshold_mode == "theoretical":
            if not 0.0 < target_fpr < 1.0:
                raise ValueError("target_fpr must be in (0, 1)")
            return float(norm.ppf(1.0 - target_fpr))
        if threshold_mode == "fixed":
            return float(fixed_z_threshold)
        if threshold_mode == "calibrated":
            if calibrated_threshold is None:
                raise ValueError("calibrated threshold mode requires calibrated_threshold")
            return float(calibrated_threshold)
        raise ValueError(f"Unsupported threshold mode: {threshold_mode}")

    def _seeds(self, context: tuple[int, ...]) -> tuple[int, int]:
        if self.prf_mode == "paper_shared":
            shared = self.prf.seed(context, "shared")
            return shared, shared
        return self.prf.seed(context, "partition"), self.prf.seed(context, "position")

    def partition_for_context(self, context: Sequence[int]) -> tuple[Partition, int]:
        context_tuple = tuple(int(token_id) for token_id in context[-self.context_width :])
        partition_seed, position_seed = self._seeds(context_tuple)
        if self.partition_engine == "v1":
            partition = self.partitioner.partition(
                seed=partition_seed,
                vocab_size=self.vocab_size,
                excluded_ids=self.excluded_token_ids,
            )
        else:
            partition = self.partitioner.partition_for_context(
                round_keys=partition_round_keys(partition_seed)
            )
        code_bit_index = self.allocator.allocate(
            seed=position_seed,
            code_length=self.ecc_codec.n,
        )
        return partition, code_bit_index

    def _events_known_boundary(
        self,
        prompt_ids: Sequence[int],
        continuation_ids: Sequence[int],
    ) -> list[_TokenEvent]:
        history = [int(token_id) for token_id in prompt_ids]
        events: list[_TokenEvent] = []
        for raw_token in continuation_ids:
            token_id = int(raw_token)
            allocation_context = tuple(history[-self.context_width :])
            if self.seeding_scheme == "selfhash":
                partition_context = tuple(
                    [*history[-(self.context_width - 1) :], token_id]
                    if self.context_width > 1
                    else [token_id]
                )
            else:
                partition_context = allocation_context
            partition_seed = self._seeds(partition_context)[0]
            position_seed = self._seeds(allocation_context)[1]
            if self.partition_engine == "v1":
                region = self.partitioner.region_for_token(
                    seed=partition_seed,
                    vocab_size=self.vocab_size,
                    excluded_ids=self.excluded_token_ids,
                    token_id=token_id,
                )
            else:
                region = self.partitioner.region_for_token(
                    round_keys=partition_round_keys(partition_seed),
                    token_id=token_id,
                )
            code_bit_index = self.allocator.allocate(
                seed=position_seed,
                code_length=self.ecc_codec.n,
            )
            events.append(
                _TokenEvent(
                    context=partition_context,
                    token_id=token_id,
                    code_bit_index=code_bit_index,
                    region=region,
                    upper_hit=region in {0, 1},
                    eligible=(
                        0 <= token_id < self.vocab_size
                        and token_id not in self.excluded_token_ids
                    ),
                )
            )
            history.append(token_id)
        return events

    def _select_events(self, events: Sequence[_TokenEvent], mode: str) -> list[_TokenEvent]:
        if mode == "all_tokens":
            return list(events)
        selected: list[_TokenEvent] = []
        seen: set[object] = set()
        for event in events:
            if mode == "unique_context":
                key: object = event.context
            elif mode == "unique_ngram":
                key = (event.context[-self.unique_ngram_width :], event.token_id)
            else:  # constructor validation makes this unreachable
                raise ValueError(f"Unsupported counting mode: {mode}")
            if key in seen:
                continue
            seen.add(key)
            selected.append(event)
        return selected

    def _count(self, events: Sequence[_TokenEvent], mode: str) -> CountingResult:
        selected = self._select_events([event for event in events if event.eligible], mode)
        n0 = [0] * self.ecc_codec.n
        n1 = [0] * self.ecc_codec.n
        upper_hits = 0
        for event in selected:
            if event.upper_hit:
                upper_hits += 1
            if event.region == 0:
                n0[event.code_bit_index] += 1
            elif event.region == 1:
                n1[event.code_bit_index] += 1
        scored_tokens = len(selected)
        if scored_tokens == 0:
            z_score = 0.0
            exact_p_value = 1.0
        else:
            mean = scored_tokens * self.null_probability
            variance = scored_tokens * self.null_probability * (1.0 - self.null_probability)
            z_score = (upper_hits - mean) / math.sqrt(variance)
            exact_p_value = float(
                binom.sf(upper_hits - 1, scored_tokens, self.null_probability)
            )
        erasures = tuple(
            index
            for index, (zero_votes, one_votes) in enumerate(zip(n0, n1, strict=True))
            if zero_votes + one_votes < self.min_tokens_per_code_bit
            or zero_votes == one_votes
        )
        return CountingResult(
            mode=mode,
            scored_tokens=scored_tokens,
            upper_hits=upper_hits,
            z_score=float(z_score),
            exact_p_value=exact_p_value,
            n0=tuple(n0),
            n1=tuple(n1),
            erasures=erasures,
            null_probability=self.null_probability,
        )

    def _hard_decisions(self, counting: CountingResult) -> list[int | None]:
        decisions: list[int | None] = []
        erasure_set = set(counting.erasures)
        for index, (zero_votes, one_votes) in enumerate(zip(counting.n0, counting.n1, strict=True)):
            if index in erasure_set:
                decisions.append(None)
            else:
                decisions.append(1 if one_votes > zero_votes else 0)
        return decisions

    def recover_codeword(
        self,
        counting: CountingResult,
        *,
        policy: str,
    ) -> tuple[int, ...] | None:
        if policy not in {"tie_zero", "strict", "hard_fill"}:
            raise ValueError("policy must be tie_zero, strict, or hard_fill")
        if policy == "tie_zero":
            return tuple(
                1 if one_votes > zero_votes else 0
                for zero_votes, one_votes in zip(counting.n0, counting.n1, strict=True)
            )
        decisions = self._hard_decisions(counting)
        if policy == "strict" and any(bit is None for bit in decisions):
            return None
        return tuple(
            self._fill_bit(index) if bit is None else int(bit)
            for index, bit in enumerate(decisions)
        )

    def _strict_decode(self, counting: CountingResult) -> DecodeResult:
        codeword = self.recover_codeword(counting, policy="strict")
        if codeword is None:
            decisions = self._hard_decisions(counting)
            return DecodeResult(
                status="insufficient_evidence",
                message_bits=None,
                corrected_errors=None,
                error=f"{sum(bit is None for bit in decisions)} ECC bits are erased",
            )
        return self.ecc_codec.decode(codeword)

    def _fill_bit(self, index: int) -> int:
        if self.hard_fill_value in {0, 1}:
            return int(self.hard_fill_value)
        return self.prf.seed([index], "hard-fill") % 2

    def _hard_fill_decode(self, counting: CountingResult) -> DecodeResult:
        codeword = self.recover_codeword(counting, policy="hard_fill")
        assert codeword is not None
        return self.ecc_codec.decode(codeword)

    def _tie_zero_decode(self, counting: CountingResult) -> DecodeResult:
        codeword = self.recover_codeword(counting, policy="tie_zero")
        assert codeword is not None
        return self.ecc_codec.decode(codeword)

    def _error_erasure_decode(self, counting: CountingResult) -> DecodeResult:
        return self.ecc_codec.decode_errors_and_erasures(
            self._hard_decisions(counting),
            max_assignments=self.max_erasure_assignments,
        )

    def _selected_decode(
        self,
        tie_zero_decode: DecodeResult,
        strict_decode: DecodeResult,
        hard_fill_decode: DecodeResult,
        error_erasure_decode: DecodeResult,
    ) -> DecodeResult:
        if self.primary_policy == "tie_zero":
            return tie_zero_decode
        if self.primary_policy == "strict":
            return strict_decode
        if self.primary_policy == "hard_fill":
            return hard_fill_decode
        return error_erasure_decode

    def _not_evaluated_decode(self, policy: str) -> DecodeResult:
        return DecodeResult(
            status="not_evaluated",
            message_bits=None,
            corrected_errors=None,
            error=f"{policy} decoding was not evaluated",
        )

    def _detect_events(
        self,
        events: Sequence[_TokenEvent],
        input_mode: str,
        *,
        decode_payload: bool = True,
    ) -> DetectionResult:
        counting = {
            mode: self._count(events, mode)
            for mode in ("all_tokens", "unique_context", "unique_ngram")
        }
        primary = counting[self.primary_counting_mode]
        detected = (
            primary.exact_p_value <= self.alpha
            if self.presence_test == "exact_binomial"
            else primary.z_score >= self.z_threshold
        )
        decoder_invoked = bool(decode_payload and detected)
        if decoder_invoked:
            tie_zero_decode = (
                self._tie_zero_decode(primary)
                if self.evaluate_all_policies or self.primary_policy == "tie_zero"
                else self._not_evaluated_decode("tie_zero")
            )
            strict_decode = (
                self._strict_decode(primary)
                if self.evaluate_all_policies or self.primary_policy == "strict"
                else self._not_evaluated_decode("strict")
            )
            hard_fill_decode = (
                self._hard_fill_decode(primary)
                if self.evaluate_all_policies or self.primary_policy == "hard_fill"
                else self._not_evaluated_decode("hard_fill")
            )
            error_erasure_decode = (
                self._error_erasure_decode(primary)
                if self.evaluate_all_policies or self.primary_policy == "error_erasure"
                else self._not_evaluated_decode("error_erasure")
            )
        else:
            tie_zero_decode = self._not_evaluated_decode("tie_zero")
            strict_decode = self._not_evaluated_decode("strict")
            hard_fill_decode = self._not_evaluated_decode("hard_fill")
            error_erasure_decode = self._not_evaluated_decode("error_erasure")
        selected_decode = self._selected_decode(
            tie_zero_decode,
            strict_decode,
            hard_fill_decode,
            error_erasure_decode,
        )
        gated_message = (
            selected_decode.message_bits
            if detected and selected_decode.status == "decoded"
            else None
        )
        return DetectionResult(
            input_mode=input_mode,
            threshold=self.threshold,
            detected=detected,
            primary_counting_mode=self.primary_counting_mode,
            counting=counting,
            strict_decode=strict_decode,
            hard_fill_decode=hard_fill_decode,
            error_erasure_decode=error_erasure_decode,
            primary_policy=self.primary_policy,
            gated_message_bits=gated_message,
            tie_zero_decode=tie_zero_decode,
            presence_test=self.presence_test,
            alpha=self.alpha,
            decoder_invoked=decoder_invoked,
        )

    def detect_continuation(
        self,
        prompt_ids: Sequence[int],
        continuation_ids: Sequence[int],
        *,
        decode_payload: bool = True,
    ) -> DetectionResult:
        events = self._events_known_boundary(prompt_ids, continuation_ids)
        return self._detect_events(events, "known_boundary", decode_payload=decode_payload)

    def detect_with_boundary(
        self,
        full_ids: Sequence[int],
        prompt_length: int,
        *,
        decode_payload: bool = True,
    ) -> DetectionResult:
        if prompt_length < 0 or prompt_length > len(full_ids):
            raise ValueError("prompt_length must be within full_ids")
        return self.detect_continuation(
            full_ids[:prompt_length],
            full_ids[prompt_length:],
            decode_payload=decode_payload,
        )

    def detect_token_ids(
        self,
        token_ids: Sequence[int],
        *,
        decode_payload: bool = True,
    ) -> DetectionResult:
        values = [int(token_id) for token_id in token_ids]
        if len(values) <= self.context_width:
            return self._detect_events([], "blind_text", decode_payload=decode_payload)
        prompt = values[: self.context_width]
        continuation = values[self.context_width :]
        events = self._events_known_boundary(prompt, continuation)
        return self._detect_events(events, "blind_text", decode_payload=decode_payload)
