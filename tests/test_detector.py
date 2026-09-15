import torch

from watermark.detector import DualLayerDetector
from watermark.ecc import BCHCodec
from watermark.logits_processor import DualLayerLogitsProcessor
from watermark.result_types import CountingResult, DecodeResult
from tests.helpers import synthetic_upper_sequence


MESSAGE = (1, 0, 1, 1, 0, 0, 1, 0)


def make_detector(primary_policy: str = "error_erasure", evaluate_all_policies: bool = True) -> DualLayerDetector:
    return DualLayerDetector(
        secret_key=b"secret",
        context_width=4,
        vocab_size=32,
        excluded_token_ids={0},
        prf_mode="paper_shared",
        ecc_codec=BCHCodec(23, 8, 3),
        threshold_mode="fixed",
        target_fpr=0.01,
        fixed_z_threshold=-100.0,
        calibrated_threshold=None,
        primary_counting_mode="all_tokens",
        unique_ngram_width=4,
        min_tokens_per_code_bit=1,
        hard_fill_value=0,
        primary_policy=primary_policy,
        max_erasure_assignments=64,
        evaluate_all_policies=evaluate_all_policies,
    )


def test_prompt_is_context_not_evidence() -> None:
    detector = make_detector()
    prompt = [9, 8, 7, 6]
    continuation = synthetic_upper_sequence(detector, prompt, 16)
    result = detector.detect_continuation(prompt, continuation)
    assert result.counting["all_tokens"].scored_tokens == 16


def test_blind_mode_skips_context_width() -> None:
    detector = make_detector()
    prefix = [9, 8, 7, 6]
    result = detector.detect_token_ids(prefix + synthetic_upper_sequence(detector, prefix, 12))
    assert result.input_mode == "blind_text"
    assert result.counting["all_tokens"].scored_tokens == 12


def test_error_erasure_policy_decodes_tied_positions() -> None:
    detector = make_detector("error_erasure")
    codeword = detector.ecc_codec.encode(MESSAGE)
    n0 = [0] * detector.ecc_codec.n
    n1 = [0] * detector.ecc_codec.n
    for index, bit in enumerate(codeword):
        if bit:
            n1[index] = 3
        else:
            n0[index] = 3
    n0[5] = 1
    n1[5] = 1
    counting = CountingResult(
        mode="all_tokens",
        scored_tokens=sum(n0) + sum(n1),
        upper_hits=sum(n0) + sum(n1),
        z_score=10.0,
        exact_p_value=0.0,
        n0=tuple(n0),
        n1=tuple(n1),
        erasures=(5,),
    )

    decoded = detector._error_erasure_decode(counting)

    assert decoded.status == "decoded"
    assert decoded.message_bits == MESSAGE
    assert decoded.erasure_count == 1


def test_detector_scores_observed_token_regions_without_materializing_partition(monkeypatch) -> None:
    detector = make_detector()

    def region_for_token(**_kwargs):
        return 1

    def fail_partition(**_kwargs):
        raise AssertionError("full partition should not be materialized while detecting")

    monkeypatch.setattr(detector.partitioner, "region_for_token", region_for_token, raising=False)
    monkeypatch.setattr(detector.partitioner, "partition", fail_partition)

    result = detector.detect_continuation([9, 8, 7, 6], [1, 2])

    assert result.counting["all_tokens"].scored_tokens == 2
    assert result.counting["all_tokens"].upper_hits == 2


def test_detector_can_skip_non_primary_decoding(monkeypatch) -> None:
    detector = make_detector(primary_policy="error_erasure", evaluate_all_policies=False)

    def fail_decode(_counting):
        raise AssertionError("non-primary decode policy should not run")

    monkeypatch.setattr(detector, "_strict_decode", fail_decode)
    monkeypatch.setattr(detector, "_hard_fill_decode", fail_decode)
    monkeypatch.setattr(
        detector,
        "_error_erasure_decode",
        lambda _counting: DecodeResult(
            status="decoded",
            message_bits=MESSAGE,
            corrected_errors=0,
            codeword_bits=detector.ecc_codec.encode(MESSAGE),
        ),
    )

    result = detector.detect_continuation([9, 8, 7, 6], [1, 2])

    assert result.strict_decode.status == "not_evaluated"
    assert result.hard_fill_decode.status == "not_evaluated"
    assert result.error_erasure_decode.status == "decoded"
    assert result.gated_message_bits == MESSAGE


def test_detector_can_skip_payload_decoding(monkeypatch) -> None:
    detector = make_detector(primary_policy="error_erasure", evaluate_all_policies=True)

    def fail_decode(_counting):
        raise AssertionError("payload decode should not run")

    monkeypatch.setattr(detector, "_strict_decode", fail_decode)
    monkeypatch.setattr(detector, "_hard_fill_decode", fail_decode)
    monkeypatch.setattr(detector, "_error_erasure_decode", fail_decode)

    result = detector.detect_continuation([9, 8, 7, 6], [1, 2], decode_payload=False)

    assert result.strict_decode.status == "not_evaluated"
    assert result.hard_fill_decode.status == "not_evaluated"
    assert result.error_erasure_decode.status == "not_evaluated"
    assert result.gated_message_bits is None


def test_selfhash_detector_uses_observed_token_for_partition_not_allocation() -> None:
    detector = DualLayerDetector(
        secret_key=b"secret",
        context_width=2,
        vocab_size=12,
        excluded_token_ids={0},
        prf_mode="paper_shared",
        ecc_codec=BCHCodec(23, 8, 3),
        threshold_mode="fixed",
        target_fpr=0.01,
        fixed_z_threshold=-100.0,
        calibrated_threshold=None,
        primary_counting_mode="all_tokens",
        unique_ngram_width=2,
        min_tokens_per_code_bit=1,
        hard_fill_value=0,
        primary_policy="tie_zero",
        max_erasure_assignments=64,
        seeding_scheme="selfhash",
        partition_engine="v2",
    )

    event = detector._events_known_boundary([2, 3], [1])[0]

    assert event.context == (3, 1)
    assert event.code_bit_index == 1
    assert event.region == 0
    assert event.upper_hit is True


def test_selfhash_unique_context_deduplicates_repeated_candidate_ngrams() -> None:
    detector = DualLayerDetector(
        secret_key=b"secret",
        context_width=2,
        vocab_size=12,
        excluded_token_ids={0},
        prf_mode="paper_shared",
        ecc_codec=BCHCodec(23, 8, 3),
        threshold_mode="fixed",
        target_fpr=0.01,
        fixed_z_threshold=-100.0,
        calibrated_threshold=None,
        primary_counting_mode="unique_context",
        unique_ngram_width=2,
        min_tokens_per_code_bit=1,
        hard_fill_value=0,
        primary_policy="tie_zero",
        max_erasure_assignments=64,
        seeding_scheme="selfhash",
        partition_engine="v2",
    )

    result = detector.detect_continuation([2, 3], [1, 3, 1])

    assert result.counting["all_tokens"].scored_tokens == 3
    assert result.counting["unique_context"].scored_tokens == 2


def test_selfhash_processor_and_detector_assign_same_candidate_region() -> None:
    codec = BCHCodec(23, 8, 3)
    processor = DualLayerLogitsProcessor(
        secret_key=b"secret",
        context_width=2,
        encoded_bits=codec.encode(MESSAGE),
        vocab_size=12,
        excluded_token_ids={0},
        presence_mode="soft",
        delta_presence=0.0,
        delta_payload=2.0,
        prf_mode="paper_shared",
        partition_engine="v2",
        seeding_scheme="selfhash",
        candidate_top_k=12,
    )
    detector = DualLayerDetector(
        secret_key=b"secret",
        context_width=2,
        vocab_size=12,
        excluded_token_ids={0},
        prf_mode="paper_shared",
        ecc_codec=codec,
        threshold_mode="fixed",
        target_fpr=0.01,
        fixed_z_threshold=-100.0,
        calibrated_threshold=None,
        primary_counting_mode="all_tokens",
        unique_ngram_width=2,
        min_tokens_per_code_bit=1,
        hard_fill_value=0,
        primary_policy="tie_zero",
        max_erasure_assignments=64,
        seeding_scheme="selfhash",
        partition_engine="v2",
    )
    processor(torch.tensor([[2, 3]]), torch.arange(12, dtype=torch.float32)[None])
    trace = processor.last_trace

    assert trace is not None
    observed_token = (trace.target_ids + trace.non_target_upper_ids)[0]
    event = detector._events_known_boundary([2, 3], [observed_token])[0]
    expected_region = (
        trace.embedded_bit
        if observed_token in trace.target_ids
        else 1 - trace.embedded_bit
    )
    assert event.context == trace.candidate_contexts[
        trace.candidate_ids.index(observed_token)
    ]
    assert event.code_bit_index == trace.code_bit_index
    assert event.region == expected_region
