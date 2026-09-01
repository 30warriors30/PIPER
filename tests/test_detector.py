from watermark.detector import DualLayerDetector
from watermark.ecc import BCHCodec
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
