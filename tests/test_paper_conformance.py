from __future__ import annotations

import math
from pathlib import Path

from experiments.config import ParetoExperimentConfig
from experiments.presets import preset_points
from watermark.config import DecodingConfig, DetectionConfig, WatermarkConfig
from watermark.detector import DualLayerDetector
from watermark.ecc import BCHCodec
from watermark.result_types import CountingResult


def make_detector(
    *,
    target_fpr: float = 0.01,
    presence_test: str = "exact_binomial",
    primary_counting_mode: str = "unique_context",
    primary_policy: str = "tie_zero",
) -> DualLayerDetector:
    return DualLayerDetector(
        secret_key=b"paper-key",
        context_width=1,
        vocab_size=10,
        excluded_token_ids={0},
        prf_mode="paper_shared",
        ecc_codec=BCHCodec(23, 8, 3),
        threshold_mode="theoretical",
        target_fpr=target_fpr,
        fixed_z_threshold=2.326347874,
        calibrated_threshold=None,
        primary_counting_mode=primary_counting_mode,
        unique_ngram_width=1,
        min_tokens_per_code_bit=1,
        hard_fill_value=0,
        primary_policy=primary_policy,
        max_erasure_assignments=64,
        evaluate_all_policies=False,
        presence_test=presence_test,
    )


def test_paper_defaults_use_child_only_payload_bias() -> None:
    watermark = WatermarkConfig(secret_key="secret")
    detection = DetectionConfig()
    decoding = DecodingConfig()

    assert watermark.presence_mode == "soft"
    assert watermark.delta_presence == 0.0
    assert watermark.delta_payload == 2.0
    assert detection.presence_test == "exact_binomial"
    assert detection.counting_mode == "unique_context"
    assert decoding.primary_policy == "tie_zero"


def test_paper_experiment_preset_contains_only_the_headline_point() -> None:
    points = preset_points("paper")
    config = ParetoExperimentConfig(
        model_path="model",
        secret_key="secret",
        output_root=Path("outputs"),
        experiment_id="paper",
    )

    assert [point.point_id for point in points] == ["soft_p0_m2"]
    assert config.presence_test == "exact_binomial"
    assert config.counting_mode == "unique_context"


def test_unique_context_exact_binomial_uses_actual_partition_probability() -> None:
    detector = make_detector(target_fpr=0.5)
    repeated_token = next(
        token
        for token in range(1, detector.vocab_size)
        if token in detector.partition_for_context((token,))[0].upper
    )

    result = detector.detect_continuation([repeated_token], [repeated_token] * 3)
    counting = result.counting["unique_context"]

    assert counting.scored_tokens == 1
    assert counting.upper_hits == 1
    assert math.isclose(counting.null_probability, 4.0 / 9.0)
    assert math.isclose(counting.exact_p_value, 4.0 / 9.0)
    assert result.detected is True


def test_excluded_tokens_do_not_enter_effective_sample_size() -> None:
    detector = make_detector()

    result = detector.detect_continuation([1], [0])

    assert result.counting["unique_context"].scored_tokens == 0
    assert result.counting["unique_context"].exact_p_value == 1.0


def test_rejected_presence_gate_never_invokes_payload_decoder(monkeypatch) -> None:
    detector = make_detector(target_fpr=1e-9)

    def fail_decode(_counting):
        raise AssertionError("payload decoder must not run before presence accepts")

    monkeypatch.setattr(detector, "_strict_decode", fail_decode)
    monkeypatch.setattr(detector, "_hard_fill_decode", fail_decode)
    monkeypatch.setattr(detector, "_error_erasure_decode", fail_decode)

    result = detector.detect_continuation([1], [0])

    assert result.detected is False
    assert result.decoder_invoked is False
    assert result.gated_message_bits is None


def test_tie_zero_policy_maps_ties_and_unseen_positions_to_zero() -> None:
    detector = make_detector()
    n0 = [0] * detector.ecc_codec.n
    n1 = [0] * detector.ecc_codec.n
    n0[0], n1[0] = 2, 2
    n0[1], n1[1] = 1, 3
    n0[2], n1[2] = 4, 1
    counting = CountingResult(
        mode="unique_context",
        scored_tokens=13,
        upper_hits=8,
        z_score=0.0,
        exact_p_value=0.5,
        n0=tuple(n0),
        n1=tuple(n1),
        erasures=tuple(range(3, detector.ecc_codec.n)),
    )

    codeword = detector.recover_codeword(counting, policy="tie_zero")

    assert codeword == (0, 1, 0) + (0,) * (detector.ecc_codec.n - 3)
