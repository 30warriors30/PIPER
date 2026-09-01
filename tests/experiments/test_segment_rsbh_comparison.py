import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.segment_rsbh_comparison import (
    build_frequency_mapping,
    build_parser,
    empirical_threshold,
    evaluate_segment_payload_prediction,
    disable_segment_rs_debug,
    repair_segment_gf_segments,
    segment_payload_to_bits,
    segment_rs_scheme,
    summarize_records,
)


def test_segment_rs_scheme_uses_8bit_adapter():
    assert segment_rs_scheme(8) == (3, 1, 8)


def test_segment_rs_scheme_rejects_unplanned_message_length():
    with pytest.raises(ValueError, match="Unsupported Segment-RSBH message length"):
        segment_rs_scheme(16)


def test_segment_payload_prediction_reports_exact_and_bit_accuracy():
    result = evaluate_segment_payload_prediction(
        pred_payload=0b00110001,
        expected_bits="00110001",
    )

    assert result["decoded_message_bits"] == "00110001"
    assert result["message_recovered"] is True
    assert result["original_bit_acc"] == 1.0


def test_segment_payload_prediction_clamps_to_requested_bit_width():
    assert segment_payload_to_bits(0b1_00110001, bit_width=8) == "00110001"


def test_segment_summary_uses_z_score_threshold_and_payload_metrics():
    summary = summarize_records(
        [
            {
                "split": "test",
                "text_class": "watermarked",
                "z_score": 3.0,
                "message_recovered": True,
                "original_bit_acc": 1.0,
            },
            {
                "split": "test",
                "text_class": "watermarked",
                "z_score": 1.0,
                "message_recovered": False,
                "original_bit_acc": 0.5,
            },
            {"split": "test", "text_class": "unwatermarked", "z_score": 2.0},
            {"split": "test", "text_class": "natural", "z_score": 0.5},
        ],
        threshold=2.5,
        target_fpr=0.01,
    )

    assert summary["tpr"] == 0.5
    assert summary["model_fpr"] == 0.0
    assert summary["natural_fpr"] == 0.0
    assert summary["combined_fpr"] == 0.0
    assert summary["exact_message_recovery"] == 0.5
    assert summary["end_to_end_exact_recovery"] == 0.5
    assert summary["mean_bit_accuracy"] == 0.75


def test_empirical_threshold_respects_target_fpr_with_ties():
    threshold = empirical_threshold([0.2, 1.0, 1.0, 3.0], target_fpr=0.25)

    assert threshold == 3.0


def test_empirical_threshold_moves_above_max_when_ties_would_exceed_fpr():
    threshold = empirical_threshold([0.2, 3.0, 3.0, 3.0], target_fpr=0.25)

    assert threshold > 3.0
    assert math.isfinite(threshold)


def test_frequency_mapping_assigns_every_vocab_id_to_a_segment():
    rows = [
        {"prompt_token_ids": [0, 1, 1, 2]},
        {"natural_token_ids": [2, 3]},
    ]

    mapping = build_frequency_mapping(rows, vocab_size=6, gf_segments_num=3)

    assert set(mapping) == set(range(6))
    assert set(mapping.values()) <= {0, 1, 2}


def test_disable_segment_rs_debug_turns_off_external_helper_debug():
    helper = SimpleNamespace(debug_active=True)
    obj = SimpleNamespace(rs=SimpleNamespace(helper=helper))

    disable_segment_rs_debug(obj)

    assert helper.debug_active is False


def test_repair_segment_gf_segments_pads_external_all_zero_rs_codeword():
    generator = SimpleNamespace(gf_segments_num=3, gf_segments=[0, 0])

    repair_segment_gf_segments(generator)

    assert generator.gf_segments == [0, 0, 0]


def test_repair_segment_gf_segments_rejects_overlong_codeword():
    generator = SimpleNamespace(gf_segments_num=3, gf_segments=[1, 2, 3, 4])

    with pytest.raises(RuntimeError, match="returned 4 GF segments"):
        repair_segment_gf_segments(generator)


def test_build_parser_defaults_match_confirmed_segment_baseline():
    args = build_parser().parse_args(
        [
            "--experiment-dir",
            "outputs/experiments/opt13b_pareto_49x200",
            "--segment-repo",
            "/tmp/segment-watermark",
            "--output-dir",
            "outputs/experiments/opt13b_pareto_49x200/comparisons/segment_rsbh_t200_b8_m2",
            "--model-path",
            "/data/yanlu/BREW/models/facebook/opt-1.3b",
        ]
    )

    assert args.exact_tokens == 200
    assert args.message_length == 8
    assert args.gamma == 0.5
    assert args.delta == 2.0
    assert args.target_fpr == 0.01
    assert args.brew_point_id == "soft_p0_m2"
    assert segment_rs_scheme(args.message_length) == (3, 1, 8)


def test_root_wrapper_exposes_segment_comparison_module():
    repo_root = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        [sys.executable, "-m", "experiments.segment_rsbh_comparison", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--segment-repo" in completed.stdout
