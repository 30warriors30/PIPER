from evaluation.metrics import evaluate_records


def test_known_boundary_and_blind_metrics_are_separate() -> None:
    records = [
        {
            "sample_id": "1",
            "text_class": "watermarked",
            "input_mode": "known_boundary",
            "z_score": 5,
            "detected": True,
            "message_bits": "10",
            "decoded_message": "10",
            "decode_status": "decoded",
            "tie_zero": {"status": "decoded", "message_bits": [1, 0]},
            "strict": {"status": "decoded", "message_bits": [1, 0]},
            "hard_fill": {"status": "decoded", "message_bits": [1, 1]},
            "error_erasure": {"status": "decoded", "message_bits": [1, 0]},
        },
        {
            "sample_id": "1",
            "text_class": "unwatermarked",
            "input_mode": "known_boundary",
            "z_score": -1,
            "detected": False,
            "decode_status": "too_many_erasures",
            "tie_zero": {"status": "not_evaluated", "message_bits": None},
            "strict": {"status": "insufficient_evidence", "message_bits": None},
            "hard_fill": {"status": "decoded", "message_bits": [0, 0]},
            "error_erasure": {"status": "too_many_erasures", "message_bits": None},
        },
        {
            "sample_id": "1",
            "text_class": "watermarked",
            "input_mode": "blind_text",
            "z_score": 0,
            "detected": False,
            "message_bits": "10",
            "decoded_message": None,
            "decode_status": "too_many_erasures",
            "tie_zero": {"status": "not_evaluated", "message_bits": None},
            "strict": {"status": "insufficient_evidence", "message_bits": None},
            "hard_fill": {"status": "decoded", "message_bits": [1, 0]},
            "error_erasure": {"status": "too_many_erasures", "message_bits": None},
        },
        {
            "sample_id": "1",
            "text_class": "unwatermarked",
            "input_mode": "blind_text",
            "z_score": 4,
            "detected": True,
            "decode_status": "too_many_erasures",
            "tie_zero": {"status": "not_evaluated", "message_bits": None},
            "strict": {"status": "insufficient_evidence", "message_bits": None},
            "hard_fill": {"status": "decoded", "message_bits": [0, 0]},
            "error_erasure": {"status": "too_many_erasures", "message_bits": None},
        },
    ]
    metrics = evaluate_records(records)
    assert metrics["presence_by_input_mode"]["known_boundary"]["tpr"] == 1.0
    assert metrics["presence_by_input_mode"]["known_boundary"]["model_spurious_attribution_rate"] == 0.0
    assert metrics["presence_by_input_mode"]["blind_text"]["tpr"] == 0.0
    payload = metrics["payload_by_input_mode"]["known_boundary"]
    assert payload["exact_message_recovery"] == 1.0
    assert payload["correct_attribution_rate"] == 1.0
    assert payload["conditional_decoding_accuracy"] == 1.0
    assert payload["tie_zero_decode_success_rate"] == 1.0
    assert payload["tie_zero_exact_message_recovery"] == 1.0
    assert payload["strict_decode_success_rate"] == 1.0
    assert payload["strict_exact_message_recovery"] == 1.0
    assert payload["hard_fill_decode_success_rate"] == 1.0
    assert payload["hard_fill_exact_message_recovery"] == 0.0
    assert payload["hard_fill_conditional_message_ber"] == 0.5
    assert payload["error_erasure_decode_success_rate"] == 1.0
    assert payload["error_erasure_exact_message_recovery"] == 1.0
    assert payload["error_erasure_decode_status_counts"] == {"decoded": 1}
