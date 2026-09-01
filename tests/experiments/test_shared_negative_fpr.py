from __future__ import annotations

import math

from experiments.score_shared_negative_fpr import empirical_threshold, summarize_fixed_thresholds


def test_summarize_fixed_thresholds_counts_scores_at_or_above_threshold() -> None:
    scores = [0.0, 2.4, 3.2, 4.0]

    summary = summarize_fixed_thresholds(scores, target_fprs=[0.01, 0.001])

    one_percent = summary["fpr_0p01"]
    assert math.isclose(one_percent["fixed_z_threshold"], 2.3263478740408408)
    assert one_percent["false_positives"] == 3
    assert one_percent["empirical_fpr"] == 0.75

    point_one_percent = summary["fpr_0p001"]
    assert math.isclose(point_one_percent["fixed_z_threshold"], 3.090232306167813)
    assert point_one_percent["false_positives"] == 2
    assert point_one_percent["empirical_fpr"] == 0.5


def test_empirical_threshold_returns_first_threshold_at_or_below_target_fpr() -> None:
    scores = [1.0, 2.0, 3.0, 4.0]

    result = empirical_threshold(scores, target_fpr=0.25)

    assert result == {
        "threshold": 4.0,
        "false_positives": 1,
        "empirical_fpr": 0.25,
    }
