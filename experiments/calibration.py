from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def calibrate_threshold_from_scores(scores: Sequence[float], *, target_fpr: float) -> dict[str, Any]:
    if not scores:
        raise ValueError("Calibration scores must not be empty")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be in (0, 1)")
    values = np.asarray(scores, dtype=float)
    quantile = float(np.quantile(values, 1.0 - target_fpr, method="higher"))
    threshold = float(np.nextafter(quantile, np.inf))
    empirical_fpr = float(np.mean(values >= threshold))
    return {
        "target_fpr": float(target_fpr),
        "calibration_count": int(values.size),
        "score_quantile": quantile,
        "calibrated_threshold": threshold,
        "empirical_fpr": empirical_fpr,
        "min_score": float(values.min()),
        "max_score": float(values.max()),
    }
