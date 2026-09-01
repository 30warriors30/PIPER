from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def calibrate_threshold(negative_scores: Sequence[float], target_fpr: float) -> float:
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    minimum = math.ceil(1.0 / target_fpr)
    if len(negative_scores) < minimum:
        raise ValueError(
            f"At least {minimum} negative scores are required for target_fpr={target_fpr}; "
            f"received {len(negative_scores)}"
        )
    return float(np.quantile(np.asarray(negative_scores, dtype=float), 1.0 - target_fpr, method="higher"))
