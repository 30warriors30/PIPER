from __future__ import annotations

from typing import Any, Sequence

from sklearn.metrics import roc_auc_score, roc_curve


def roc_summary(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    if not labels or len(set(labels)) < 2:
        return {"auc": None, "fpr": [], "tpr": [], "thresholds": []}
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "fpr": [float(value) for value in fpr],
        "tpr": [float(value) for value in tpr],
        "thresholds": [float(value) for value in thresholds],
    }
