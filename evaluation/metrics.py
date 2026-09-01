from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from evaluation.roc import roc_summary


def _rate(records: Sequence[dict[str, Any]], field: str = "detected") -> float | None:
    if not records:
        return None
    return sum(bool(record.get(field, False)) for record in records) / len(records)


def _bits(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(int(char) for char in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(bit) for bit in value)
    return None


def _spurious_attribution_rate(records: Sequence[dict[str, Any]]) -> float | None:
    """Rate of negative texts that both pass presence and receive an identity."""
    if not records:
        return None
    attributed = sum(
        bool(record.get("detected", False))
        and _bits(record.get("decoded_message")) is not None
        for record in records
    )
    return attributed / len(records)


def _ber(pairs: Sequence[tuple[tuple[int, ...], tuple[int, ...]]]) -> float | None:
    errors = total = 0
    for expected, observed in pairs:
        if len(expected) != len(observed):
            continue
        errors += sum(left != right for left, right in zip(expected, observed, strict=True))
        total += len(expected)
    return None if total == 0 else errors / total


def _presence(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    watermarked = [record for record in records if record.get("text_class") == "watermarked"]
    unwatermarked = [record for record in records if record.get("text_class") == "unwatermarked"]
    natural = [record for record in records if record.get("text_class") == "natural"]
    labels = [1 if record.get("text_class") == "watermarked" else 0 for record in records]
    scores = [float(record.get("z_score", 0.0)) for record in records]
    curve = roc_summary(labels, scores)
    tpr = _rate(watermarked)
    model_fpr = _rate(unwatermarked)
    natural_fpr = _rate(natural)
    return {
        "count": len(records),
        "watermarked_count": len(watermarked),
        "unwatermarked_count": len(unwatermarked),
        "natural_count": len(natural),
        "tpr": tpr,
        "fnr": None if tpr is None else 1.0 - tpr,
        "model_fpr": model_fpr,
        "model_tnr": None if model_fpr is None else 1.0 - model_fpr,
        "model_spurious_attribution_rate": _spurious_attribution_rate(unwatermarked),
        "natural_fpr": natural_fpr,
        "natural_tnr": None if natural_fpr is None else 1.0 - natural_fpr,
        "natural_spurious_attribution_rate": _spurious_attribution_rate(natural),
        "auc": curve["auc"],
        "roc": {"fpr": curve["fpr"], "tpr": curve["tpr"], "thresholds": curve["thresholds"]},
    }


def _policy_metrics(records: Sequence[dict[str, Any]], policy: str) -> dict[str, Any]:
    decoded_count = 0
    exact_count = 0
    message_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    statuses: Counter[str] = Counter()
    for record in records:
        payload = record.get(policy)
        if not isinstance(payload, dict):
            statuses["missing"] += 1
            continue
        status = str(payload.get("status", "missing"))
        statuses[status] += 1
        if status != "decoded":
            continue
        decoded_count += 1
        expected = _bits(record.get("message_bits"))
        observed = _bits(payload.get("message_bits"))
        if expected is None or observed is None:
            continue
        message_pairs.append((expected, observed))
        if expected == observed:
            exact_count += 1
    total = len(records)
    return {
        f"{policy}_decode_success_rate": None if total == 0 else decoded_count / total,
        f"{policy}_exact_message_recovery": None if total == 0 else exact_count / total,
        f"{policy}_conditional_message_ber": _ber(message_pairs),
        f"{policy}_decode_status_counts": dict(sorted(statuses.items())),
    }


def _payload(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    watermarked = [record for record in records if record.get("text_class") == "watermarked"]
    exact = 0
    wrong = 0
    detected_count = 0
    detected_abstentions = 0
    message_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    code_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for record in watermarked:
        detected = bool(record.get("detected", False))
        detected_count += int(detected)
        expected_message = _bits(record.get("message_bits"))
        decoded = _bits(record.get("decoded_message"))
        expected_code = _bits(record.get("encoded_bits"))
        recovered_code = _bits(record.get("recovered_codeword_bits"))
        if expected_message is not None and decoded is not None:
            message_pairs.append((expected_message, decoded))
            if expected_message == decoded:
                exact += 1
            else:
                wrong += 1
        elif detected:
            detected_abstentions += 1
        if expected_code is not None and recovered_code is not None:
            code_pairs.append((expected_code, recovered_code))
    statuses = Counter(str(record.get("decode_status")) for record in watermarked)
    result: dict[str, Any] = {
        "watermarked_count": len(watermarked),
        "exact_message_recovery": None if not watermarked else exact / len(watermarked),
        "correct_attribution_rate": None if not watermarked else exact / len(watermarked),
        "conditional_decoding_accuracy": None
        if detected_count == 0
        else exact / detected_count,
        "misattribution_rate": None if not watermarked else wrong / len(watermarked),
        "decoder_abstention_rate": None
        if detected_count == 0
        else detected_abstentions / detected_count,
        "message_ber": _ber(message_pairs),
        "codeword_ber": _ber(code_pairs),
        "decode_status_counts": dict(sorted(statuses.items())),
    }
    for policy in ("tie_zero", "strict", "hard_fill", "error_erasure"):
        result.update(_policy_metrics(watermarked, policy))
    return result


def deduplicate(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        if record.get("status", "completed") != "completed":
            continue
        key = (str(record.get("sample_id")), str(record.get("text_class")), str(record.get("input_mode")))
        keyed[key] = record
    return list(keyed.values())


def evaluate_records(records: Sequence[dict[str, Any]], input_mode: str = "all") -> dict[str, Any]:
    clean = deduplicate(records)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in clean:
        by_mode[str(record.get("input_mode"))].append(record)
    if input_mode != "all":
        by_mode = defaultdict(list, {input_mode: by_mode.get(input_mode, [])})
    headline = "blind_text" if "blind_text" in by_mode else (sorted(by_mode)[0] if by_mode else "unspecified")
    runtime: dict[str, list[float]] = defaultdict(list)
    for record in clean:
        for key, value in record.items():
            if key.endswith("_seconds") and isinstance(value, (int, float)):
                runtime[key].append(float(value))
    return {
        "headline_input_mode": headline,
        "presence": _presence(by_mode.get(headline, [])),
        "presence_by_input_mode": {mode: _presence(values) for mode, values in sorted(by_mode.items())},
        "payload": _payload(by_mode.get(headline, [])),
        "payload_by_input_mode": {mode: _payload(values) for mode, values in sorted(by_mode.items())},
        "runtime": {key: float(np.mean(values)) for key, values in sorted(runtime.items()) if values},
    }
