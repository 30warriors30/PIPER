from __future__ import annotations

from typing import Sequence

from watermark.detector import DualLayerDetector


def synthetic_upper_sequence(detector: DualLayerDetector, prompt: Sequence[int], length: int) -> list[int]:
    history = list(prompt)
    output: list[int] = []
    for _ in range(length):
        partition, _ = detector.partition_for_context(history[-detector.context_width:])
        token = partition.bit0[0]
        output.append(token)
        history.append(token)
    return output
