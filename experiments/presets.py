from __future__ import annotations

from experiments.config import OperatingPoint


_SMOKE = (
    OperatingPoint("soft", 0.5, 0.5),
    OperatingPoint("soft", 1.0, 2.0),
    OperatingPoint("soft", 2.0, 3.0),
    OperatingPoint("hard", 0.0, 2.0),
)

_PAPER = (OperatingPoint("soft", 0.0, 2.0),)


def preset_points(name: str) -> tuple[OperatingPoint, ...]:
    normalized = name.strip().lower()
    if normalized == "paper":
        return _PAPER
    if normalized == "smoke":
        return _SMOKE
    if normalized in {"pilot", "full"}:
        presence_values = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
        payload_values = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
        soft = tuple(
            OperatingPoint("soft", delta_presence, delta_payload)
            for delta_presence in presence_values
            for delta_payload in payload_values
        )
        hard = tuple(OperatingPoint("hard", 0.0, delta_payload) for delta_payload in payload_values)
        return soft + hard
    raise ValueError("preset must be one of: paper, smoke, pilot, full")
