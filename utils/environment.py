from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "transformers", "datasets", "galois", "numpy", "scipy", "scikit-learn")
        },
    }
    try:
        import torch

        result["cuda"] = {
            "available": torch.cuda.is_available(),
            "runtime": torch.version.cuda,
            "architectures": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover
        result["cuda_error"] = str(exc)
    return result
