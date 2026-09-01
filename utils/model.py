from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # Keep detector-only workflows usable without the generation stack.
    torch = None  # type: ignore[assignment]

from watermark.config import ModelConfig


def resolve_device(value: str) -> torch.device:
    if torch is None:
        raise RuntimeError("PyTorch is required for model loading and generation")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def torch_dtype(value: str) -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for model loading and generation")
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {value}")
    return mapping[value]


def load_model_and_tokenizer(config: ModelConfig) -> tuple[Any, Any, torch.device]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(config.path).expanduser()
    if config.local_files_only and not path.exists():
        raise FileNotFoundError(path)
    device = resolve_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.path, local_files_only=config.local_files_only)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"local_files_only": config.local_files_only}
    dtype = torch_dtype(config.dtype)
    # Transformers supports dtype in current releases; fall back for older 4.x builds.
    try:
        model = AutoModelForCausalLM.from_pretrained(config.path, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(config.path, torch_dtype=dtype, **kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def model_max_length(model: Any, tokenizer: Any) -> int:
    candidates = [
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    valid = [int(value) for value in candidates if isinstance(value, int) and value > 0 and value < 10**8]
    return min(valid) if valid else 2048


def excluded_token_ids(tokenizer: Any, *, exclude_special_tokens: bool, exclude_eos: bool) -> set[int]:
    if not exclude_special_tokens:
        return set()
    excluded = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}
    eos = getattr(tokenizer, "eos_token_id", None)
    if not exclude_eos and eos is not None:
        excluded.discard(int(eos))
    return excluded
