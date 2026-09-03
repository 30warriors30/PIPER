from __future__ import annotations

import gc
import inspect
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class GeneratedSequence:
    token_ids: tuple[int, ...]
    text: str
    seconds: float
    batch_size: int


@dataclass(frozen=True)
class CompletedBatch(Generic[T, R]):
    items: tuple[T, ...]
    results: tuple[R, ...]
    batch_size: int


def is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    message = str(exc).lower()
    return "cuda out of memory" in message or "cublas_status_alloc_failed" in message


def _clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def adaptive_batches(
    items: Sequence[T],
    *,
    batch_size: int,
    run: Callable[[Sequence[T]], Sequence[R]],
) -> Iterator[CompletedBatch[T, R]]:
    """Run ordered batches, halving the active size after a CUDA OOM.

    A reduced size remains in effect for the rest of the input, so a long job
    does not repeatedly hit the same OOM. Non-OOM failures are never hidden.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    position = 0
    active_size = int(batch_size)
    while position < len(items):
        chunk = tuple(items[position : position + active_size])
        try:
            results = tuple(run(chunk))
        except RuntimeError as exc:
            if not is_cuda_oom(exc) or len(chunk) <= 1:
                raise
            active_size = max(1, len(chunk) // 2)
            _clear_cuda_cache()
            continue
        if len(results) != len(chunk):
            raise RuntimeError(
                f"Batch runner returned {len(results)} results for {len(chunk)} inputs"
            )
        yield CompletedBatch(items=chunk, results=results, batch_size=len(chunk))
        position += len(chunk)


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative > float(top_p)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    return torch.full_like(logits, -torch.inf).scatter(
        -1, sorted_indices, sorted_logits
    )


def _eos_ids(tokenizer: Any) -> tuple[int, ...]:
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(int(token_id) for token_id in value)
    return (int(value),)


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def generate_exact_batch(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt_token_ids: Sequence[Sequence[int]],
    seeds: Sequence[int],
    exact_tokens: int,
    temperature: float,
    top_p: float,
    processor: Any | None,
    do_sample: bool = True,
) -> tuple[GeneratedSequence, ...]:
    """Generate exact-length decoder-only continuations in one model batch.

    Model forward passes and KV cache updates are batched. Sampling keeps one
    generator per row, preserving each sample's output when batch grouping or
    resume boundaries change.
    """

    if not prompt_token_ids:
        return ()
    if len(prompt_token_ids) != len(seeds):
        raise ValueError("prompt_token_ids and seeds must have the same length")
    if exact_tokens <= 0:
        raise ValueError("exact_tokens must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False)):
        raise ValueError(
            "PIPER batched generation currently supports decoder-only models"
        )

    prompts = [tuple(int(value) for value in prompt) for prompt in prompt_token_ids]
    if any(not prompt for prompt in prompts):
        raise ValueError("prompt token sequences must be non-empty")
    eos_ids = _eos_ids(tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_ids[0] if eos_ids else None
    if pad_token_id is None:
        raise ValueError(
            "A pad_token_id or eos_token_id is required for batched generation"
        )

    batch_size = len(prompts)
    prompt_width = max(len(prompt) for prompt in prompts)
    input_ids = torch.full(
        (batch_size, prompt_width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, prompt in enumerate(prompts):
        input_ids[row_index, -len(prompt) :] = torch.tensor(
            prompt, dtype=torch.long, device=device
        )
        attention_mask[row_index, -len(prompt) :] = 1

    generators: list[torch.Generator] = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        generators.append(generator)

    full_ids = input_ids
    next_model_ids = input_ids
    past_key_values = None
    forward = getattr(model, "forward", None) or model.__call__
    try:
        forward_parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        forward_parameters = {}
    keeps_last_logit = "logits_to_keep" in forward_parameters
    accepts_position_ids = "position_ids" in forward_parameters
    _sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(int(exact_tokens)):
            model_kwargs: dict[str, Any] = {
                "input_ids": next_model_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
                "return_dict": True,
            }
            if past_key_values is not None:
                model_kwargs["past_key_values"] = past_key_values
            if keeps_last_logit:
                model_kwargs["logits_to_keep"] = 1
            if accepts_position_ids:
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
                model_kwargs["position_ids"] = (
                    position_ids[:, -1:]
                    if past_key_values is not None
                    else position_ids
                )
            outputs = model(**model_kwargs)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                logits = outputs[0]
            scores = logits[:, -1, :].to(dtype=torch.float32)
            if processor is not None:
                scores = processor(full_ids, scores)
            scores = scores / float(temperature)
            if eos_ids:
                scores[:, list(eos_ids)] = -torch.inf
            scores = _top_p_filter(scores, float(top_p))
            probabilities = torch.softmax(scores, dim=-1)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError(
                    "Non-finite sampling probabilities during generation"
                )
            if do_sample:
                sampled = [
                    torch.multinomial(probabilities[row], 1, generator=generators[row])
                    for row in range(batch_size)
                ]
                next_tokens = torch.cat(sampled, dim=0)
            else:
                next_tokens = torch.argmax(probabilities, dim=-1)
            full_ids = torch.cat((full_ids, next_tokens[:, None]), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (batch_size, 1), dtype=attention_mask.dtype, device=device
                    ),
                ),
                dim=1,
            )
            past_key_values = getattr(outputs, "past_key_values", None)
            next_model_ids = (
                next_tokens[:, None] if past_key_values is not None else full_ids
            )

    _sync(device)
    elapsed = time.perf_counter() - started
    amortized_seconds = elapsed / batch_size
    continuations = full_ids[:, prompt_width:]
    results = []
    for row in range(batch_size):
        token_ids = tuple(int(value) for value in continuations[row].tolist())
        if len(token_ids) != int(exact_tokens):
            raise RuntimeError(
                f"Exact-length invariant failed: got {len(token_ids)}, expected {exact_tokens}"
            )
        results.append(
            GeneratedSequence(
                token_ids=token_ids,
                text=tokenizer.decode(token_ids, skip_special_tokens=True),
                seconds=amortized_seconds,
                batch_size=batch_size,
            )
        )
    return tuple(results)
