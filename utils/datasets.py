from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from watermark.config import DatasetConfig


class SampleFilterError(ValueError):
    """A candidate that cannot form one exact-length three-class sample."""

    def __init__(
        self,
        sample_id: str,
        reason: str,
        *,
        raw_token_count: int,
        required_token_count: int,
    ) -> None:
        self.sample_id = str(sample_id)
        self.reason = str(reason)
        self.raw_token_count = int(raw_token_count)
        self.required_token_count = int(required_token_count)
        super().__init__(
            f"Sample {self.sample_id} filtered: {self.reason} "
            f"({self.raw_token_count} tokens, requires {self.required_token_count})"
        )


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    dataset: str
    prompt: str
    natural_completion: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_token_ids: tuple[int, ...] | None = None
    natural_token_ids: tuple[int, ...] | None = None


def _local_rows(config: DatasetConfig) -> Iterable[dict[str, Any]]:
    if not config.path:
        raise ValueError("--dataset-path is required for local JSON/JSONL datasets")
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl" or config.kind == "jsonl":

        def rows() -> Iterator[dict[str, Any]]:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number} must be a JSON object")
                    yield value

        return rows()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("JSON dataset must be a list of objects")
    return value


def _remote_rows(config: DatasetConfig) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    return load_dataset(
        config.name,
        config.config,
        split=config.split,
        streaming=config.streaming,
    )


def load_samples(config: DatasetConfig) -> Iterator[SampleRecord]:
    """Yield dataset candidates after the raw-row offset.

    The completed-sample target is enforced by ``run_generation()`` rather than
    here, because short candidates may be filtered after tokenization.
    """

    rows = _local_rows(config) if config.path else _remote_rows(config)
    for index, row in enumerate(rows):
        if index < config.sample_offset:
            continue
        sample_id = str(row.get(config.id_field, index)) if config.id_field else str(index)
        if config.kind == "c4" and "text" in row:
            yield SampleRecord(
                sample_id=sample_id,
                dataset="c4",
                prompt="",
                natural_completion=None,
                metadata={"raw_text": str(row["text"]), **{k: v for k, v in row.items() if k != "text"}},
            )
            continue
        if config.prompt_field not in row:
            raise ValueError(f"Missing prompt field '{config.prompt_field}' in row {index}")
        completion = None
        if config.completion_field:
            if config.completion_field not in row:
                raise ValueError(f"Missing completion field '{config.completion_field}' in row {index}")
            raw_completion = row[config.completion_field]
            completion = None if raw_completion is None else str(raw_completion)
        yield SampleRecord(
            sample_id=sample_id,
            dataset=config.kind,
            prompt=str(row[config.prompt_field]),
            natural_completion=completion,
            metadata={
                k: v
                for k, v in row.items()
                if k not in {config.prompt_field, config.completion_field, config.id_field}
            },
        )


def _extract_ids(value: Any) -> tuple[int, ...]:
    ids = value["input_ids"] if isinstance(value, dict) else value.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return tuple(int(x) for x in ids)


def tokenize_without_specials(tokenizer: Any, text: str) -> tuple[int, ...]:
    return _extract_ids(tokenizer(text, add_special_tokens=False))


def _build_inputs_with_special_tokens(tokenizer: Any, source_ids: Sequence[int]) -> tuple[int, ...]:
    values = [int(value) for value in source_ids]
    builder = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if callable(builder):
        return tuple(int(value) for value in builder(values))
    preparer = getattr(tokenizer, "prepare_for_model", None)
    if callable(preparer):
        return _extract_ids(
            preparer(
                values,
                add_special_tokens=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        )
    # Dependency-light fallback for injected test tokenizers. Production HF
    # tokenizers expose one of the two token-ID-native methods above.
    decoded = tokenizer.decode(values, skip_special_tokens=True)
    return _extract_ids(tokenizer(decoded, add_special_tokens=True))


def _prepare_prompt_ids(
    tokenizer: Any,
    source_ids: Sequence[int],
    *,
    prompt_capacity: int,
    context_width: int,
    sample_id: str,
) -> tuple[int, ...]:
    if prompt_capacity < context_width:
        raise ValueError(
            f"Model prompt capacity {prompt_capacity} is smaller than context width {context_width}"
        )
    candidate = tuple(int(value) for value in source_ids)
    while candidate:
        model_ids = _build_inputs_with_special_tokens(tokenizer, candidate)
        if len(model_ids) <= prompt_capacity:
            if len(model_ids) < context_width:
                raise SampleFilterError(
                    sample_id,
                    "insufficient_prompt_context",
                    raw_token_count=len(model_ids),
                    required_token_count=context_width,
                )
            return model_ids
        overflow = max(1, len(model_ids) - prompt_capacity)
        candidate = candidate[overflow:]
    raise SampleFilterError(
        sample_id,
        "insufficient_prompt_context",
        raw_token_count=0,
        required_token_count=context_width,
    )


def prepare_sample(
    sample: SampleRecord,
    tokenizer: Any,
    *,
    max_new_tokens: int,
    context_width: int,
    model_max_length: int,
) -> SampleRecord:
    """Prepare exact prompt/natural token IDs without decode/re-tokenize drift."""

    exact_length = int(max_new_tokens)
    if exact_length <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompt_capacity = int(model_max_length) - exact_length
    if prompt_capacity < context_width:
        raise ValueError(
            f"model_max_length={model_max_length} cannot fit context_width={context_width} "
            f"and exact continuation length={exact_length}"
        )

    raw_text = sample.metadata.get("raw_text")
    if raw_text is not None and not sample.prompt:
        raw_ids = tokenize_without_specials(tokenizer, str(raw_text))
        required = context_width + exact_length
        if len(raw_ids) < required:
            raise SampleFilterError(
                sample.sample_id,
                "insufficient_natural_tokens",
                raw_token_count=len(raw_ids),
                required_token_count=required,
            )
        natural_ids = raw_ids[-exact_length:]
        prompt_source_ids = raw_ids[:-exact_length]
    else:
        prompt_source_ids = tokenize_without_specials(tokenizer, sample.prompt)
        completion_ids = (
            ()
            if sample.natural_completion is None
            else tokenize_without_specials(tokenizer, sample.natural_completion)
        )
        if len(completion_ids) < exact_length:
            raise SampleFilterError(
                sample.sample_id,
                "insufficient_natural_tokens",
                raw_token_count=len(completion_ids),
                required_token_count=exact_length,
            )
        natural_ids = completion_ids[:exact_length]

    prompt_ids = _prepare_prompt_ids(
        tokenizer,
        prompt_source_ids,
        prompt_capacity=prompt_capacity,
        context_width=context_width,
        sample_id=sample.sample_id,
    )
    return replace(
        sample,
        prompt=tokenizer.decode(prompt_ids, skip_special_tokens=True),
        natural_completion=tokenizer.decode(natural_ids, skip_special_tokens=True),
        prompt_token_ids=tuple(prompt_ids),
        natural_token_ids=tuple(natural_ids),
    )
