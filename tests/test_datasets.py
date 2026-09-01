import json
from pathlib import Path

import pytest

from utils.datasets import SampleFilterError, SampleRecord, load_samples, prepare_sample
from watermark.config import DatasetConfig


class Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        ids = list(range(len(str(text).split())))
        if add_special_tokens:
            ids = [99] + ids
        return {"input_ids": ids}

    def build_inputs_with_special_tokens(self, ids):
        return [99] + list(ids)

    def decode(self, ids, skip_special_tokens=True):
        values = [int(value) for value in ids if not (skip_special_tokens and int(value) == 99)]
        return " ".join(f"t{value}" for value in values)


def test_c4_sample_preserves_exact_natural_tail_and_model_prompt_ids() -> None:
    sample = SampleRecord("1", "c4", "", None, {"raw_text": " ".join(["x"] * 30)})

    prepared = prepare_sample(
        sample,
        Tokenizer(),
        max_new_tokens=20,
        context_width=4,
        model_max_length=25,
    )

    assert prepared.natural_token_ids == tuple(range(10, 30))
    assert prepared.prompt_token_ids == (99, 6, 7, 8, 9)
    assert len(prepared.prompt_token_ids) + len(prepared.natural_token_ids) == 25
    assert prepared.prompt
    assert prepared.natural_completion


def test_c4_sample_shorter_than_context_plus_continuation_is_filtered() -> None:
    sample = SampleRecord("short", "c4", "", None, {"raw_text": " ".join(["x"] * 23)})

    with pytest.raises(SampleFilterError) as exc_info:
        prepare_sample(
            sample,
            Tokenizer(),
            max_new_tokens=20,
            context_width=4,
            model_max_length=64,
        )

    error = exc_info.value
    assert error.reason == "insufficient_natural_tokens"
    assert error.raw_token_count == 23
    assert error.required_token_count == 24


def test_load_samples_streams_past_requested_completed_target(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(json.dumps({"id": str(index), "prompt": f"p{index}", "completion": "a b c"}) + "\n" for index in range(3)),
        encoding="utf-8",
    )
    config = DatasetConfig(
        kind="jsonl",
        path=str(path),
        id_field="id",
        max_samples=1,
    )

    assert [sample.sample_id for sample in load_samples(config)] == ["0", "1", "2"]
