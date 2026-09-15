from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.generation import PairedGenerator, run_generation
from utils.io import iter_jsonl
from watermark.config import (
    DatasetConfig,
    DecodingConfig,
    DetectionConfig,
    ECCConfig,
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
    OutputConfig,
    WatermarkConfig,
)
from watermark.ecc import BCHCodec


class FakeTokenizer:
    all_special_ids = [0, 1, 2]
    eos_token_id = 2
    pad_token_id = 0
    model_max_length = 128

    def __len__(self):
        return 32

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = ([1] if add_special_tokens else []) + [3 + (ord(char) % 20) for char in str(text)]
        if return_tensors == "pt":
            tensor = torch.tensor([ids], dtype=torch.long)
            return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}
        return {"input_ids": ids}

    def build_inputs_with_special_tokens(self, ids):
        return [1] + list(ids)

    def decode(self, ids, skip_special_tokens=True):
        values = [int(value) for value in ids if not (skip_special_tokens and int(value) in self.all_special_ids)]
        return "".join(chr(65 + (value % 26)) for value in values)


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(vocab_size=32, max_position_embeddings=128)
        self.generation_calls: list[dict[str, int]] = []
        self.processor_configs: list[tuple[str, str, int | None]] = []

    def generate(self, input_ids, min_new_tokens, max_new_tokens, logits_processor=None, **kwargs):
        self.generation_calls.append(
            {"min_new_tokens": int(min_new_tokens), "max_new_tokens": int(max_new_tokens)}
        )
        if logits_processor:
            processors = (
                logits_processor
                if isinstance(logits_processor, (list, tuple))
                else [logits_processor]
            )
            processor = processors[0]
            self.processor_configs.append(
                (
                    processor.seeding_scheme,
                    processor.partition_engine,
                    processor.candidate_top_k,
                )
            )
        history = input_ids.clone()
        for _ in range(max_new_tokens):
            scores = torch.randn((1, self.config.vocab_size))
            if logits_processor:
                for processor in processors:
                    scores = processor(history, scores)
            probs = torch.softmax(scores, dim=-1)
            token = torch.multinomial(probs, 1)
            history = torch.cat([history, token], dim=1)
        return history

    def eval(self):
        return self


def make_config(tmp_path: Path, data: Path, *, kind: str = "jsonl", max_samples: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(path="fake", device="cpu", dtype="float32"),
        dataset=DatasetConfig(
            kind=kind,
            path=str(data),
            prompt_field="prompt",
            completion_field="completion",
            id_field="id",
            max_samples=max_samples,
        ),
        generation=GenerationConfig(max_new_tokens=8, top_k=50),
        watermark=WatermarkConfig(
            secret_key="secret",
            context_width=2,
            candidate_top_k=50,
            seeding_scheme="selfhash",
            partition_engine="v2",
        ),
        ecc=ECCConfig(),
        detection=DetectionConfig(),
        decoding=DecodingConfig(),
        output=OutputConfig(root=str(tmp_path / "outputs"), run_id="run"),
    )


def test_generation_kwargs_force_equal_minimum_and_maximum(tmp_path: Path) -> None:
    data = tmp_path / "unused.jsonl"
    data.write_text("", encoding="utf-8")
    config = make_config(tmp_path, data)
    model = FakeModel()
    generator = PairedGenerator(
        config,
        BCHCodec(23, 8, 3),
        model=model,
        tokenizer=FakeTokenizer(),
        device=torch.device("cpu"),
    )

    kwargs = generator._generation_kwargs()

    assert kwargs["min_new_tokens"] == 8
    assert kwargs["max_new_tokens"] == 8
    assert kwargs["top_k"] == 50


def test_generation_workflow_writes_three_exact_length_text_classes(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps({"id": "a", "prompt": "Alpha", "completion": "NaturalText"}) + "\n",
        encoding="utf-8",
    )
    config = make_config(tmp_path, data)
    model = FakeModel()

    result = run_generation(
        config,
        model=model,
        tokenizer=FakeTokenizer(),
        device=torch.device("cpu"),
    )

    assert result == {
        "run_dir": str(config.run_dir),
        "processed": 1,
        "skipped": 0,
        "filtered": 0,
        "failed": 0,
    }
    records = list(iter_jsonl(tmp_path / "outputs" / "run" / "samples.jsonl"))
    record = records[0]
    assert record["status"] == "completed"
    assert len(record["message_bits"]) == 8
    assert len(record["encoded_bits"]) == 23
    assert len(record["watermarked"]["token_ids"]) == 8
    assert len(record["unwatermarked"]["token_ids"]) == 8
    assert len(record["natural"]["token_ids"]) == 8
    assert all(call == {"min_new_tokens": 8, "max_new_tokens": 8} for call in model.generation_calls)
    assert model.processor_configs == [("selfhash", "v2", 50)]


def test_generation_filters_short_c4_and_continues_until_target(tmp_path: Path) -> None:
    data = tmp_path / "c4.jsonl"
    data.write_text(
        json.dumps({"id": "short", "text": "abc"}) + "\n"
        + json.dumps({"id": "valid", "text": "abcdefghijklmnopqrst"}) + "\n",
        encoding="utf-8",
    )
    config = make_config(tmp_path, data, kind="c4", max_samples=1)

    result = run_generation(
        config,
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device=torch.device("cpu"),
    )

    assert result["processed"] == 1
    assert result["filtered"] == 1
    completed = [
        record
        for record in iter_jsonl(config.run_dir / "samples.jsonl")
        if record.get("status") == "completed"
    ]
    assert [record["sample_id"] for record in completed] == ["valid"]
    filtered = list(iter_jsonl(config.run_dir / "errors" / "filtered_samples.jsonl"))
    assert filtered[0]["sample_id"] == "short"
    assert filtered[0]["reason"] == "insufficient_natural_tokens"
