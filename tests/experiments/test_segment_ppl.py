import math
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.evaluate_segment_ppl import load_segment_quality_examples, score_segment_ppl
from utils.io import append_jsonl, iter_jsonl, write_json


class UniformCausalModel:
    def __init__(self, vocab_size: int = 8):
        self.config = SimpleNamespace(vocab_size=vocab_size, max_position_embeddings=128)

    def __call__(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(batch, length, self.config.vocab_size, device=input_ids.device))


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.model_max_length = 128
        self._vocab = {f"t{i}": i for i in range(8)}

    def get_vocab(self):
        return dict(self._vocab)


def _build_segment_comparison(root: Path) -> Path:
    experiment_dir = root / "experiment"
    comparison_dir = experiment_dir / "comparisons" / "segment_rsbh_t200_b8_m2"
    write_json(
        comparison_dir / "metadata.json",
        {
            "schema_version": 1,
            "baseline": "Segment-RSBH",
            "experiment_dir": str(experiment_dir),
            "model_path": "/models/opt-1.3b",
            "rs_scheme": {"n": 3, "k": 1, "m": 8},
        },
    )
    append_jsonl(
        experiment_dir / "shared" / "baseline.jsonl",
        {
            "sample_id": "a",
            "split": "test",
            "prompt_token_ids": [1, 2],
            "unwatermarked_token_ids": [3, 4],
            "natural_token_ids": [4, 4],
        },
    )
    append_jsonl(
        comparison_dir / "records.jsonl",
        {
            "sample_id": "a",
            "split": "test",
            "text_class": "watermarked",
            "prompt_token_ids": [1, 2],
            "token_ids": [5, 6],
        },
    )
    append_jsonl(
        comparison_dir / "records.jsonl",
        {
            "sample_id": "b",
            "split": "calibration",
            "text_class": "watermarked",
            "prompt_token_ids": [1, 2],
            "token_ids": [6, 7],
        },
    )
    return comparison_dir


def test_load_segment_quality_examples_uses_test_watermarked_and_shared_baseline(tmp_path: Path) -> None:
    comparison_dir = _build_segment_comparison(tmp_path)

    examples = load_segment_quality_examples(comparison_dir, include_shared=True)

    assert [(row["sample_id"], row["text_class"]) for row in examples] == [
        ("a", "unwatermarked"),
        ("a", "natural"),
        ("a", "watermarked"),
    ]
    assert examples[-1]["continuation_token_ids"] == [5, 6]


def test_score_segment_ppl_writes_quality_rows_and_summary(tmp_path: Path) -> None:
    comparison_dir = _build_segment_comparison(tmp_path)
    tokenizer = FakeTokenizer()

    result = score_segment_ppl(
        comparison_dir=comparison_dir,
        evaluator_model_path="/models/opt-6.7b",
        evaluator_name="opt-6.7b",
        batch_size=2,
        model=UniformCausalModel(),
        generation_tokenizer=tokenizer,
        evaluator_tokenizer=tokenizer,
        device=torch.device("cpu"),
        overwrite=False,
    )

    output_dir = comparison_dir / "external_quality" / "opt-6_7b"
    rows = list(iter_jsonl(output_dir / "quality.jsonl"))
    assert result["scored"] == 3
    assert len(rows) == 3
    assert {row["text_class"] for row in rows} == {"unwatermarked", "natural", "watermarked"}
    assert math.isclose(result["summary"]["ppl_ratio"], 1.0, abs_tol=1e-8)
    assert math.isclose(
        result["summary"]["classes"]["watermarked"]["corpus_ppl"],
        8.0,
        rel_tol=1e-6,
    )
    assert (output_dir / "summary.json").exists()
