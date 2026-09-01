import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.external_quality import score_external_evaluator
from utils.io import append_jsonl, write_json


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


def _build_experiment(root: Path) -> Path:
    experiment_dir = root / "exp"
    write_json(
        experiment_dir / "experiment.json",
        {
            "schema_version": 1,
            "experiment_type": "quality_detection_pareto",
            "config": {"model_path": "/models/opt-1.3b"},
        },
    )
    append_jsonl(
        experiment_dir / "shared" / "baseline.jsonl",
        {
            "sample_id": "a",
            "split": "test",
            "prompt_token_ids": [1, 2],
            "unwatermarked_token_ids": [3, 4, 5],
            "natural_token_ids": [4, 4, 4],
        },
    )
    append_jsonl(
        experiment_dir / "runs" / "soft_p0_m2" / "watermarked.jsonl",
        {
            "sample_id": "a",
            "split": "test",
            "prompt_token_ids": [1, 2],
            "watermarked_token_ids": [5, 6, 7],
        },
    )
    write_json(
        experiment_dir / "runs" / "soft_p0_m2" / "metrics.json",
        {
            "operating_point": {
                "point_id": "soft_p0_m2",
                "presence_mode": "soft",
                "delta_presence": 0.0,
                "delta_payload": 2.0,
            },
            "presence": {"tpr": 0.99, "model_fpr": 0.01, "natural_fpr": 0.01, "auc": 1.0},
            "payload": {"error_erasure_exact_message_recovery": 0.9},
        },
    )
    (experiment_dir / "sweep_results.json").write_text(
        json.dumps(
            [
                {
                    "point_id": "soft_p0_m2",
                    "presence_mode": "soft",
                    "delta_presence": 0.0,
                    "delta_payload": 2.0,
                    "tpr": 0.99,
                    "model_fpr": 0.01,
                    "natural_fpr": 0.01,
                    "auc": 1.0,
                    "error_erasure_exact_recovery": 0.9,
                    "relative_ppl_increase": 0.4,
                }
            ]
        ),
        encoding="utf-8",
    )
    return experiment_dir


def test_external_pipeline_scores_existing_text_and_preserves_original_quality(tmp_path: Path) -> None:
    experiment_dir = _build_experiment(tmp_path)
    original_sweep = (experiment_dir / "sweep_results.json").read_text(encoding="utf-8")
    tokenizer = FakeTokenizer()

    result = score_external_evaluator(
        experiment_dir=experiment_dir,
        evaluator_model_path="/models/opt-6.7b",
        evaluator_name="opt-6.7b",
        batch_size=2,
        model=UniformCausalModel(),
        generation_tokenizer=tokenizer,
        evaluator_tokenizer=tokenizer,
        device=torch.device("cpu"),
        resume=False,
        overwrite=False,
    )

    output_dir = experiment_dir / "external_quality" / "opt-6_7b"
    assert result["point_count"] == 1
    assert result["shared_scored"] == 2
    assert result["watermarked_scored"] == 1
    assert (output_dir / "shared" / "quality.jsonl").exists()
    assert (output_dir / "runs" / "soft_p0_m2" / "quality.jsonl").exists()
    assert (output_dir / "sweep_results.csv").exists()
    assert (output_dir / "figures" / "quality_vs_tpr.png").exists()
    assert (output_dir / "figures" / "quality_vs_recovery.png").exists()
    assert (experiment_dir / "sweep_results.json").read_text(encoding="utf-8") == original_sweep

    with (output_dir / "sweep_results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert math.isclose(float(rows[0]["external_paired_delta_nll"]), 0.0, abs_tol=1e-8)
    assert math.isclose(float(rows[0]["external_ppl_ratio"]), 1.0, abs_tol=1e-8)
    assert math.isclose(float(rows[0]["external_relative_ppl_increase"]), 0.0, abs_tol=1e-8)


def test_external_pipeline_resume_does_not_duplicate_scores(tmp_path: Path) -> None:
    experiment_dir = _build_experiment(tmp_path)
    tokenizer = FakeTokenizer()
    kwargs = dict(
        experiment_dir=experiment_dir,
        evaluator_model_path="/models/opt-13b",
        evaluator_name="opt-13b",
        batch_size=2,
        model=UniformCausalModel(),
        generation_tokenizer=tokenizer,
        evaluator_tokenizer=tokenizer,
        device=torch.device("cpu"),
    )

    first = score_external_evaluator(**kwargs, resume=False, overwrite=False)
    second = score_external_evaluator(**kwargs, resume=True, overwrite=False)

    output_dir = experiment_dir / "external_quality" / "opt-13b"
    shared_lines = (output_dir / "shared" / "quality.jsonl").read_text(encoding="utf-8").splitlines()
    point_lines = (output_dir / "runs" / "soft_p0_m2" / "quality.jsonl").read_text(encoding="utf-8").splitlines()
    assert first["shared_scored"] == 2
    assert first["watermarked_scored"] == 1
    assert second["shared_scored"] == 0
    assert second["watermarked_scored"] == 0
    assert len(shared_lines) == 2
    assert len(point_lines) == 1
