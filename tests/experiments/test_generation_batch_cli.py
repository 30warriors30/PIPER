from types import SimpleNamespace

import torch

from experiments.arguments import experiment_parser
from experiments.config import OperatingPoint, ParetoExperimentConfig
from experiments.generation import ExperimentGenerator
from experiments.run_capacity_fpr import build_parser as capacity_parser
from experiments.run_length_sweep import build_parser as length_parser
from utils.arguments import generation_parser


class _FakeTokenizer:
    all_special_ids = [0]
    eos_token_id = 1

    def __len__(self) -> int:
        return 12


def test_piper_generation_commands_default_to_gpu_batching() -> None:
    common = ["--model-path", "/tmp/model", "--secret-key", "secret"]
    pareto = experiment_parser().parse_args(common + ["--experiment-id", "test"])
    capacity = capacity_parser().parse_args(common)
    length = length_parser().parse_args(common)
    legacy = generation_parser().parse_args(common + ["--run-id", "test"])

    assert pareto.generation_batch_size == 16
    assert capacity.generation_batch_size == 16
    assert length.generation_batch_size == 16
    assert legacy.generation_batch_size == 16
    assert pareto.top_k == 50
    assert pareto.candidate_top_k == 50
    assert pareto.seeding_scheme == "selfhash"
    assert pareto.partition_engine == "v2"
    assert capacity.top_k == 50
    assert capacity.candidate_top_k == 50
    assert capacity.seeding_scheme == "selfhash"
    assert capacity.partition_engine == "v2"
    assert length.top_k == 50
    assert length.candidate_top_k == 50
    assert length.seeding_scheme == "selfhash"
    assert length.partition_engine == "v2"
    assert legacy.top_k == 50
    assert legacy.candidate_top_k == 50
    assert legacy.seeding_scheme == "selfhash"
    assert legacy.partition_engine == "v2"


def test_experiment_generator_threads_top_k_selfhash_configuration(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def fake_generate_exact_batch(**kwargs):
        captured.update(kwargs)
        return ()

    monkeypatch.setattr("experiments.generation.generate_exact_batch", fake_generate_exact_batch)
    config = ParetoExperimentConfig(
        model_path="model",
        secret_key="secret",
        output_root=tmp_path,
        experiment_id="test",
        top_k=50,
        candidate_top_k=50,
        seeding_scheme="selfhash",
        partition_engine="v2",
    )
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=12))
    tokenizer = _FakeTokenizer()
    generator = ExperimentGenerator(
        config,
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )

    generator.generate_watermarked_batch(
        ({"prompt_token_ids": [2, 3], "generation_seed": 7, "encoded_bits": "01"},),
        OperatingPoint("soft", 0.0, 2.0),
    )

    assert captured["top_k"] == 50
    processor = captured["processor"]
    assert processor.seeding_scheme == "selfhash"
    assert processor.partition_engine == "v2"
    assert processor.candidate_top_k == 50
