from experiments.arguments import experiment_parser
from experiments.run_capacity_fpr import build_parser as capacity_parser
from experiments.run_length_sweep import build_parser as length_parser
from utils.arguments import generation_parser


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
