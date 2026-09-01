from utils.arguments import generation_config_from_args, generation_parser


def test_generation_parser_builds_config() -> None:
    args = generation_parser().parse_args([
        "--model-path", "/tmp/model", "--secret-key", "secret", "--run-id", "test",
        "--ecc-n", "23", "--ecc-k", "8", "--ecc-t", "3",
    ])
    config = generation_config_from_args(args)
    assert config.ecc.k == 8
    assert config.dataset.kind == "c4"
    assert config.output.run_id == "test"
    assert config.watermark.presence_mode == "soft"
    assert config.watermark.delta_presence == 0.0
    assert config.watermark.delta_payload == 2.0
    assert config.detection.presence_test == "exact_binomial"
    assert config.detection.counting_mode == "unique_context"
    assert config.decoding.primary_policy == "tie_zero"
    assert config.decoding.max_erasure_assignments == 64
