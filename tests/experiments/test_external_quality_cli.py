from experiments.evaluate_external_ppl import build_parser


def test_external_ppl_cli_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--experiment-dir",
            "outputs/experiments/exp",
            "--evaluator-model",
            "/models/opt-6.7b",
        ]
    )
    assert args.batch_size == 1
    assert args.device == "cuda"
    assert args.dtype == "float16"
    assert args.local_files_only is True
    assert args.only_point == []
