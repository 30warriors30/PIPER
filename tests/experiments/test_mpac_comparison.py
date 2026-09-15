import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from experiments.mpac_comparison import (
    PerRowMPACLogitsProcessor,
    build_parser,
    embedded_message_length,
    embedded_payload_bits,
    empirical_threshold,
    evaluate_payload_prediction,
    generate_mpac_watermarked_batch,
    import_mpac,
    load_manifest_rows,
    mpac_digits_from_bits,
    resolve_input_path,
    resolve_output_path,
    summarize_records,
    validate_runtime_args,
)
from watermark.ecc import BCHCodec


def test_load_manifest_rows_filters_split_and_limit(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"sample_id":"0","split":"calibration"}\n'
        '{"sample_id":"1","split":"test"}\n'
        '{"sample_id":"2","split":"test"}\n'
    )

    rows = load_manifest_rows(manifest, split="test", limit=1)

    assert [row["sample_id"] for row in rows] == ["1"]


def test_empirical_threshold_respects_target_fpr_with_ties():
    threshold = empirical_threshold([0.2, 1.0, 1.0, 3.0], target_fpr=0.25)

    assert threshold == 3.0


def test_empirical_threshold_moves_above_max_when_ties_would_exceed_fpr():
    threshold = empirical_threshold([0.2, 3.0, 3.0, 3.0], target_fpr=0.25)

    assert threshold > 3.0
    assert math.isfinite(threshold)


def test_summarize_records_reports_fpr_tpr_and_payload_metrics():
    summary = summarize_records(
        [
            {"text_class": "watermarked", "z_score": 3.0, "message_recovered": True, "original_bit_acc": 1.0, "embedded_bit_acc": 1.0},
            {"text_class": "watermarked", "z_score": 1.0, "message_recovered": True, "original_bit_acc": 1.0, "embedded_bit_acc": 1.0},
            {"text_class": "watermarked", "z_score": 3.0, "message_recovered": False, "original_bit_acc": 0.75, "embedded_bit_acc": 0.5},
            {"text_class": "unwatermarked", "z_score": 2.5},
            {"text_class": "natural", "z_score": 0.0},
        ],
        threshold=2.0,
        target_fpr=0.01,
    )

    assert summary["tpr"] == 2 / 3
    assert summary["model_fpr"] == 1.0
    assert summary["natural_fpr"] == 0.0
    assert summary["combined_fpr"] == 0.5
    assert summary["exact_message_recovery"] == 2 / 3
    assert summary["end_to_end_exact_recovery"] == 1 / 3
    assert summary["mean_bit_accuracy"] == pytest.approx(11 / 12)
    assert summary["mean_embedded_bit_accuracy"] == pytest.approx(5 / 6)


def test_build_parser_defaults_match_mpac_baseline():
    args = build_parser().parse_args(
        [
            "--experiment-dir",
            "outputs/experiments/opt13b_pareto_49x200",
            "--mb-repo",
            "/tmp/mb-lm-watermarking",
            "--output-dir",
            "outputs/experiments/opt13b_pareto_49x200/comparisons/mpac_t200_b8_m2",
            "--model-path",
            "/data/yanlu/BREW/models/facebook/opt-1.3b",
        ]
    )

    assert args.exact_tokens == 200
    assert args.message_length == 8
    assert args.base == 4
    assert args.gamma == 0.25
    assert args.delta == 2.0
    assert args.seeding_scheme == "lefthash"
    assert args.target_fpr == 0.01
    assert args.mpac_ecc == "none"
    assert args.generation_batch_size == 1
    assert embedded_message_length(args) == 8


def test_validate_runtime_args_rejects_nonpositive_generation_batch_size():
    args = build_parser().parse_args(
        [
            "--experiment-dir",
            "outputs/experiments/opt13b_pareto_49x200",
            "--mb-repo",
            "/tmp/mb-lm-watermarking",
            "--output-dir",
            "outputs/baselines/mpac",
            "--model-path",
            "/data/yanlu/BREW/models/facebook/opt-1.3b",
            "--device",
            "cpu",
            "--generation-batch-size",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="generation-batch-size must be positive"):
        validate_runtime_args(args)


def test_per_row_processor_keeps_payload_state_separate():
    class RowProcessor:
        def __init__(self, offset):
            self.offset = offset
            self.seen = []

        def __call__(self, input_ids, scores):
            self.seen.append(input_ids.tolist())
            return scores + self.offset

    first = RowProcessor(10.0)
    second = RowProcessor(20.0)
    processor = PerRowMPACLogitsProcessor([first, second])

    output = processor(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        torch.zeros((2, 3), dtype=torch.float32),
    )

    assert output.tolist() == [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]
    assert first.seen == [[[1, 2]]]
    assert second.seen == [[[3, 4]]]


def test_generate_mpac_watermarked_batch_batches_model_and_preserves_rows():
    from types import SimpleNamespace

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(vocab_size=12, is_encoder_decoder=False)
            self.forward_batch_sizes = []

        def __call__(
            self,
            *,
            input_ids,
            attention_mask,
            past_key_values=None,
            use_cache,
            return_dict,
        ):
            del attention_mask, use_cache, return_dict
            self.forward_batch_sizes.append(int(input_ids.shape[0]))
            batch, length = input_ids.shape
            logits = torch.arange(12, dtype=torch.float32).repeat(batch, length, 1)
            return SimpleNamespace(logits=logits, past_key_values=(past_key_values, length))

    class FakeTokenizer:
        vocab_size = 10
        pad_token_id = 0
        eos_token_id = 1

        def get_vocab(self):
            return {str(index): index for index in range(self.vocab_size)}

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return " ".join(str(int(value)) for value in ids)

    class FakeMPACProcessor:
        def __init__(self, **kwargs):
            del kwargs
            self.message = None
            self.positions = []
            self.position_increment = 0

        def set_message(self, message):
            self.message = message

        def __call__(self, input_ids, scores):
            del input_ids
            self.positions.append(self.message[-1])
            return scores

        def flush_position(self):
            positions = "".join(self.positions)
            self.positions = []
            return [positions]

    args = build_parser().parse_args(
        [
            "--experiment-dir",
            "experiment",
            "--mb-repo",
            "mpac",
            "--output-dir",
            "output",
            "--model-path",
            "model",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--exact-tokens",
            "3",
            "--top-p",
            "1.0",
            "--generation-batch-size",
            "2",
        ]
    )
    rows = [
        {
            "sample_id": "a",
            "split": "test",
            "message_bits": "00000000",
            "generation_seed": 10,
            "prompt_token_ids": [2, 3],
        },
        {
            "sample_id": "b",
            "split": "test",
            "message_bits": "00000001",
            "generation_seed": 20,
            "prompt_token_ids": [4, 5, 6],
        },
    ]
    model = FakeModel()

    generated = generate_mpac_watermarked_batch(
        rows,
        args=args,
        model=model,
        tokenizer=FakeTokenizer(),
        device=torch.device("cpu"),
        processor_cls=FakeMPACProcessor,
    )

    assert [row["sample_id"] for row in generated] == ["a", "b"]
    assert [row["embedded_message_bits"] for row in generated] == [
        "00000000",
        "00000001",
    ]
    assert [row["sampled_positions"] for row in generated] == ["000", "111"]
    assert all(len(row["token_ids"]) == 3 for row in generated)
    assert all(row["generation_batch_size"] == 2 for row in generated)
    assert model.forward_batch_sizes == [2, 2, 2]


def test_root_wrapper_exposes_mpac_comparison_module():
    repo_root = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        [sys.executable, "-m", "experiments.mpac_comparison", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--mpac-ecc {none,bch23}" in completed.stdout
    assert "--generation-batch-size GENERATION_BATCH_SIZE" in completed.stdout


def test_root_wrapper_does_not_shadow_project_package_imports():
    repo_root = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from dual_layer_watermark_fixed_length_error_erasure.experiments.mpac_comparison "
                "import build_parser; print(build_parser().description)"
            ),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Run MPAC" in completed.stdout


def test_validate_runtime_args_rejects_unavailable_cuda(monkeypatch):
    args = build_parser().parse_args(
        [
            "--experiment-dir",
            "outputs/experiments/opt13b_pareto_49x200",
            "--mb-repo",
            "/tmp/mb-lm-watermarking",
            "--output-dir",
            "outputs/experiments/opt13b_pareto_49x200/comparisons/mpac_t200_b8_m2",
            "--model-path",
            "/data/yanlu/BREW/models/facebook/opt-1.3b",
            "--device",
            "cuda",
        ]
    )
    monkeypatch.setattr("experiments.mpac_comparison.torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA device requested"):
        validate_runtime_args(args)


def test_relative_paths_resolve_under_project_when_called_from_repo_root(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    project_dir = repo_root / "dual_layer_watermark_fixed_length_error_erasure"
    monkeypatch.chdir(repo_root)

    experiment_dir = resolve_input_path("outputs/experiments/opt13b_pareto_49x200")
    output_dir = resolve_output_path("outputs/experiments/opt13b_pareto_49x200/comparisons/mpac_t200_b8_m2")

    assert experiment_dir == project_dir / "outputs/experiments/opt13b_pareto_49x200"
    assert output_dir == project_dir / "outputs/experiments/opt13b_pareto_49x200/comparisons/mpac_t200_b8_m2"


def test_bch23_mode_embeds_bch_codeword_not_raw_message():
    args = build_parser().parse_args(
        [
            "--experiment-dir",
            "outputs/experiments/opt13b_pareto_49x200",
            "--mb-repo",
            "/tmp/mb-lm-watermarking",
            "--output-dir",
            "outputs/experiments/opt13b_pareto_49x200/comparisons/mpac_t200_b8_m2_bch23",
            "--model-path",
            "/data/yanlu/BREW/models/facebook/opt-1.3b",
            "--mpac-ecc",
            "bch23",
        ]
    )
    expected = "".join(map(str, BCHCodec(23, 8, 3).encode([0, 0, 1, 1, 0, 0, 0, 1])))

    assert args.message_length == 8
    assert embedded_message_length(args) == 23
    assert embedded_payload_bits("00110001", args) == expected


def test_import_mpac_patches_short_message_candidate_overflow():
    mb_repo = Path("/tmp/mb-lm-watermarking")
    if not mb_repo.exists():
        pytest.skip("MPAC repository is not checked out")
    _, detector_cls = import_mpac(mb_repo)
    detector = detector_cls(
        vocab=list(range(128)),
        gamma=0.25,
        base=4,
        seeding_scheme="lefthash",
        message_length=8,
        code_length=8,
        use_position_prf=False,
        use_fixed_position=False,
        device="cpu",
        tokenizer=object(),
        z_threshold=0.0,
        normalizers=[],
    )
    position_cnt = {pos: 1 for pos in range(1, detector.converted_msg_length + 1)}
    green_cnt_by_position = {
        pos: [1, 0, 0, 0] for pos in range(1, detector.converted_msg_length + 1)
    }
    p_val_per_position = [1.0 for _ in range(detector.converted_msg_length)]

    decoded, random_decoded, _ = detector._predict_message(
        position_cnt,
        green_cnt_by_position,
        p_val_per_position,
    )

    assert len(decoded) >= 1
    assert len(random_decoded) == detector.converted_msg_length + 1


def test_bch23_prediction_decodes_codeword_before_recovery_metric():
    args = build_parser().parse_args(
        [
            "--experiment-dir",
            "outputs/experiments/opt13b_pareto_49x200",
            "--mb-repo",
            "/tmp/mb-lm-watermarking",
            "--output-dir",
            "outputs/experiments/opt13b_pareto_49x200/comparisons/mpac_t200_b8_m2_bch23",
            "--model-path",
            "/data/yanlu/BREW/models/facebook/opt-1.3b",
            "--mpac-ecc",
            "bch23",
        ]
    )
    embedded = embedded_payload_bits("00110001", args)
    corrupted = list(embedded)
    corrupted[0] = "1" if corrupted[0] == "0" else "0"
    pred_message = mpac_digits_from_bits("".join(corrupted), base=4)

    result = evaluate_payload_prediction(
        pred_message=pred_message,
        original_message_bits="00110001",
        embedded_message_bits=embedded,
        args=args,
    )

    assert result["message_recovered"] is True
    assert result["bch_decode_status"] == "decoded"
    assert result["original_bit_acc"] == 1.0
    assert result["embedded_bit_acc"] < 1.0
