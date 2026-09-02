from __future__ import annotations

from pathlib import Path

from utils.detection import (
    _updated_detection_config,
    build_detector,
    run_samples_detection,
)
from utils.io import append_jsonl, iter_jsonl, write_json
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
from watermark.execution import BatchExecutionConfig


class FakeTokenizer:
    all_special_ids = [0]
    eos_token_id = 2

    def __len__(self):
        return 32


def test_updated_detection_config_preserves_execution() -> None:
    execution = BatchExecutionConfig(
        generation_batch_size=3,
        detection_batch_size=5,
        detection_workers=2,
    )
    config = ExperimentConfig(
        model=ModelConfig(path="fake", device="cpu", dtype="float32"),
        dataset=DatasetConfig(),
        generation=GenerationConfig(),
        watermark=WatermarkConfig(secret_key="secret"),
        ecc=ECCConfig(),
        detection=DetectionConfig(),
        decoding=DecodingConfig(),
        output=OutputConfig(root="outputs", run_id="run"),
        execution=execution,
    )

    updated = _updated_detection_config(
        config,
        presence_test="exact_binomial",
        threshold_mode="theoretical",
        target_fpr=0.01,
        fixed_threshold=2.326347874,
        calibrated_threshold=None,
        counting_mode="unique_context",
        unique_ngram_width=4,
        primary_policy="tie_zero",
        min_tokens_per_code_bit=1,
        hard_fill_value=0,
        max_erasure_assignments=64,
    )

    assert updated.execution == execution


def test_samples_detection_writes_six_records(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = ExperimentConfig(
        model=ModelConfig(path="fake", device="cpu", dtype="float32"),
        dataset=DatasetConfig(),
        generation=GenerationConfig(),
        watermark=WatermarkConfig(secret_key="secret", context_width=4),
        ecc=ECCConfig(),
        detection=DetectionConfig(fixed_threshold=-100.0, threshold_mode="fixed"),
        decoding=DecodingConfig(primary_policy="error_erasure"),
        output=OutputConfig(root=str(tmp_path), run_id="run"),
    )
    write_json(run_dir / "experiment.json", config.to_dict())
    codec = BCHCodec(23, 8, 3)
    payload = (1, 0, 1, 1, 0, 0, 1, 0)
    encoded = codec.encode(payload)
    tokenizer = FakeTokenizer()
    detector = build_detector(config, tokenizer, 32)
    prompt = [1, 2, 3, 4]
    history = list(prompt)
    watermarked: list[int] = []
    for _ in range(92):
        partition, index = detector.partition_for_context(history[-4:])
        token = partition.bit1[0] if encoded[index] else partition.bit0[0]
        watermarked.append(token)
        history.append(token)
    append_jsonl(
        run_dir / "samples.jsonl",
        {
            "sample_id": "x",
            "status": "completed",
            "prompt_token_ids": prompt,
            "message_bits": "".join(map(str, payload)),
            "encoded_bits": "".join(map(str, encoded)),
            "watermarked": {"token_ids": watermarked},
            "unwatermarked": {"token_ids": [3, 4, 5, 6, 7, 8]},
            "natural": {"token_ids": [8, 7, 6, 5, 4, 3]},
        },
    )
    monkeypatch.setattr("utils.detection.load_tokenizer", lambda *args, **kwargs: (tokenizer, 32))
    result = run_samples_detection(run_dir, threshold_mode="fixed", fixed_threshold=-100.0, primary_policy="error_erasure")
    assert result["processed"] == 6
    records = list(iter_jsonl(run_dir / "detections.jsonl"))
    assert len(records) == 6
    assert all("error_erasure" in record for record in records)
    assert all(record["primary_decoding_policy"] == "error_erasure" for record in records)
    assert {(r["text_class"], r["input_mode"]) for r in records} == {
        (text_class, mode)
        for text_class in ("watermarked", "unwatermarked", "natural")
        for mode in ("known_boundary", "blind_text")
    }
