from __future__ import annotations
from tqdm import tqdm

import time
from pathlib import Path
from typing import Any, Callable

from experiments.calibration import calibrate_threshold_from_scores
from experiments.config import OperatingPoint, ParetoExperimentConfig
from utils.detection import build_detector, detection_record, load_tokenizer
from utils.io import append_jsonl, iter_jsonl, read_json, write_json
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
from watermark.execution import BatchExecutionConfig


def _runtime_config(
    config: ParetoExperimentConfig,
    calibrated_threshold: float | None,
) -> ExperimentConfig:
    if config.presence_test == "z_score" and calibrated_threshold is None:
        raise ValueError("z_score detection requires a calibrated Z threshold")
    z_threshold = None if calibrated_threshold is None else float(calibrated_threshold)
    return ExperimentConfig(
        model=ModelConfig(
            path=config.model_path,
            device=config.device,
            dtype=config.dtype,
            local_files_only=config.local_files_only,
        ),
        dataset=DatasetConfig(kind="c4"),
        generation=GenerationConfig(
            max_new_tokens=config.exact_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            global_seed=config.global_seed,
            message_seed=config.message_seed,
        ),
        watermark=WatermarkConfig(
            secret_key=config.secret_key,
            context_width=config.context_width,
            presence_mode="soft",
            delta_presence=0.0,
            delta_payload=0.0,
            prf_mode=config.prf_mode,
            allocation_mode=config.allocation_mode,
            candidate_top_k=config.candidate_top_k,
            seeding_scheme=config.seeding_scheme,
            partition_engine=config.partition_engine,
        ),
        ecc=ECCConfig(n=config.ecc_n, k=config.ecc_k, t=config.ecc_t),
        detection=DetectionConfig(
            presence_test=config.presence_test,
            threshold_mode="theoretical" if config.presence_test == "exact_binomial" else "calibrated",
            target_fpr=config.target_fpr,
            calibrated_threshold=z_threshold,
            counting_mode=config.counting_mode,
        ),
        decoding=DecodingConfig(
            primary_policy="tie_zero",
            max_erasure_assignments=config.max_erasure_assignments,
            evaluate_all_policies=True,
        ),
        output=OutputConfig(root=str(config.experiment_dir), run_id="experiment"),
        execution=BatchExecutionConfig(
            generation_batch_size=config.generation_batch_size,
            detection_batch_size=config.detection_batch_size,
            detection_workers=config.detection_workers,
        ),
    )


def _timed(function: Callable[..., Any], *args: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - started


def _prepare(path: Path, *, resume: bool, overwrite: bool) -> set[tuple[str, str, str]]:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if overwrite and path.exists():
        path.unlink()
    if path.exists() and path.stat().st_size and not resume and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Use resume=True or overwrite=True.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    if not resume:
        return set()
    return {
        (str(row["sample_id"]), str(row["text_class"]), str(row["input_mode"]))
        for row in iter_jsonl(path)
        if row.get("status", "completed") == "completed"
    }


def _detect_modes(detector: Any, prompt_ids: list[int], token_ids: list[int]):
    known, known_seconds = _timed(detector.detect_continuation, prompt_ids, token_ids)
    blind, blind_seconds = _timed(detector.detect_token_ids, token_ids)
    return ((known, known_seconds), (blind, blind_seconds))


from tqdm.auto import tqdm


def run_shared_negative_detection(
    config,
    *,
    tokenizer=None,
    vocab_size=None,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    if tokenizer is None or vocab_size is None:
        tokenizer, vocab_size = load_tokenizer(
            config.model_path,
            local_files_only=config.local_files_only,
        )

    runtime = _runtime_config(
        config,
        calibrated_threshold=0.0 if config.presence_test == "z_score" else None,
    )
    detector = build_detector(runtime, tokenizer, int(vocab_size))

    output = (
        config.experiment_dir
        / "shared"
        / "negative_detections.jsonl"
    )

    completed = _prepare(
        output,
        resume=resume,
        overwrite=overwrite,
    )

    processed = 0
    skipped = 0

    rows = list(
        iter_jsonl(
            config.experiment_dir
            / "shared"
            / "baseline.jsonl"
        )
    )

    # 每条样本：
    # 2 类负文本 × 2 种检测模式 = 4 个结果
    total_results = len(rows) * 4

    with tqdm(
        total=total_results,
        desc="Shared negative detection",
        unit="result",
        dynamic_ncols=True,
    ) as progress:
        for row in rows:
            sample_id = str(row["sample_id"])
            prompt_ids = [
                int(value)
                for value in row["prompt_token_ids"]
            ]

            for text_class, field in (
                (
                    "unwatermarked",
                    "unwatermarked_token_ids",
                ),
                (
                    "natural",
                    "natural_token_ids",
                ),
            ):
                token_ids = [
                    int(value)
                    for value in row[field]
                ]

                for result, elapsed in _detect_modes(
                    detector,
                    prompt_ids,
                    token_ids,
                ):
                    key = (
                        sample_id,
                        text_class,
                        result.input_mode,
                    )

                    if key in completed:
                        skipped += 1
                        progress.update(1)
                        continue

                    record = detection_record(
                        config=runtime,
                        detector=detector,
                        sample_id=sample_id,
                        text_class=text_class,
                        result=result,
                        elapsed=elapsed,
                        message_bits=None,
                        encoded_bits=None,
                    )

                    record["split"] = row["split"]
                    append_jsonl(output, record)

                    # 防止同一次运行中出现重复 key
                    completed.add(key)
                    processed += 1
                    progress.update(1)

    return {
        "processed": processed,
        "skipped": skipped,
    }


def calibrate_shared_z_threshold(
    config: ParetoExperimentConfig,
) -> dict[str, Any] | None:
    if config.presence_test != "z_score":
        return None
    records = [
        row
        for row in iter_jsonl(config.experiment_dir / "shared" / "negative_detections.jsonl")
        if row.get("split") == "calibration"
        and row.get("input_mode") == "known_boundary"
        and row.get("text_class") in {"unwatermarked", "natural"}
    ]
    result = calibrate_threshold_from_scores(
        [float(row["z_score"]) for row in records],
        target_fpr=config.target_fpr,
    )
    result["negative_sources"] = ["unwatermarked", "natural"]
    result["input_mode"] = "known_boundary"
    result["score_type"] = "z_score"
    write_json(config.experiment_dir / "shared" / "z_calibration.json", result)
    return result


def load_shared_z_threshold(config: ParetoExperimentConfig) -> float | None:
    if config.presence_test != "z_score":
        return None
    calibration = read_json(config.experiment_dir / "shared" / "z_calibration.json")
    return float(calibration["calibrated_threshold"])


def run_operating_point_detection(
    config: ParetoExperimentConfig,
    point: OperatingPoint,
    *,
    calibrated_threshold: float | None,
    tokenizer: Any | None = None,
    vocab_size: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    if tokenizer is None or vocab_size is None:
        tokenizer, vocab_size = load_tokenizer(config.model_path, local_files_only=config.local_files_only)
    runtime = _runtime_config(config, calibrated_threshold)
    detector = build_detector(runtime, tokenizer, int(vocab_size))
    run_dir = config.experiment_dir / "runs" / point.point_id
    output = run_dir / "watermarked_detections.jsonl"
    completed = _prepare(output, resume=resume, overwrite=overwrite)
    processed = skipped = 0
    for row in iter_jsonl(run_dir / "watermarked.jsonl"):
        sample_id = str(row["sample_id"])
        prompt_ids = [int(value) for value in row["prompt_token_ids"]]
        token_ids = [int(value) for value in row["watermarked_token_ids"]]
        for result, elapsed in _detect_modes(detector, prompt_ids, token_ids):
            key = (sample_id, "watermarked", result.input_mode)
            if key in completed:
                skipped += 1
                continue
            record = detection_record(
                config=runtime,
                detector=detector,
                sample_id=sample_id,
                text_class="watermarked",
                result=result,
                elapsed=elapsed,
                message_bits=str(row["message_bits"]),
                encoded_bits=str(row["encoded_bits"]),
            )
            record["split"] = "test"
            record["operating_point"] = point.to_dict()
            append_jsonl(output, record)
            processed += 1
    write_json(
        run_dir / "detection_config.json",
        {
            "z_calibrated_threshold": calibrated_threshold,
            "target_fpr": config.target_fpr,
            "presence_test": config.presence_test,
            "counting_mode": config.counting_mode,
            "primary_policy": "tie_zero",
        },
    )
    return {"processed": processed, "skipped": skipped}
