from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from utils.io import RunStore, append_jsonl, iter_jsonl, read_json, write_json
from utils.logging import configure_logging
from utils.model import excluded_token_ids
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
from watermark.detector import DualLayerDetector
from watermark.ecc import BCHCodec
from watermark.execution import BatchExecutionConfig
from watermark.result_types import DetectionResult


def load_tokenizer(model_path: str, *, local_files_only: bool = True) -> Any:
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
    vocab_size = int(getattr(config, "vocab_size", len(tokenizer)))
    return tokenizer, vocab_size


def build_detector(config: ExperimentConfig, tokenizer: Any, vocab_size: int) -> DualLayerDetector:
    codec = BCHCodec(config.ecc.n, config.ecc.k, config.ecc.t)
    return DualLayerDetector(
        secret_key=config.watermark.secret_key.encode("utf-8"),
        context_width=config.watermark.context_width,
        vocab_size=vocab_size,
        excluded_token_ids=excluded_token_ids(
            tokenizer,
            exclude_special_tokens=config.watermark.exclude_special_tokens,
            exclude_eos=config.watermark.exclude_eos,
        ),
        prf_mode=config.watermark.prf_mode,
        ecc_codec=codec,
        presence_test=config.detection.presence_test,
        threshold_mode=config.detection.threshold_mode,
        target_fpr=config.detection.target_fpr,
        fixed_z_threshold=config.detection.fixed_threshold,
        calibrated_threshold=config.detection.calibrated_threshold,
        primary_counting_mode=config.detection.counting_mode,
        unique_ngram_width=config.detection.unique_ngram_width,
        min_tokens_per_code_bit=config.decoding.min_tokens_per_code_bit,
        hard_fill_value=config.decoding.hard_fill_value,
        primary_policy=config.decoding.primary_policy,
        max_erasure_assignments=config.decoding.max_erasure_assignments,
        evaluate_all_policies=config.decoding.evaluate_all_policies,
        seeding_scheme=config.watermark.seeding_scheme,
        partition_engine=config.watermark.partition_engine,
    )


def _selected_decode(config: ExperimentConfig, result: DetectionResult):
    if config.decoding.primary_policy == "tie_zero":
        return result.tie_zero_decode
    if config.decoding.primary_policy == "strict":
        return result.strict_decode
    if config.decoding.primary_policy == "hard_fill":
        return result.hard_fill_decode
    return result.error_erasure_decode


def detection_record(
    *,
    config: ExperimentConfig,
    detector: DualLayerDetector,
    sample_id: str,
    text_class: str,
    result: DetectionResult,
    elapsed: float,
    message_bits: str | None,
    encoded_bits: str | None,
) -> dict[str, Any]:
    primary = result.counting[result.primary_counting_mode]
    selected = _selected_decode(config, result)
    assert selected is not None
    codeword = selected.codeword_bits
    if (
        codeword is None
        and selected.status != "not_evaluated"
        and config.decoding.primary_policy in {"tie_zero", "strict", "hard_fill"}
    ):
        codeword = detector.recover_codeword(primary, policy=config.decoding.primary_policy)
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "status": "completed",
        "text_class": text_class,
        "input_mode": result.input_mode,
        "num_scored_tokens": primary.scored_tokens,
        "upper_count": primary.upper_hits,
        "z_score": primary.z_score,
        "binomial_p_value": primary.exact_p_value,
        "null_probability": primary.null_probability,
        "presence_test": getattr(result, "presence_test", "z_score"),
        "alpha": getattr(result, "alpha", config.detection.target_fpr if hasattr(config, "detection") else 0.01),
        "threshold": result.threshold,
        "detected": result.detected,
        "decoder_invoked": getattr(result, "decoder_invoked", selected.status != "not_evaluated"),
        "message_bits": message_bits,
        "encoded_bits": encoded_bits,
        "recovered_codeword_bits": None if codeword is None else "".join(map(str, codeword)),
        "decoded_message": None if result.gated_message_bits is None else "".join(map(str, result.gated_message_bits)),
        "ungated_decoded_message": None if selected.message_bits is None else "".join(map(str, selected.message_bits)),
        "decode_status": selected.status,
        "primary_decoding_policy": config.decoding.primary_policy,
        "tie_zero": None
        if getattr(result, "tie_zero_decode", None) is None
        else asdict(result.tie_zero_decode),
        "strict": asdict(result.strict_decode),
        "hard_fill": asdict(result.hard_fill_decode),
        "error_erasure": asdict(result.error_erasure_decode),
        "code_bit_counts_0": list(primary.n0),
        "code_bit_counts_1": list(primary.n1),
        "code_bit_margins": [one - zero for zero, one in zip(primary.n0, primary.n1, strict=True)],
        "erasure_positions": list(primary.erasures),
        "counting": {mode: asdict(value) for mode, value in result.counting.items()},
        "detection_time_seconds": elapsed,
    }


def _detect_timed(function: Any, *args: Any) -> tuple[DetectionResult, float]:
    started = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - started


def _updated_detection_config(
    config: ExperimentConfig,
    *,
    presence_test: str,
    threshold_mode: str,
    target_fpr: float,
    fixed_threshold: float,
    calibrated_threshold: float | None,
    counting_mode: str,
    unique_ngram_width: int,
    primary_policy: str,
    min_tokens_per_code_bit: int,
    hard_fill_value: int | str,
    max_erasure_assignments: int,
    execution: BatchExecutionConfig | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        model=config.model,
        dataset=config.dataset,
        generation=config.generation,
        watermark=config.watermark,
        ecc=config.ecc,
        detection=DetectionConfig(
            presence_test=presence_test,
            threshold_mode=threshold_mode,
            target_fpr=target_fpr,
            fixed_threshold=fixed_threshold,
            calibrated_threshold=calibrated_threshold,
            counting_mode=counting_mode,
            unique_ngram_width=unique_ngram_width,
        ),
        decoding=DecodingConfig(
            primary_policy=primary_policy,
            min_tokens_per_code_bit=min_tokens_per_code_bit,
            hard_fill_value=hard_fill_value,
            max_erasure_assignments=max_erasure_assignments,
            evaluate_all_policies=True,
        ),
        output=config.output,
        execution=config.execution if execution is None else execution,
        schema_version=config.schema_version,
    )


def run_samples_detection(
    run_dir: Path,
    *,
    presence_test: str = "exact_binomial",
    threshold_mode: str = "theoretical",
    target_fpr: float = 0.01,
    fixed_threshold: float = 2.326347874,
    calibrated_threshold: float | None = None,
    counting_mode: str = "unique_context",
    unique_ngram_width: int = 4,
    primary_policy: str = "tie_zero",
    min_tokens_per_code_bit: int = 1,
    hard_fill_value: int | str = 0,
    max_erasure_assignments: int = 64,
    output_file: str = "detections.jsonl",
    resume: bool = False,
    overwrite: bool = False,
    execution: BatchExecutionConfig | None = None,
) -> dict[str, Any]:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    experiment = read_json(run_dir / "experiment.json")
    config = ExperimentConfig.from_dict(experiment)
    config = _updated_detection_config(
        config,
        presence_test=presence_test,
        threshold_mode=threshold_mode,
        target_fpr=target_fpr,
        fixed_threshold=fixed_threshold,
        calibrated_threshold=calibrated_threshold,
        counting_mode=counting_mode,
        unique_ngram_width=unique_ngram_width,
        primary_policy=primary_policy,
        min_tokens_per_code_bit=min_tokens_per_code_bit,
        hard_fill_value=hard_fill_value,
        max_erasure_assignments=max_erasure_assignments,
        execution=execution,
    )
    tokenizer, vocab_size = load_tokenizer(config.model.path, local_files_only=config.model.local_files_only)
    detector = build_detector(config, tokenizer, vocab_size)
    output_path = run_dir / output_file
    if output_path.exists() and overwrite:
        output_path.unlink()
    if output_path.exists() and output_path.stat().st_size and not resume and not overwrite:
        raise FileExistsError(f"Detection output exists: {output_path}. Use --resume or --overwrite.")
    output_path.touch(exist_ok=True)
    store = RunStore(run_dir)
    completed = store.completed_detection_keys(output_path) if resume else set()
    logger = configure_logging(run_dir / "logs" / "detection.log", f"detection.{run_dir.name}")
    processed = skipped = failed = 0
    records = [record for record in iter_jsonl(run_dir / "samples.jsonl") if record.get("status") == "completed"]
    for sample in tqdm(records, desc="Detecting", unit="sample"):
        sample_id = str(sample["sample_id"])
        prompt_ids = tuple(int(value) for value in sample["prompt_token_ids"])
        for text_class in ("watermarked", "unwatermarked", "natural"):
            token_ids = tuple(int(value) for value in sample[text_class]["token_ids"])
            truth_message = sample.get("message_bits") if text_class == "watermarked" else None
            truth_encoded = sample.get("encoded_bits") if text_class == "watermarked" else None
            for mode in ("known_boundary", "blind_text"):
                key = (sample_id, text_class, mode)
                if key in completed:
                    skipped += 1
                    continue
                try:
                    if mode == "known_boundary":
                        result, elapsed = _detect_timed(detector.detect_continuation, prompt_ids, token_ids)
                    else:
                        result, elapsed = _detect_timed(detector.detect_token_ids, token_ids)
                    append_jsonl(
                        output_path,
                        detection_record(
                            config=config,
                            detector=detector,
                            sample_id=sample_id,
                            text_class=text_class,
                            result=result,
                            elapsed=elapsed,
                            message_bits=truth_message,
                            encoded_bits=truth_encoded,
                        ),
                    )
                    processed += 1
                except Exception as exc:
                    failed += 1
                    error = {
                        "schema_version": 1,
                        "sample_id": sample_id,
                        "text_class": text_class,
                        "input_mode": mode,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-20:]),
                    }
                    append_jsonl(output_path, error)
                    append_jsonl(run_dir / "errors" / "detection_errors.jsonl", error)
                    logger.exception("Detection failed for %s/%s/%s", sample_id, text_class, mode)
    write_json(
        run_dir / "detection_config.json",
        {
            "threshold_mode": threshold_mode,
            "presence_test": presence_test,
            "target_fpr": target_fpr,
            "fixed_threshold": fixed_threshold,
            "calibrated_threshold": calibrated_threshold,
            "counting_mode": counting_mode,
            "primary_policy": primary_policy,
            "max_erasure_assignments": max_erasure_assignments,
            "output_file": output_file,
        },
    )
    return {"output_file": str(output_path), "processed": processed, "skipped": skipped, "failed": failed}


def _external_config(
    *,
    model_path: str,
    secret_key: str,
    context_width: int,
    prf_mode: str,
    partition_mode: str,
    allocation_mode: str,
    seeding_scheme: str,
    partition_engine: str,
    ecc_n: int,
    ecc_k: int,
    ecc_t: int,
    presence_test: str,
    threshold_mode: str,
    target_fpr: float,
    fixed_threshold: float,
    calibrated_threshold: float | None,
    counting_mode: str,
    unique_ngram_width: int,
    primary_policy: str,
    min_tokens_per_code_bit: int,
    hard_fill_value: int | str,
    max_erasure_assignments: int,
    output_dir: Path,
    execution: BatchExecutionConfig,
) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(path=model_path, device="cpu", dtype="float32", local_files_only=True),
        dataset=DatasetConfig(kind="jsonl", path=None),
        generation=GenerationConfig(),
        watermark=WatermarkConfig(
            secret_key=secret_key,
            context_width=context_width,
            prf_mode=prf_mode,
            partition_mode=partition_mode,
            allocation_mode=allocation_mode,
            seeding_scheme=seeding_scheme,
            partition_engine=partition_engine,
        ),
        ecc=ECCConfig(n=ecc_n, k=ecc_k, t=ecc_t),
        detection=DetectionConfig(
            presence_test=presence_test,
            threshold_mode=threshold_mode,
            target_fpr=target_fpr,
            fixed_threshold=fixed_threshold,
            calibrated_threshold=calibrated_threshold,
            counting_mode=counting_mode,
            unique_ngram_width=unique_ngram_width,
        ),
        decoding=DecodingConfig(
            primary_policy=primary_policy,
            min_tokens_per_code_bit=min_tokens_per_code_bit,
            hard_fill_value=hard_fill_value,
            max_erasure_assignments=max_erasure_assignments,
        ),
        output=OutputConfig(root=str(output_dir.parent), run_id=output_dir.name),
        execution=execution,
    )


def run_text_detection(
    *,
    input_file: Path,
    output_dir: Path,
    output_file: str,
    text_field: str,
    prompt_field: str | None,
    continuation_field: str | None,
    model_path: str,
    secret_key: str,
    context_width: int,
    prf_mode: str,
    partition_mode: str,
    allocation_mode: str,
    seeding_scheme: str = "selfhash",
    partition_engine: str = "v2",
    ecc_n: int,
    ecc_k: int,
    ecc_t: int,
    presence_test: str = "exact_binomial",
    threshold_mode: str,
    target_fpr: float,
    fixed_threshold: float,
    calibrated_threshold: float | None,
    counting_mode: str,
    unique_ngram_width: int,
    primary_policy: str,
    min_tokens_per_code_bit: int,
    hard_fill_value: int | str,
    max_erasure_assignments: int = 64,
    overwrite: bool = False,
    execution: BatchExecutionConfig | None = None,
) -> dict[str, Any]:
    if not input_file.exists():
        raise FileNotFoundError(input_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_file
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite.")
    if output_path.exists():
        output_path.unlink()
    config = _external_config(
        model_path=model_path,
        secret_key=secret_key,
        context_width=context_width,
        prf_mode=prf_mode,
        partition_mode=partition_mode,
        allocation_mode=allocation_mode,
        seeding_scheme=seeding_scheme,
        partition_engine=partition_engine,
        ecc_n=ecc_n,
        ecc_k=ecc_k,
        ecc_t=ecc_t,
        presence_test=presence_test,
        threshold_mode=threshold_mode,
        target_fpr=target_fpr,
        fixed_threshold=fixed_threshold,
        calibrated_threshold=calibrated_threshold,
        counting_mode=counting_mode,
        unique_ngram_width=unique_ngram_width,
        primary_policy=primary_policy,
        min_tokens_per_code_bit=min_tokens_per_code_bit,
        hard_fill_value=hard_fill_value,
        max_erasure_assignments=max_erasure_assignments,
        output_dir=output_dir,
        execution=execution if execution is not None else BatchExecutionConfig(),
    )
    tokenizer, vocab_size = load_tokenizer(model_path)
    detector = build_detector(config, tokenizer, vocab_size)
    processed = failed = 0
    for index, row in enumerate(iter_jsonl(input_file)):
        sample_id = str(row.get("sample_id", index))
        try:
            if prompt_field and continuation_field:
                prompt_ids = tokenizer(str(row[prompt_field]), add_special_tokens=False)["input_ids"]
                continuation_ids = tokenizer(str(row[continuation_field]), add_special_tokens=False)["input_ids"]
                result, elapsed = _detect_timed(detector.detect_continuation, prompt_ids, continuation_ids)
            else:
                token_ids = tokenizer(str(row[text_field]), add_special_tokens=False)["input_ids"]
                result, elapsed = _detect_timed(detector.detect_token_ids, token_ids)
            append_jsonl(
                output_path,
                detection_record(
                    config=config,
                    detector=detector,
                    sample_id=sample_id,
                    text_class="external",
                    result=result,
                    elapsed=elapsed,
                    message_bits=None,
                    encoded_bits=None,
                ),
            )
            processed += 1
        except Exception as exc:
            failed += 1
            append_jsonl(
                output_path,
                {
                    "sample_id": sample_id,
                    "text_class": "external",
                    "input_mode": "known_boundary" if prompt_field and continuation_field else "blind_text",
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
    write_json(output_dir / "external_detection_config.json", config.to_dict())
    return {"output_file": str(output_path), "processed": processed, "failed": failed}
