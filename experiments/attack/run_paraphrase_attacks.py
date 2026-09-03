from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm

from evaluation.metrics import evaluate_records
from utils.detection import build_detector, detection_record, load_tokenizer
from utils.io import plain, write_json

from experiments.attack.run_synonym_attacks import (
    DEFAULT_EXPERIMENT_DIR,
    INPUT_MODES,
    AttackExample,
    _prepare_output,
    _sample_seed,
    _timed_detect,
    _write_jsonl,
    build_runtime_config,
    load_attack_z_threshold,
    load_attack_examples,
    resolve_input_path,
    resolve_output_path,
    validate_input_modes,
    write_roc_points,
)

DEFAULT_PARAPHRASER_MODEL = (
    "/data/yanlu/.cache/huggingface/hub/models--tuner007--pegasus_paraphrase/"
    "snapshots/0159e2949ca73657a2f1329898f51b7bb53b9ab2"
)


@dataclass(frozen=True)
class ParaphraseResult:
    original_text: str
    attacked_text: str
    attacked_token_ids: list[int]
    original_token_count: int
    attacked_token_count: int
    token_length_delta: int
    compression_ratio: float | None
    paraphrase_seconds: float
    paraphraser_model: str
    generation: dict[str, Any]


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "model"


def default_output_dir(experiment_dir: Path, point_id: str, paraphraser_slug: str) -> Path:
    return experiment_dir / "attacks" / f"{point_id}_paraphrase_{_safe_slug(paraphraser_slug)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paraphrase rewrite attacks for the dual-layer watermark")
    parser.add_argument("--experiment-dir", default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--point-id", default="soft_p0_m2")
    parser.add_argument("--attack", default="paraphrase")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--headline-input-mode", choices=["known_boundary", "blind_text"], default="known_boundary")
    parser.add_argument("--input-modes", nargs="+", choices=list(INPUT_MODES), default=list(INPUT_MODES))
    parser.add_argument("--evaluate-all-policies", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--paraphraser-model", default=DEFAULT_PARAPHRASER_MODEL)
    parser.add_argument("--paraphraser-slug", default="pegasus")
    parser.add_argument("--paraphraser-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--paraphraser-batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=60)
    parser.add_argument("--max-output-tokens", type=int, default=80)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _bounded_token_limit(requested: int, model_limit: int | None, argument_name: str) -> int:
    if requested <= 0:
        raise ValueError(f"{argument_name} must be positive")
    if model_limit is None or model_limit <= 0:
        return int(requested)
    return min(int(requested), int(model_limit))


def _ints(values: Sequence[int]) -> list[int]:
    return [int(value) for value in values]


def make_paraphrase_result(
    text: str,
    attacked_text: str,
    tokenizer: Any,
    paraphrase_seconds: float,
    model_path: str,
    generation: Mapping[str, Any],
) -> ParaphraseResult:
    original_token_ids = _ints(tokenizer.encode(text, add_special_tokens=False))
    attacked_token_ids = _ints(tokenizer.encode(attacked_text, add_special_tokens=False))
    original_token_count = len(original_token_ids)
    attacked_token_count = len(attacked_token_ids)
    compression_ratio = None if original_token_count == 0 else attacked_token_count / original_token_count
    return ParaphraseResult(
        original_text=text,
        attacked_text=attacked_text,
        attacked_token_ids=attacked_token_ids,
        original_token_count=original_token_count,
        attacked_token_count=attacked_token_count,
        token_length_delta=attacked_token_count - original_token_count,
        compression_ratio=compression_ratio,
        paraphrase_seconds=float(paraphrase_seconds),
        paraphraser_model=str(model_path),
        generation=dict(generation),
    )


def _paraphrase_record(
    example: AttackExample,
    attack: str,
    result: ParaphraseResult,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sample_id": example.sample_id,
        "split": example.split,
        "text_class": example.text_class,
        "attack": attack,
        "attack_seed": seed,
        "original_text": result.original_text,
        "attacked_text": result.attacked_text,
        "original_token_ids": example.token_ids,
        "attacked_token_ids": result.attacked_token_ids,
        "original_token_count": result.original_token_count,
        "attacked_token_count": result.attacked_token_count,
        "token_length_delta": result.token_length_delta,
        "compression_ratio": result.compression_ratio,
        "paraphrase_seconds": result.paraphrase_seconds,
        "paraphraser_model": result.paraphraser_model,
        "generation": result.generation,
    }


def detect_paraphrased_example(
    example: AttackExample,
    result: ParaphraseResult,
    detector: Any,
    runtime_config: Any,
    attack: str,
    *,
    attack_seed: int,
    input_modes: Sequence[str] = INPUT_MODES,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    decode_payload = example.text_class == "watermarked"
    for input_mode in input_modes:
        if input_mode == "known_boundary":
            detection_result, elapsed = _timed_detect(
                detector.detect_continuation,
                example.prompt_token_ids,
                result.attacked_token_ids,
                decode_payload=decode_payload,
            )
        elif input_mode == "blind_text":
            detection_result, elapsed = _timed_detect(
                detector.detect_token_ids,
                result.attacked_token_ids,
                decode_payload=decode_payload,
            )
        else:
            raise ValueError(f"Unsupported input mode: {input_mode}")
        record = detection_record(
            config=runtime_config,
            detector=detector,
            sample_id=example.sample_id,
            text_class=example.text_class,
            result=detection_result,
            elapsed=elapsed,
            message_bits=example.message_bits,
            encoded_bits=example.encoded_bits,
        )
        record.update(
            {
                "split": example.split,
                "attack": attack,
                "attack_seed": attack_seed,
                "paraphrase_seconds": result.paraphrase_seconds,
                "attack_seconds": result.paraphrase_seconds,
                "original_token_count": result.original_token_count,
                "attacked_token_count": result.attacked_token_count,
                "token_length_delta": result.token_length_delta,
                "compression_ratio": result.compression_ratio,
                "paraphraser_model": result.paraphraser_model,
            }
        )
        record.setdefault("input_mode", input_mode)
        records.append(record)
    return records


def _mean_optional(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return None if not clean else mean(clean)


def _paraphrase_diagnostics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    return {
        "count": len(records),
        "mean_original_token_count": mean(float(row["original_token_count"]) for row in records),
        "mean_attacked_token_count": mean(float(row["attacked_token_count"]) for row in records),
        "mean_token_length_delta": mean(float(row["token_length_delta"]) for row in records),
        "mean_compression_ratio": _mean_optional([row.get("compression_ratio") for row in records]),
        "mean_paraphrase_seconds": mean(float(row["paraphrase_seconds"]) for row in records),
    }


def evaluate_paraphrase_attack(
    detections: Sequence[dict[str, Any]],
    attacked_records: Sequence[dict[str, Any]],
    *,
    threshold: float | None,
    input_mode: str,
    presence_test: str = "z_score",
) -> dict[str, Any]:
    frozen: list[dict[str, Any]] = []
    for row in detections:
        updated = dict(row)
        if presence_test == "z_score":
            if threshold is None:
                raise ValueError("z_score attack evaluation requires a calibrated Z threshold")
            updated["threshold"] = threshold
            updated["detected"] = float(updated.get("z_score", 0.0)) >= threshold
        frozen.append(updated)
    metrics = evaluate_records(frozen, input_mode=input_mode)
    return {
        "schema_version": 1,
        "presence": metrics["presence"],
        "presence_by_input_mode": metrics["presence_by_input_mode"],
        "payload": metrics["payload"],
        "payload_by_input_mode": metrics["payload_by_input_mode"],
        "runtime": metrics.get("runtime", {}),
        "attack": _paraphrase_diagnostics(attacked_records),
    }


def _write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "attack",
        "tpr",
        "model_fpr",
        "natural_fpr",
        "auc",
        "exact_message_recovery",
        "message_ber",
        "mean_token_length_delta",
        "mean_compression_ratio",
        "mean_paraphrase_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def plot_roc(metrics: Mapping[str, Any], output_dir: Path) -> None:
    cache_dir = Path("/tmp/matplotlib-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.8, 5.2))
    roc = metrics["presence"]["roc"]
    axis.plot(roc.get("fpr", []), roc.get("tpr", []), marker="o", linewidth=1.8, label="paraphrase")
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1.0)
    axis.set_xlabel("FPR")
    axis.set_ylabel("TPR")
    axis.set_title("Paraphrase attack")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "roc_paraphrase_attack.png", dpi=180)
    figure.savefig(figures / "roc_paraphrase_attack.pdf")
    plt.close(figure)


def _sentence_units(text: str) -> list[str]:
    units = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    return units if units else [text.strip()]


class Seq2SeqParaphraser:
    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        local_files_only: bool,
        max_input_tokens: int,
        max_output_tokens: int,
        num_beams: int,
        num_return_sequences: int,
        do_sample: bool,
        temperature: float | None,
        batch_size: int,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if max_input_tokens <= 0:
            raise ValueError("--max-input-tokens must be positive")
        if max_output_tokens <= 0:
            raise ValueError("--max-output-tokens must be positive")
        if num_beams <= 0:
            raise ValueError("--num-beams must be positive")
        if num_return_sequences <= 0:
            raise ValueError("--num-return-sequences must be positive")
        if batch_size <= 0:
            raise ValueError("--paraphraser-batch-size must be positive")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for the paraphraser, but torch.cuda.is_available() is False")

        self.model_path = str(model_path)
        self.device = device
        self.max_input_tokens = int(max_input_tokens)
        self.max_output_tokens = int(max_output_tokens)
        self.num_beams = int(num_beams)
        self.num_return_sequences = int(num_return_sequences)
        self.do_sample = bool(do_sample)
        self.temperature = temperature
        self.batch_size = int(batch_size)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=local_files_only)
        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        self.requested_max_input_tokens = int(max_input_tokens)
        self.requested_max_output_tokens = int(max_output_tokens)
        self.model_position_limit = None if model_limit is None else int(model_limit)
        self.max_input_tokens = _bounded_token_limit(
            max_input_tokens,
            self.model_position_limit,
            "--max-input-tokens",
        )
        self.max_output_tokens = _bounded_token_limit(
            max_output_tokens,
            self.model_position_limit,
            "--max-output-tokens",
        )
        self.model.to(device)
        self.model.eval()

    def generation_config(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "device": self.device,
            "requested_max_input_tokens": self.requested_max_input_tokens,
            "requested_max_output_tokens": self.requested_max_output_tokens,
            "model_position_limit": self.model_position_limit,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "num_beams": self.num_beams,
            "num_return_sequences": self.num_return_sequences,
            "do_sample": self.do_sample,
            "batch_size": self.batch_size,
        }
        if self.temperature is not None:
            values["temperature"] = self.temperature
        return values

    def _token_count(self, text: str) -> int:
        try:
            return len(
                self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                    verbose=False,
                )
            )
        except TypeError:
            if hasattr(self.tokenizer, "tokenize"):
                return len(self.tokenizer.tokenize(text))
            return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_oversized_unit(self, unit: str) -> list[str]:
        words = unit.split()
        if not words:
            return [unit]
        chunks: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word]).strip()
            if current and self._token_count(candidate) > self.max_input_tokens:
                chunks.append(" ".join(current).strip())
                current = [word]
            else:
                current.append(word)
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    def _chunks(self, text: str) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        units: list[str] = []
        for sentence in _sentence_units(text):
            if self._token_count(sentence) > self.max_input_tokens:
                units.extend(self._split_oversized_unit(sentence))
            else:
                units.append(sentence)
        for unit in units:
            candidate = " ".join([*current, unit]).strip()
            if current and self._token_count(candidate) > self.max_input_tokens:
                chunks.append(" ".join(current).strip())
                current = [unit]
            else:
                current.append(unit)
        if current:
            chunks.append(" ".join(current).strip())
        return chunks if chunks else [text]

    def paraphrase(self, text: str, *, seed: int) -> str:
        if not text.strip():
            return text
        self.torch.manual_seed(int(seed))
        if self.device == "cuda":
            self.torch.cuda.manual_seed_all(int(seed))
        chunks = self._chunks(text)
        outputs: list[str] = []
        with self.torch.no_grad():
            for start in range(0, len(chunks), self.batch_size):
                batch_text = chunks[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch_text,
                    truncation=True,
                    padding=True,
                    max_length=self.max_input_tokens,
                    return_tensors="pt",
                ).to(self.device)
                kwargs: dict[str, Any] = {
                    "max_length": self.max_output_tokens,
                    "num_beams": self.num_beams,
                    "num_return_sequences": self.num_return_sequences,
                    "do_sample": self.do_sample,
                }
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                generated = self.model.generate(**encoded, **kwargs)
                decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                stride = self.num_return_sequences
                for index in range(0, len(decoded), stride):
                    outputs.append(decoded[index].strip())
        rewritten = " ".join(part for part in outputs if part)
        return rewritten if rewritten else text


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_input_modes(args.headline_input_mode, args.input_modes)
    experiment_dir = resolve_input_path(args.experiment_dir)
    output_dir = (
        resolve_output_path(args.output_dir)
        if args.output_dir
        else default_output_dir(experiment_dir, args.point_id, args.paraphraser_slug)
    )
    _prepare_output(output_dir, resume=args.resume, overwrite=args.overwrite)

    threshold = load_attack_z_threshold(experiment_dir)
    runtime_config = build_runtime_config(
        experiment_dir,
        threshold,
        evaluate_all_policies=args.evaluate_all_policies,
    )
    tokenizer, vocab_size = load_tokenizer(
        runtime_config.model.path,
        local_files_only=runtime_config.model.local_files_only,
    )
    detector = build_detector(runtime_config, tokenizer, int(vocab_size))
    examples = load_attack_examples(experiment_dir, args.point_id, max_samples=args.max_samples)
    paraphraser = Seq2SeqParaphraser(
        args.paraphraser_model,
        device=args.paraphraser_device,
        local_files_only=args.local_files_only,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        do_sample=args.do_sample,
        temperature=args.temperature,
        batch_size=args.paraphraser_batch_size,
    )
    generation = paraphraser.generation_config()

    metadata = {
        "schema_version": 1,
        "experiment_dir": str(experiment_dir),
        "point_id": args.point_id,
        "attack": args.attack,
        "z_calibrated_threshold": threshold,
        "headline_input_mode": args.headline_input_mode,
        "input_modes": list(args.input_modes),
        "evaluate_all_policies": args.evaluate_all_policies,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "paraphraser_model": str(args.paraphraser_model),
        "paraphraser_slug": args.paraphraser_slug,
        "generation": generation,
    }
    write_json(output_dir / "metadata.json", metadata)

    attack_dir = output_dir / args.attack
    attacked_records: list[dict[str, Any]] = []
    detection_records: list[dict[str, Any]] = []
    for example in tqdm(examples, desc=f"Attack {args.attack}", unit="sample"):
        seed = _sample_seed(args.seed, args.attack, example.sample_id, example.text_class)
        import time

        begin = time.perf_counter()
        attacked_text = paraphraser.paraphrase(example.text, seed=seed)
        paraphrase_seconds = time.perf_counter() - begin
        result = make_paraphrase_result(
            example.text,
            attacked_text,
            tokenizer,
            paraphrase_seconds=paraphrase_seconds,
            model_path=args.paraphraser_model,
            generation=generation,
        )
        attacked_records.append(_paraphrase_record(example, args.attack, result, seed))
        for detection in detect_paraphrased_example(
            example,
            result,
            detector,
            runtime_config,
            args.attack,
            attack_seed=seed,
            input_modes=args.input_modes,
        ):
            detection_records.append(detection)

    _write_jsonl(attack_dir / "attacked_records.jsonl", attacked_records)
    _write_jsonl(attack_dir / "detections.jsonl", detection_records)
    metrics = evaluate_paraphrase_attack(
        detection_records,
        attacked_records,
        threshold=threshold,
        input_mode=args.headline_input_mode,
        presence_test=runtime_config.detection.presence_test,
    )
    write_json(attack_dir / "metrics.json", metrics)
    write_roc_points(attack_dir / "roc_points.csv", metrics["presence"]["roc"])
    summary = {"schema_version": 1, "metadata": metadata, "attacks": {args.attack: metrics}}
    write_json(output_dir / "summary.json", summary)
    _write_summary_csv(
        output_dir / "summary.csv",
        [
            {
                "attack": args.attack,
                "tpr": metrics["presence"]["tpr"],
                "model_fpr": metrics["presence"]["model_fpr"],
                "natural_fpr": metrics["presence"]["natural_fpr"],
                "auc": metrics["presence"]["auc"],
                "exact_message_recovery": metrics["payload"]["exact_message_recovery"],
                "message_ber": metrics["payload"]["message_ber"],
                "mean_token_length_delta": metrics["attack"]["mean_token_length_delta"],
                "mean_compression_ratio": metrics["attack"]["mean_compression_ratio"],
                "mean_paraphrase_seconds": metrics["attack"]["mean_paraphrase_seconds"],
            }
        ],
    )
    plot_roc(metrics, output_dir)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
