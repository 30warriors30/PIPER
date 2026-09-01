from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import shutil
from statistics import mean
import time
from typing import Any

from tqdm import tqdm

from evaluation.metrics import evaluate_records
from experiments.config import ParetoExperimentConfig
from experiments.detection import _runtime_config
from utils.detection import build_detector, detection_record, load_tokenizer
from utils.io import iter_jsonl, plain, read_json, write_json

from experiments.attack.synonym_attack import AttackKind, AttackResult, WordNetSynonymProvider, attack_text

DEFAULT_EXPERIMENT_DIR = "outputs/experiments/opt13b_pareto_49x200"
DEFAULT_ATTACKS: tuple[AttackKind, ...] = ("replacement", "deletion", "insertion")
INPUT_MODES = ("known_boundary", "blind_text")


@dataclass(frozen=True)
class AttackExample:
    sample_id: str
    split: str
    text_class: str
    prompt_token_ids: list[int]
    text: str
    token_ids: list[int]
    message_bits: str | None
    encoded_bits: str | None


def project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return project_dir() / path


def resolve_input_path(value: str | Path) -> Path:
    return _resolve_path(value)


def resolve_output_path(value: str | Path) -> Path:
    return _resolve_path(value)


def _rate_slug(attack_rate: float) -> str:
    return f"{attack_rate * 100:g}".replace(".", "p")


def default_output_dir(experiment_dir: Path, point_id: str, attack_rate: float) -> Path:
    return experiment_dir / "attacks" / f"{point_id}_synonym_rate{_rate_slug(attack_rate)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paper-aligned synonym substitution attacks for the dual-layer watermark"
    )
    parser.add_argument("--experiment-dir", default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--point-id", default="soft_p0_m2")
    parser.add_argument("--attack-rate", type=float, default=0.10)
    parser.add_argument("--attacks", nargs="+", choices=list(DEFAULT_ATTACKS), default=list(DEFAULT_ATTACKS))
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--headline-input-mode", choices=["known_boundary", "blind_text"], default="known_boundary")
    parser.add_argument("--input-modes", nargs="+", choices=list(INPUT_MODES), default=list(INPUT_MODES))
    parser.add_argument("--evaluate-all-policies", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-full-rate", action="store_true")
    return parser


def _ints(values: Sequence[int]) -> list[int]:
    return [int(value) for value in values]


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def load_attack_examples(
    experiment_dir: Path,
    point_id: str,
    max_samples: int | None = None,
) -> list[AttackExample]:
    if max_samples is not None and max_samples <= 0:
        return []

    watermarked_path = experiment_dir / "runs" / point_id / "watermarked.jsonl"
    baseline_path = experiment_dir / "shared" / "baseline.jsonl"
    examples: list[AttackExample] = []

    for row in iter_jsonl(watermarked_path):
        if row.get("split") != "test":
            continue
        examples.append(
            AttackExample(
                sample_id=str(row["sample_id"]),
                split="test",
                text_class="watermarked",
                prompt_token_ids=_ints(row["prompt_token_ids"]),
                text=str(row["watermarked_text"]),
                token_ids=_ints(row["watermarked_token_ids"]),
                message_bits=_optional_str(row["message_bits"]),
                encoded_bits=_optional_str(row["encoded_bits"]),
            )
        )
        if max_samples is not None and len([x for x in examples if x.text_class == "watermarked"]) >= max_samples:
            break

    selected_ids = {row.sample_id for row in examples if row.text_class == "watermarked"}
    for row in iter_jsonl(baseline_path):
        if row.get("split") != "test" or str(row["sample_id"]) not in selected_ids:
            continue
        for text_class, text_field, token_field in (
            ("unwatermarked", "unwatermarked_text", "unwatermarked_token_ids"),
            ("natural", "natural_text", "natural_token_ids"),
        ):
            examples.append(
                AttackExample(
                    sample_id=str(row["sample_id"]),
                    split="test",
                    text_class=text_class,
                    prompt_token_ids=_ints(row["prompt_token_ids"]),
                    text=str(row[text_field]),
                    token_ids=_ints(row[token_field]),
                    message_bits=None,
                    encoded_bits=None,
                )
            )
    return examples


def build_pareto_config(experiment_dir: Path) -> ParetoExperimentConfig:
    payload = read_json(experiment_dir / "experiment.json")["config"]
    values = dict(payload)
    values["output_root"] = experiment_dir.parent
    values["experiment_id"] = experiment_dir.name
    return ParetoExperimentConfig(
        **{
            key: value
            for key, value in values.items()
            if key in ParetoExperimentConfig.__dataclass_fields__
        }
    )


def build_runtime_config(
    experiment_dir: Path,
    calibrated_threshold: float,
    *,
    evaluate_all_policies: bool = False,
):
    runtime = _runtime_config(build_pareto_config(experiment_dir), calibrated_threshold)
    return replace(
        runtime,
        model=replace(runtime.model, device="cpu"),
        decoding=replace(runtime.decoding, evaluate_all_policies=evaluate_all_policies),
    )


def _timed_detect(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def _attack_record(
    example: AttackExample,
    attack: AttackKind,
    result: AttackResult,
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
        "editable_word_count": result.editable_word_count,
        "target_edit_count": result.target_edit_count,
        "achieved_edit_count": result.achieved_edit_count,
        "achieved_attack_rate": result.achieved_attack_rate,
        "token_length_delta": result.token_length_delta,
        "edits": [edit.__dict__ for edit in result.edits],
    }


def detect_attacked_example(
    example: AttackExample,
    result: AttackResult,
    detector: Any,
    runtime_config: Any,
    attack: AttackKind,
    *,
    attack_seed: int,
    attack_seconds: float,
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
                "attack_seconds": attack_seconds,
                "original_token_count": result.original_token_count,
                "attacked_token_count": result.attacked_token_count,
                "target_edit_count": result.target_edit_count,
                "achieved_edit_count": result.achieved_edit_count,
                "achieved_attack_rate": result.achieved_attack_rate,
                "token_length_delta": result.token_length_delta,
            }
        )
        records.append(record)
    return records


def _attack_diagnostics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    target_total = sum(int(row["target_edit_count"]) for row in records)
    achieved_total = sum(int(row["achieved_edit_count"]) for row in records)
    return {
        "count": len(records),
        "target_edit_count": target_total,
        "achieved_edit_count": achieved_total,
        "mean_achieved_attack_rate": mean(float(row["achieved_attack_rate"]) for row in records),
        "mean_candidate_coverage": None if target_total == 0 else achieved_total / target_total,
        "mean_token_length_delta": mean(float(row["token_length_delta"]) for row in records),
    }


def evaluate_attack(
    detections: Sequence[dict[str, Any]],
    attacked_records: Sequence[dict[str, Any]],
    *,
    threshold: float,
    input_mode: str,
    presence_test: str = "z_score",
) -> dict[str, Any]:
    frozen: list[dict[str, Any]] = []
    for row in detections:
        updated = dict(row)
        if presence_test == "z_score":
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
        "attack": _attack_diagnostics(attacked_records),
    }


def write_roc_points(path: Path, roc: Mapping[str, Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fpr", "tpr", "threshold"])
        for fpr, tpr, threshold in zip(
            roc.get("fpr", []),
            roc.get("tpr", []),
            roc.get("thresholds", []),
            strict=True,
        ):
            writer.writerow([fpr, tpr, threshold])


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = json.dumps(plain(row), ensure_ascii=False, separators=(",", ":"))
            handle.write(payload + "\n")


def validate_input_modes(headline_input_mode: str, input_modes: Sequence[str]) -> None:
    if len(set(input_modes)) != len(input_modes):
        raise ValueError("--input-modes contains duplicate entries")
    if headline_input_mode not in set(input_modes):
        raise ValueError("headline input mode must be included in --input-modes")


def _prepare_output(path: Path, *, resume: bool, overwrite: bool) -> None:
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if resume:
        raise NotImplementedError("resume is not supported for synonym attacks")
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()) and not (resume or overwrite):
        raise FileExistsError(f"Output directory exists: {path}. Use --overwrite to replace it.")
    path.mkdir(parents=True, exist_ok=True)


def _sample_seed(base_seed: int, attack: str, sample_id: str, text_class: str) -> int:
    values = (str(int(base_seed)), attack, sample_id, text_class)
    payload = "\0".join(values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="big")


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
        "mean_achieved_attack_rate",
        "mean_candidate_coverage",
        "mean_token_length_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def plot_rocs(metrics_by_attack: Mapping[str, dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.8, 5.2))
    for attack, metrics in metrics_by_attack.items():
        roc = metrics["presence"]["roc"]
        axis.plot(roc.get("fpr", []), roc.get("tpr", []), marker="o", linewidth=1.8, label=attack)
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1.0)
    axis.set_xlabel("FPR")
    axis.set_ylabel("TPR")
    axis.set_title("Synonym substitution attacks")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "roc_synonym_attacks.png", dpi=180)
    figure.savefig(figures / "roc_synonym_attacks.pdf")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_input_modes(args.headline_input_mode, args.input_modes)
    experiment_dir = resolve_input_path(args.experiment_dir)
    output_dir = (
        resolve_output_path(args.output_dir)
        if args.output_dir
        else default_output_dir(experiment_dir, args.point_id, args.attack_rate)
    )
    _prepare_output(output_dir, resume=args.resume, overwrite=args.overwrite)

    calibration = read_json(experiment_dir / "shared" / "calibration.json")
    threshold = float(calibration["calibrated_threshold"])
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
    provider = WordNetSynonymProvider()
    examples = load_attack_examples(experiment_dir, args.point_id, max_samples=args.max_samples)

    metadata = {
        "schema_version": 1,
        "experiment_dir": str(experiment_dir),
        "point_id": args.point_id,
        "attack_rate": args.attack_rate,
        "attacks": list(args.attacks),
        "calibrated_threshold": threshold,
        "headline_input_mode": args.headline_input_mode,
        "input_modes": list(args.input_modes),
        "evaluate_all_policies": args.evaluate_all_policies,
        "max_samples": args.max_samples,
        "seed": args.seed,
    }
    write_json(output_dir / "metadata.json", metadata)

    summary_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"schema_version": 1, "metadata": metadata, "attacks": {}}
    for attack in args.attacks:
        attack_dir = output_dir / attack
        attacked_records: list[dict[str, Any]] = []
        detection_records: list[dict[str, Any]] = []
        for example in tqdm(examples, desc=f"Attack {attack}", unit="sample"):
            seed = _sample_seed(args.seed, attack, example.sample_id, example.text_class)
            started = time.perf_counter()
            attack_result = attack_text(
                example.text,
                tokenizer,
                provider,
                attack,
                attack_rate=args.attack_rate,
                seed=seed,
                require_full_rate=args.require_full_rate,
            )
            attack_seconds = time.perf_counter() - started
            attacked_record = _attack_record(example, attack, attack_result, seed)
            attacked_records.append(attacked_record)
            for detection in detect_attacked_example(
                example,
                attack_result,
                detector,
                runtime_config,
                attack,
                attack_seed=seed,
                attack_seconds=attack_seconds,
                input_modes=args.input_modes,
            ):
                detection_records.append(detection)

        _write_jsonl(attack_dir / "attacked_records.jsonl", attacked_records)
        _write_jsonl(attack_dir / "detections.jsonl", detection_records)
        metrics = evaluate_attack(
            detection_records,
            attacked_records,
            threshold=threshold,
            input_mode=args.headline_input_mode,
            presence_test=runtime_config.detection.presence_test,
        )
        write_json(attack_dir / "metrics.json", metrics)
        write_roc_points(attack_dir / "roc_points.csv", metrics["presence"]["roc"])
        summary["attacks"][attack] = metrics
        summary_rows.append(
            {
                "attack": attack,
                "tpr": metrics["presence"]["tpr"],
                "model_fpr": metrics["presence"]["model_fpr"],
                "natural_fpr": metrics["presence"]["natural_fpr"],
                "auc": metrics["presence"]["auc"],
                "exact_message_recovery": metrics["payload"]["exact_message_recovery"],
                "message_ber": metrics["payload"]["message_ber"],
                "mean_achieved_attack_rate": metrics["attack"]["mean_achieved_attack_rate"],
                "mean_candidate_coverage": metrics["attack"]["mean_candidate_coverage"],
                "mean_token_length_delta": metrics["attack"]["mean_token_length_delta"],
            }
        )
    write_json(output_dir / "summary.json", summary)
    _write_summary_csv(output_dir / "summary.csv", summary_rows)
    plot_rocs(summary["attacks"], output_dir)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
