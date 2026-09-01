from __future__ import annotations

import argparse
from functools import lru_cache
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from tqdm import tqdm

from utils.generation import paired_rng
from watermark.ecc import BCHCodec


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_input_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    current_candidate = (Path.cwd() / path).resolve()
    if current_candidate.exists():
        return current_candidate
    project_candidate = (_project_dir() / path).resolve()
    if project_candidate.exists():
        return project_candidate
    repo_candidate = (_project_dir().parent / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return current_candidate


def resolve_output_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd = Path.cwd().resolve()
    project_dir = _project_dir().resolve()
    if _is_relative_to(cwd, project_dir):
        return (cwd / path).resolve()
    return (project_dir / path).resolve()


@lru_cache(maxsize=1)
def _bch23_codec() -> BCHCodec:
    return BCHCodec(23, 8, 3)


def _bits(value: str, *, expected_length: int | None = None) -> tuple[int, ...]:
    bits = tuple(int(bit) for bit in value)
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("message bits must contain only 0 and 1")
    if expected_length is not None and len(bits) != expected_length:
        raise ValueError(f"message bits must contain exactly {expected_length} bits")
    return bits


def _bit_accuracy(expected: str, observed: str) -> float | None:
    if len(expected) != len(observed):
        return None
    if not expected:
        return None
    correct = sum(left == right for left, right in zip(expected, observed, strict=True))
    return correct / len(expected)


def mpac_digits_from_bits(bits: str, *, base: int) -> str:
    _bits(bits)
    if int(base) < 2:
        raise ValueError("base must be at least 2")
    decimal = int(bits, 2) if bits else 0
    if decimal == 0:
        return "0"
    digits: list[str] = []
    while decimal:
        digits.append(str(decimal % int(base)))
        decimal //= int(base)
    return "".join(reversed(digits))


def _mpac_bits_from_digits(digits: str, *, width: int, base: int) -> str:
    if int(base) < 2:
        raise ValueError("base must be at least 2")
    decimal = int(digits or "0", int(base))
    decimal = min(decimal, 2 ** int(width) - 1)
    return format(decimal, f"0{int(width)}b")


def embedded_message_length(args: argparse.Namespace) -> int:
    if args.mpac_ecc == "none":
        return int(args.message_length)
    if args.mpac_ecc == "bch23":
        return 23
    raise ValueError(f"Unsupported MPAC ECC mode: {args.mpac_ecc}")


def embedded_payload_bits(message_bits: str, args: argparse.Namespace) -> str:
    if args.mpac_ecc == "none":
        _bits(message_bits, expected_length=int(args.message_length))
        return message_bits
    if args.mpac_ecc == "bch23":
        if int(args.message_length) != 8:
            raise ValueError("MPAC BCH(23,8,3) mode requires --message-length 8")
        codeword = _bch23_codec().encode(_bits(message_bits, expected_length=8))
        return "".join(map(str, codeword))
    raise ValueError(f"Unsupported MPAC ECC mode: {args.mpac_ecc}")


def validate_runtime_args(args: argparse.Namespace) -> None:
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested, but no CUDA GPU is visible in this environment. "
            "Run on a GPU node/session or use --device cpu --dtype float32 only for tiny debugging runs."
        )
    if args.mpac_ecc == "bch23" and int(args.message_length) != 8:
        raise ValueError("MPAC BCH(23,8,3) ablation requires --message-length 8")


def evaluate_payload_prediction(
    *,
    pred_message: str,
    original_message_bits: str,
    embedded_message_bits: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    predicted_embedded_bits = _mpac_bits_from_digits(
        pred_message,
        width=len(embedded_message_bits),
        base=int(args.base),
    )
    embedded_bit_acc = _bit_accuracy(embedded_message_bits, predicted_embedded_bits)
    result: dict[str, Any] = {
        "predicted_embedded_bits": predicted_embedded_bits,
        "embedded_bit_acc": embedded_bit_acc,
        "bch_decode_status": None,
        "bch_corrected_errors": None,
        "decoded_message_bits": None,
        "original_bit_acc": None,
        "message_recovered": False,
    }

    if args.mpac_ecc == "none":
        original_bit_acc = _bit_accuracy(original_message_bits, predicted_embedded_bits)
        result.update(
            {
                "decoded_message_bits": predicted_embedded_bits,
                "original_bit_acc": original_bit_acc,
                "message_recovered": predicted_embedded_bits == original_message_bits,
            }
        )
        return result

    if args.mpac_ecc == "bch23":
        decoded = _bch23_codec().decode(_bits(predicted_embedded_bits, expected_length=23))
        result["bch_decode_status"] = decoded.status
        result["bch_corrected_errors"] = decoded.corrected_errors
        if decoded.status == "decoded" and decoded.message_bits is not None:
            decoded_message_bits = "".join(map(str, decoded.message_bits))
            result.update(
                {
                    "decoded_message_bits": decoded_message_bits,
                    "original_bit_acc": _bit_accuracy(original_message_bits, decoded_message_bits),
                    "message_recovered": decoded_message_bits == original_message_bits,
                }
            )
        return result

    raise ValueError(f"Unsupported MPAC ECC mode: {args.mpac_ecc}")


def load_manifest_rows(path: Path, *, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        if str(row.get("split")) != split:
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def empirical_threshold(scores: Sequence[float], target_fpr: float) -> float:
    if not scores:
        raise ValueError("scores must not be empty")
    if not 0.0 <= float(target_fpr) < 1.0:
        raise ValueError("target_fpr must be in [0, 1)")
    values = [float(score) for score in scores]
    count = len(values)
    for candidate in sorted(set(values)):
        false_positive_rate = sum(score >= candidate for score in values) / count
        if false_positive_rate <= float(target_fpr):
            return float(candidate)
    return math.nextafter(max(values), math.inf)


def _rate(rows: Sequence[dict[str, Any]], field: str, threshold: float) -> float | None:
    if not rows:
        return None
    return sum(float(row[field]) >= threshold for row in rows) / len(rows)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_records(
    records: Iterable[dict[str, Any]],
    *,
    threshold: float,
    target_fpr: float,
) -> dict[str, Any]:
    rows = list(records)
    watermarked = [row for row in rows if row.get("text_class") == "watermarked"]
    model_negatives = [row for row in rows if row.get("text_class") == "unwatermarked"]
    natural_negatives = [row for row in rows if row.get("text_class") == "natural"]
    negatives = model_negatives + natural_negatives

    exact = [
        bool(row.get("message_recovered", row.get("bit_match")))
        for row in watermarked
    ]
    detected_exact = [
        bool(row.get("message_recovered", row.get("bit_match"))) and float(row["z_score"]) >= threshold
        for row in watermarked
    ]
    bit_acc = [
        float(row.get("original_bit_acc", row.get("bit_acc")))
        for row in watermarked
        if isinstance(row.get("original_bit_acc", row.get("bit_acc")), (float, int))
        and math.isfinite(float(row.get("original_bit_acc", row.get("bit_acc"))))
    ]
    embedded_bit_acc = [
        float(row["embedded_bit_acc"])
        for row in watermarked
        if isinstance(row.get("embedded_bit_acc"), (float, int))
        and math.isfinite(float(row["embedded_bit_acc"]))
    ]

    return {
        "target_fpr": float(target_fpr),
        "threshold": float(threshold),
        "num_watermarked": len(watermarked),
        "num_model_negatives": len(model_negatives),
        "num_natural_negatives": len(natural_negatives),
        "tpr": _rate(watermarked, "z_score", threshold),
        "model_fpr": _rate(model_negatives, "z_score", threshold),
        "natural_fpr": _rate(natural_negatives, "z_score", threshold),
        "combined_fpr": _rate(negatives, "z_score", threshold),
        "exact_message_recovery": _mean([float(value) for value in exact]),
        "end_to_end_exact_recovery": _mean([float(value) for value in detected_exact]),
        "mean_bit_accuracy": _mean(bit_acc),
        "mean_embedded_bit_accuracy": _mean(embedded_bit_acc),
        "mean_watermarked_z_score": _mean([float(row["z_score"]) for row in watermarked]),
        "mean_negative_z_score": _mean([float(row["z_score"]) for row in negatives]),
    }


def _powerset(iterable: Iterable[Any]) -> Iterable[tuple[Any, ...]]:
    values = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(values, size) for size in range(1, len(values) + 1)
    )


def _patch_mpac_candidate_overflow(mpac_module: Any) -> None:
    detector_cls = getattr(mpac_module, "WatermarkDetector")
    if getattr(detector_cls, "_brew_candidate_overflow_patch", False):
        return

    def _predict_message(self: Any, position_cnt: Any, green_cnt_by_position: Any, p_val_per_pos: Any, num_candidates: int = 16) -> tuple[list[Any], list[Any], float]:
        started = time.time()
        msg_prediction = []
        confidence_per_pos = []
        for pos in range(1, self.converted_msg_length + 1):
            p_val = p_val_per_pos[pos - 1]
            if position_cnt.get(pos) is None:
                position_cnt[pos] = -1
                pred, next_idx = random.sample(list(range(self.base)), 2)
                confidence_per_pos.append((-p_val, pred, next_idx, pos))
            else:
                green_counts = green_cnt_by_position[pos]
                sorted_idx = sorted(range(len(green_counts)), key=lambda idx: (green_counts[idx], idx))
                max_idx, next_idx = sorted_idx[-1], sorted_idx[-2]
                pred = max_idx
                confidence_per_pos.append((-p_val, max_idx, next_idx, pos))
            msg_prediction.append(pred)

        random.shuffle(confidence_per_pos)
        random_prediction_list = [msg_prediction]
        for _, _max_idx, next_idx, pos in confidence_per_pos[: int(num_candidates)]:
            cand_msg = msg_prediction.copy()
            cand_msg[pos - 1] = next_idx
            random_prediction_list.append(cand_msg)

        num_candidate_position = min(
            math.ceil(math.log2(int(num_candidates) + 1)),
            len(confidence_per_pos),
        )
        confidence_subset = sorted(confidence_per_pos, key=lambda item: item[0])[:num_candidate_position]
        msg_prediction_list = [msg_prediction]
        candidate_iter = iter(_powerset(confidence_subset))
        while len(msg_prediction_list) <= int(num_candidates):
            try:
                candidate = next(candidate_iter)
            except StopIteration:
                break
            cand_msg = msg_prediction.copy()
            for _, _max_idx, next_idx, pos in candidate:
                cand_msg[pos - 1] = next_idx
            msg_prediction_list.append(cand_msg)

        return msg_prediction_list, random_prediction_list, time.time() - started

    detector_cls._predict_message = _predict_message
    detector_cls._brew_candidate_overflow_patch = True


def import_mpac(mb_repo: Path) -> tuple[Any, Any]:
    release_dir = mb_repo / "watermark_reliability_release"
    import_dir = release_dir if release_dir.exists() else mb_repo
    if not import_dir.exists():
        raise FileNotFoundError(import_dir)
    sys.path.insert(0, str(import_dir))
    import mb_watermark_processor

    _patch_mpac_candidate_overflow(mb_watermark_processor)
    return mb_watermark_processor.WatermarkLogitsProcessor, mb_watermark_processor.WatermarkDetector


def _torch_dtype(name: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, torch.device]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"local_files_only": args.local_files_only}
    dtype = _torch_dtype(args.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, **kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _generation_kwargs(args: argparse.Namespace, tokenizer: Any, processor: Any) -> dict[str, Any]:
    from transformers import LogitsProcessorList

    kwargs: dict[str, Any] = {
        "min_new_tokens": args.exact_tokens,
        "max_new_tokens": args.exact_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "logits_processor": LogitsProcessorList([processor]),
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def make_processor(args: argparse.Namespace, tokenizer: Any, cls: Any, device: torch.device) -> Any:
    message_length = embedded_message_length(args)
    return cls(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        delta=args.delta,
        base=args.base,
        seeding_scheme=args.seeding_scheme,
        store_spike_ents=False,
        select_green_tokens=True,
        message_length=message_length,
        code_length=message_length,
        use_position_prf=args.use_position_prf,
        use_fixed_position=args.use_fixed_position,
        use_feedback=False,
        device=str(device),
    )


def make_detector(args: argparse.Namespace, tokenizer: Any, cls: Any, device: torch.device) -> Any:
    message_length = embedded_message_length(args)
    return cls(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seeding_scheme=args.seeding_scheme,
        device=str(device),
        tokenizer=tokenizer,
        z_threshold=0.0,
        normalizers=[],
        ignore_repeated_ngrams=args.ignore_repeated_ngrams,
        message_length=message_length,
        code_length=message_length,
        base=args.base,
        use_position_prf=args.use_position_prf,
        use_fixed_position=args.use_fixed_position,
    )


def score_text(
    detector: Any,
    *,
    token_ids: Sequence[int],
    message_bits: str,
    sampled_positions: str,
    device: torch.device,
) -> dict[str, Any]:
    detector.position_increment = 0
    tensor = torch.tensor(list(token_ids), dtype=torch.long, device=device)
    score = detector.detect(
        tokenized_text=tensor,
        return_prediction=False,
        convert_to_float=True,
        return_z_at_T=False,
        message=message_bits,
        position=sampled_positions,
    )
    return {
        "z_score": float(score["z_score"]),
        "p_value": float(score["p_value"]),
        "pred_message": str(score["pred_message"]),
        "bit_acc": float(score["bit_acc"]),
        "bit_match": bool(score["bit_match"]),
        "cand_match": bool(score["cand_match"]),
        "cand_acc": float(score["cand_acc"]),
        "num_tokens_scored": int(score["num_tokens_scored"]),
        "num_green_tokens": int(score["num_green_tokens"]),
        "green_fraction": float(score["green_fraction"]),
        "position_acc": float(score["position_acc"]),
    }


def generate_mpac_watermarked(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    processor_cls: Any,
) -> dict[str, Any]:
    processor = make_processor(args, tokenizer, processor_cls, device)
    original_message_bits = str(row["message_bits"])
    embedded_bits = embedded_payload_bits(original_message_bits, args)
    processor.set_message(embedded_bits)
    input_ids = torch.tensor([list(row["prompt_token_ids"])], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    kwargs = _generation_kwargs(args, tokenizer, processor)
    kwargs["attention_mask"] = attention_mask
    started = time.perf_counter()
    with paired_rng(int(row["generation_seed"]), device), torch.inference_mode():
        output = model.generate(input_ids=input_ids, **kwargs)
    elapsed = time.perf_counter() - started
    if not isinstance(output, torch.Tensor):
        output = output.sequences
    continuation = [int(value) for value in output[0, input_ids.shape[1] :].tolist()]
    if len(continuation) != args.exact_tokens:
        raise RuntimeError(
            f"Exact-length invariant failed for sample {row['sample_id']}: "
            f"got {len(continuation)}, expected {args.exact_tokens}"
        )
    sampled_positions = processor.flush_position()[0]
    processor.position_increment = 0
    return {
        "sample_id": str(row["sample_id"]),
        "split": str(row["split"]),
        "text_class": "watermarked",
        "message_bits": original_message_bits,
        "embedded_message_bits": embedded_bits,
        "mpac_ecc": args.mpac_ecc,
        "generation_seed": int(row["generation_seed"]),
        "prompt_token_ids": list(row["prompt_token_ids"]),
        "token_ids": continuation,
        "text": tokenizer.decode(continuation, skip_special_tokens=True),
        "sampled_positions": sampled_positions,
        "generation_seconds": elapsed,
    }


def _dummy_positions(token_ids: Sequence[int], context_width: int, self_salt: bool) -> str:
    return "0" * (len(token_ids) + context_width + int(self_salt))


def build_negative_records(
    baseline_path: Path,
    *,
    args: argparse.Namespace,
    detector: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in tqdm(list(iter_jsonl(baseline_path)), desc="Score MPAC negatives", unit="sample"):
        for text_class, text_field, ids_field in (
            ("unwatermarked", "unwatermarked_text", "unwatermarked_token_ids"),
            ("natural", "natural_text", "natural_token_ids"),
        ):
            token_ids = [int(value) for value in row[ids_field]]
            original_message_bits = str(row["message_bits"])
            embedded_bits = embedded_payload_bits(original_message_bits, args)
            score = score_text(
                detector,
                token_ids=token_ids,
                message_bits=embedded_bits,
                sampled_positions=_dummy_positions(
                    token_ids,
                    int(detector.context_width),
                    bool(detector.self_salt),
                ),
                device=device,
            )
            records.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "split": str(row["split"]),
                    "text_class": text_class,
                    "message_bits": original_message_bits,
                    "embedded_message_bits": embedded_bits,
                    "mpac_ecc": args.mpac_ecc,
                    "token_ids": token_ids,
                    "text": str(row.get(text_field, "")),
                    **score,
                }
            )
    return records


def load_brew_summary(experiment_dir: Path, point_id: str) -> dict[str, Any]:
    metrics_path = experiment_dir / "runs" / point_id / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    presence = metrics.get("presence", {})
    payload = metrics.get("payload", {})
    diagnostics = metrics.get("payload_diagnostics", {})
    quality = metrics.get("quality", {})
    return {
        "point_id": point_id,
        "tpr": presence.get("tpr"),
        "model_fpr": presence.get("model_fpr"),
        "natural_fpr": presence.get("natural_fpr"),
        "exact_message_recovery": payload.get("error_erasure_exact_message_recovery"),
        "wrong_message_rate": diagnostics.get("wrong_message_rate"),
        "abstention_rate": diagnostics.get("abstention_rate"),
        "mean_bit_accuracy": None,
        "paired_delta_nll": quality.get("paired_delta_nll", metrics.get("paired_delta_nll")),
        "ppl_ratio": quality.get("ppl_ratio", metrics.get("ppl_ratio")),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_runtime_args(args)
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    records_path = output_dir / "records.jsonl"
    if records_path.exists() and not args.overwrite:
        raise FileExistsError(f"{records_path} exists; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    if records_path.exists():
        records_path.unlink()

    metadata = {
        "schema_version": 1,
        "baseline": "MPAC",
        "source_repo": "https://github.com/bangawayoo/mb-lm-watermarking",
        "mb_repo": str(Path(args.mb_repo)),
        "experiment_dir": str(experiment_dir),
        "model_path": args.model_path,
        "exact_tokens": args.exact_tokens,
        "message_length": args.message_length,
        "base": args.base,
        "gamma": args.gamma,
        "delta": args.delta,
        "seeding_scheme": args.seeding_scheme,
        "use_position_prf": args.use_position_prf,
        "use_fixed_position": args.use_fixed_position,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "target_fpr": args.target_fpr,
        "mpac_ecc": args.mpac_ecc,
        "embedded_message_length": embedded_message_length(args),
    }
    if args.mpac_ecc == "bch23":
        metadata.update({"ecc_n": 23, "ecc_k": 8, "ecc_t": 3})
    write_json(output_dir / "metadata.json", metadata)

    processor_cls, detector_cls = import_mpac(Path(args.mb_repo))
    model, tokenizer, device = load_model_and_tokenizer(args)
    detector = make_detector(args, tokenizer, detector_cls, device)

    records = build_negative_records(
        experiment_dir / "shared" / "baseline.jsonl",
        args=args,
        detector=detector,
        device=device,
    )
    test_rows = load_manifest_rows(experiment_dir / "manifest.jsonl", split="test", limit=args.limit_test)
    for row in tqdm(test_rows, desc="Generate and score MPAC positives", unit="sample"):
        generated = generate_mpac_watermarked(
            row,
            args=args,
            model=model,
            tokenizer=tokenizer,
            device=device,
            processor_cls=processor_cls,
        )
        score = score_text(
            detector,
            token_ids=generated["token_ids"],
            message_bits=generated["embedded_message_bits"],
            sampled_positions=generated["sampled_positions"],
            device=device,
        )
        payload_eval = evaluate_payload_prediction(
            pred_message=score["pred_message"],
            original_message_bits=generated["message_bits"],
            embedded_message_bits=generated["embedded_message_bits"],
            args=args,
        )
        records.append({**generated, **score, **payload_eval})

    for record in records:
        append_jsonl(records_path, record)

    calibration_negative_scores = [
        float(row["z_score"])
        for row in records
        if row.get("split") == "calibration" and row.get("text_class") != "watermarked"
    ]
    threshold = empirical_threshold(calibration_negative_scores, args.target_fpr)
    test_records = [row for row in records if row.get("split") == "test"]
    mpac_summary = summarize_records(test_records, threshold=threshold, target_fpr=args.target_fpr)
    summary = {
        "schema_version": 1,
        "mpac": mpac_summary,
        "brew": load_brew_summary(experiment_dir, args.brew_point_id),
    }
    write_json(output_dir / "metrics.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MPAC on the existing BREW T=200 b=8 data.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--mb-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--brew-point-id", default="soft_p0_m2")
    parser.add_argument("--exact-tokens", type=int, default=200)
    parser.add_argument("--message-length", type=int, default=8)
    parser.add_argument("--mpac-ecc", choices=["none", "bch23"], default="none")
    parser.add_argument("--base", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--seeding-scheme", default="lefthash")
    parser.add_argument("--use-position-prf", action="store_true")
    parser.add_argument("--use-fixed-position", action="store_true")
    parser.add_argument("--ignore-repeated-ngrams", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
