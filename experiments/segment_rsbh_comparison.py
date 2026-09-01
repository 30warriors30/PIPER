from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import importlib
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from tqdm import tqdm

from utils.generation import paired_rng


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


def _bits(value: str, *, expected_length: int | None = None) -> tuple[int, ...]:
    bits = tuple(int(bit) for bit in value)
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("message bits must contain only 0 and 1")
    if expected_length is not None and len(bits) != expected_length:
        raise ValueError(f"message bits must contain exactly {expected_length} bits")
    return bits


def _bit_accuracy(expected: str, observed: str) -> float | None:
    if len(expected) != len(observed) or not expected:
        return None
    correct = sum(left == right for left, right in zip(expected, observed, strict=True))
    return correct / len(expected)


def segment_rs_scheme(message_length: int) -> tuple[int, int, int]:
    if int(message_length) == 8:
        return 3, 1, 8
    raise ValueError(
        "Unsupported Segment-RSBH message length for this comparison. "
        "The confirmed adapter only defines b=8 as RS(n=3,k=1,m=8)."
    )


def segment_payload_from_bits(bits: str) -> int:
    _bits(bits)
    return int(bits, 2) if bits else 0


def segment_payload_to_bits(payload: int, *, bit_width: int) -> str:
    if int(bit_width) <= 0:
        raise ValueError("bit_width must be positive")
    mask = (1 << int(bit_width)) - 1
    return format(int(payload) & mask, f"0{int(bit_width)}b")


def evaluate_segment_payload_prediction(
    *,
    pred_payload: int,
    expected_bits: str,
) -> dict[str, Any]:
    _bits(expected_bits)
    decoded_bits = segment_payload_to_bits(pred_payload, bit_width=len(expected_bits))
    return {
        "predicted_payload": int(pred_payload),
        "decoded_message_bits": decoded_bits,
        "original_bit_acc": _bit_accuracy(expected_bits, decoded_bits),
        "message_recovered": decoded_bits == expected_bits,
    }


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

    exact = [bool(row.get("message_recovered")) for row in watermarked]
    detected_exact = [
        bool(row.get("message_recovered")) and float(row["z_score"]) >= threshold
        for row in watermarked
    ]
    bit_acc = [
        float(row["original_bit_acc"])
        for row in watermarked
        if isinstance(row.get("original_bit_acc"), (float, int))
        and math.isfinite(float(row["original_bit_acc"]))
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
        "mean_watermarked_z_score": _mean([float(row["z_score"]) for row in watermarked]),
        "mean_negative_z_score": _mean([float(row["z_score"]) for row in negatives]),
    }


def _token_frequency_rows(rows: Iterable[dict[str, Any]]) -> Iterable[int]:
    fields = (
        "prompt_token_ids",
        "unwatermarked_token_ids",
        "natural_token_ids",
        "token_ids",
    )
    for row in rows:
        for field in fields:
            for token_id in row.get(field, []) or []:
                yield int(token_id)


def build_frequency_mapping(
    rows: Iterable[dict[str, Any]],
    *,
    vocab_size: int,
    gf_segments_num: int,
) -> dict[int, int]:
    if int(vocab_size) <= 0:
        raise ValueError("vocab_size must be positive")
    if int(gf_segments_num) <= 0:
        raise ValueError("gf_segments_num must be positive")
    counts = Counter(
        token_id
        for token_id in _token_frequency_rows(rows)
        if 0 <= int(token_id) < int(vocab_size)
    )
    frequencies = [float(counts.get(token_id, 0) + 1) for token_id in range(int(vocab_size))]
    order = sorted(range(int(vocab_size)), key=lambda token_id: (-frequencies[token_id], token_id))
    loads = [0.0 for _ in range(int(gf_segments_num))]
    sizes = [0 for _ in range(int(gf_segments_num))]
    mapping: dict[int, int] = {}
    for token_id in order:
        segment = min(range(int(gf_segments_num)), key=lambda idx: (loads[idx], sizes[idx], idx))
        mapping[token_id] = segment
        loads[segment] += frequencies[token_id]
        sizes[segment] += 1
    return mapping


def mapping_summary(mapping: dict[int, int], rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    token_counts = Counter(_token_frequency_rows(rows))
    segment_counts: Counter[int] = Counter()
    for token_id, count in token_counts.items():
        segment = mapping.get(int(token_id))
        if segment is not None:
            segment_counts[int(segment)] += int(count)
    segment_vocab_sizes = Counter(mapping.values())
    return {
        "segment_token_counts": {str(key): int(value) for key, value in sorted(segment_counts.items())},
        "segment_vocab_sizes": {str(key): int(value) for key, value in sorted(segment_vocab_sizes.items())},
    }


def import_segment_rsbh(segment_repo: Path) -> tuple[type[Any], type[Any], Any]:
    segment_repo = Path(segment_repo).resolve()
    if not segment_repo.exists():
        raise FileNotFoundError(segment_repo)
    if not (segment_repo / "wm" / "generator.py").exists():
        raise FileNotFoundError(segment_repo / "wm" / "generator.py")
    sys.path.insert(0, str(segment_repo))
    wm_module = importlib.import_module("wm")
    return (
        getattr(wm_module, "RSBHGenerator"),
        getattr(wm_module, "RSBHDecoder"),
        getattr(wm_module, "get_pvalue_segment_based"),
    )


def disable_segment_rs_debug(obj: Any) -> None:
    helper = getattr(getattr(obj, "rs", None), "helper", None)
    if helper is not None and hasattr(helper, "debug_active"):
        helper.debug_active = False


def repair_segment_gf_segments(generator: Any) -> None:
    expected = int(getattr(generator, "gf_segments_num"))
    values = [int(value) for value in getattr(generator, "gf_segments", [])]
    if len(values) > expected:
        raise RuntimeError(
            f"Segment-RSBH RS encoder returned {len(values)} GF segments; expected {expected}"
        )
    if len(values) < expected:
        values.extend([0] * (expected - len(values)))
    generator.gf_segments = values


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


def validate_runtime_args(args: argparse.Namespace) -> None:
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested, but no CUDA GPU is visible in this environment. "
            "Run on a GPU node/session or use --device cpu --dtype float32 only for tiny debugging runs."
        )
    segment_rs_scheme(args.message_length)


class SegmentRSBHLogitsProcessor:
    def __init__(self, generator: Any) -> None:
        self.generator = generator

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.ndim != 2 or scores.ndim != 2:
            raise ValueError("input_ids and scores must have shape [batch, sequence/vocabulary]")
        if input_ids.shape[0] != 1:
            raise ValueError("Segment-RSBH comparison supports batch_size=1")
        ngram = int(self.generator.ngram)
        if input_ids.shape[1] < ngram:
            return scores
        self.generator.is_slash_n = int(input_ids[0, -1].item()) == int(self.generator.newline_idx)
        ngram_tokens = input_ids[:, -ngram:]
        return self.generator.logits_processor(scores, ngram_tokens)


def _patch_newline_token(generator: Any, tokenizer: Any) -> None:
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if newline_ids:
        generator.newline_idx = int(newline_ids[-1])


def make_segment_generator(
    args: argparse.Namespace,
    *,
    model: Any,
    tokenizer: Any,
    generator_cls: type[Any],
    mapping: dict[int, int],
    payload: int = 0,
) -> Any:
    n, k, m = segment_rs_scheme(args.message_length)
    with redirect_stdout(io.StringIO()):
        generator = generator_cls(
            model,
            tokenizer,
            args.ngram,
            args.seed,
            args.seeding,
            args.hash_key,
            payload=payload,
            gamma=args.gamma,
            delta=args.delta,
            segments_num=k,
            gf_segments_num=n,
            segment_bit=m,
            bh_mapping=mapping,
            model_name="opt",
        )
    disable_segment_rs_debug(generator)
    repair_segment_gf_segments(generator)
    _patch_newline_token(generator, tokenizer)
    return generator


def make_segment_detector(
    args: argparse.Namespace,
    *,
    tokenizer: Any,
    detector_cls: type[Any],
    mapping: dict[int, int],
) -> Any:
    n, k, m = segment_rs_scheme(args.message_length)
    with redirect_stdout(io.StringIO()):
        detector = detector_cls(
            tokenizer,
            args.ngram,
            args.seed,
            args.seeding,
            args.hash_key,
            gamma=args.gamma,
            delta=args.delta,
            segments_num=k,
            gf_segments_num=n,
            segment_bit=m,
            bh_mapping=mapping,
        )
    disable_segment_rs_debug(detector)
    return detector


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


def generate_segment_watermarked(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    generator: Any,
) -> dict[str, Any]:
    message_bits = str(row["message_bits"])
    payload = segment_payload_from_bits(message_bits)
    generator.set_payload(payload)
    repair_segment_gf_segments(generator)
    generator.is_slash_n = False
    input_ids = torch.tensor([list(row["prompt_token_ids"])], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    processor = SegmentRSBHLogitsProcessor(generator)
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
    return {
        "sample_id": str(row["sample_id"]),
        "split": str(row["split"]),
        "text_class": "watermarked",
        "message_bits": message_bits,
        "payload": payload,
        "generation_seed": int(row["generation_seed"]),
        "prompt_token_ids": list(row["prompt_token_ids"]),
        "token_ids": continuation,
        "text": tokenizer.decode(continuation, skip_special_tokens=True),
        "generation_seconds": elapsed,
    }


def score_segment_token_ids(
    detector: Any,
    get_pvalue_segment_based: Any,
    *,
    token_ids: Sequence[int],
    scoring_method: str = "none",
) -> dict[str, Any]:
    tokens = [int(value) for value in token_ids]
    total_len = len(tokens)
    start_pos = int(detector.ngram) + 1
    score_lists = [[torch.zeros(2 ** int(detector.segment_bit)) for _ in range(int(detector.gf_segments_num))]]
    segment_allocation: list[int] = []
    seen_ntuples: set[tuple[int, ...]] = set()
    for cur_pos in range(start_pos, total_len):
        ngram_tokens = tokens[cur_pos - int(detector.ngram) : cur_pos]
        if scoring_method == "v1":
            unique_key = tuple(ngram_tokens)
            if unique_key in seen_ntuples:
                continue
            seen_ntuples.add(unique_key)
        elif scoring_method == "v2":
            unique_key = tuple(ngram_tokens + tokens[cur_pos : cur_pos + 1])
            if unique_key in seen_ntuples:
                continue
            seen_ntuples.add(unique_key)
        elif scoring_method != "none":
            raise ValueError(f"Unsupported scoring_method: {scoring_method}")
        rt = detector.score_tok(ngram_tokens, tokens[cur_pos])
        rt = rt[: (2 ** int(detector.segment_bit))]
        segment = int(detector.mapping[int(ngram_tokens[0])])
        score_lists[0][segment] += rt
        segment_allocation.append(segment)

    scores = [[arr.numpy() for arr in score_lists[0]]]
    num_tokens = np.asarray([max(total_len - start_pos, 0)])
    payloads = detector.get_decoded_payload(scores)
    zscores, pvalues = get_pvalue_segment_based(scores, num_tokens)
    score_value = float(sum(np.max(arr) for arr in scores[0]))
    return {
        "num_tokens_scored": int(num_tokens[0]),
        "score": score_value,
        "all_score": [arr.tolist() for arr in scores[0]],
        "predicted_payload": int(payloads[0]),
        "z_score": float(zscores[0]),
        "p_value": float(pvalues[0]),
        "tokens_segment": segment_allocation,
    }


def build_negative_records(
    baseline_path: Path,
    *,
    args: argparse.Namespace,
    detector: Any,
    get_pvalue_segment_based: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in tqdm(list(iter_jsonl(baseline_path)), desc="Score Segment-RSBH negatives", unit="sample"):
        for text_class, text_field, ids_field in (
            ("unwatermarked", "unwatermarked_text", "unwatermarked_token_ids"),
            ("natural", "natural_text", "natural_token_ids"),
        ):
            token_ids = [int(value) for value in row[ids_field]]
            score = score_segment_token_ids(
                detector,
                get_pvalue_segment_based,
                token_ids=token_ids,
                scoring_method=args.scoring_method,
            )
            payload_eval = evaluate_segment_payload_prediction(
                pred_payload=int(score["predicted_payload"]),
                expected_bits=str(row["message_bits"]),
            )
            records.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "split": str(row["split"]),
                    "text_class": text_class,
                    "message_bits": str(row["message_bits"]),
                    "token_ids": token_ids,
                    "text": str(row.get(text_field, "")),
                    **score,
                    **payload_eval,
                }
            )
    return records


def load_manifest_rows(path: Path, *, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        if str(row.get("split")) != split:
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


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


def _mapping_rows(experiment_dir: Path) -> list[dict[str, Any]]:
    return [
        row
        for row in iter_jsonl(experiment_dir / "shared" / "baseline.jsonl")
        if row.get("split") == "calibration"
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_runtime_args(args)
    experiment_dir = resolve_input_path(args.experiment_dir)
    segment_repo = resolve_input_path(args.segment_repo)
    output_dir = resolve_output_path(args.output_dir)
    records_path = output_dir / "records.jsonl"
    if records_path.exists() and not args.overwrite:
        raise FileExistsError(f"{records_path} exists; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    if records_path.exists():
        records_path.unlink()

    generator_cls, detector_cls, get_pvalue_segment_based = import_segment_rsbh(segment_repo)
    model, tokenizer, device = load_model_and_tokenizer(args)
    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))
    n, k, m = segment_rs_scheme(args.message_length)
    calibration_rows = _mapping_rows(experiment_dir)
    mapping = build_frequency_mapping(
        calibration_rows,
        vocab_size=vocab_size,
        gf_segments_num=n,
    )
    mapping_info = mapping_summary(mapping, calibration_rows)
    write_json(
        output_dir / "balance_hash_mapping.json",
        {"schema_version": 1, "mapping": {str(key): int(value) for key, value in sorted(mapping.items())}},
    )

    metadata = {
        "schema_version": 1,
        "baseline": "Segment-RSBH",
        "source_repo": "https://github.com/randomizedtree/segment-watermark",
        "segment_repo": str(segment_repo),
        "experiment_dir": str(experiment_dir),
        "model_path": args.model_path,
        "brew_point_id": args.brew_point_id,
        "exact_tokens": args.exact_tokens,
        "message_length": args.message_length,
        "payload_max": 2 ** int(args.message_length),
        "gamma": args.gamma,
        "delta": args.delta,
        "ngram": args.ngram,
        "seed": args.seed,
        "seeding": args.seeding,
        "hash_key": args.hash_key,
        "scoring_method": args.scoring_method,
        "target_fpr": args.target_fpr,
        "rs_scheme": {"n": n, "k": k, "m": m},
        "mapping_source": "calibration split shared baseline token frequencies",
        **mapping_info,
    }
    write_json(output_dir / "metadata.json", metadata)

    detector = make_segment_detector(args, tokenizer=tokenizer, detector_cls=detector_cls, mapping=mapping)
    records = build_negative_records(
        experiment_dir / "shared" / "baseline.jsonl",
        args=args,
        detector=detector,
        get_pvalue_segment_based=get_pvalue_segment_based,
    )
    generator = make_segment_generator(
        args,
        model=model,
        tokenizer=tokenizer,
        generator_cls=generator_cls,
        mapping=mapping,
    )
    test_rows = load_manifest_rows(experiment_dir / "manifest.jsonl", split="test", limit=args.limit_test)
    for row in tqdm(test_rows, desc="Generate and score Segment-RSBH positives", unit="sample"):
        generated = generate_segment_watermarked(
            row,
            args=args,
            model=model,
            tokenizer=tokenizer,
            device=device,
            generator=generator,
        )
        score = score_segment_token_ids(
            detector,
            get_pvalue_segment_based,
            token_ids=generated["token_ids"],
            scoring_method=args.scoring_method,
        )
        payload_eval = evaluate_segment_payload_prediction(
            pred_payload=int(score["predicted_payload"]),
            expected_bits=generated["message_bits"],
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
    segment_summary = summarize_records(
        test_records,
        threshold=threshold,
        target_fpr=args.target_fpr,
    )
    summary = {
        "schema_version": 1,
        "segment_rsbh": segment_summary,
        "brew": load_brew_summary(experiment_dir, args.brew_point_id),
    }
    write_json(output_dir / "metrics.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Segment-RSBH on the existing BREW T=200 b=8 data."
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--segment-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--brew-point-id", default="soft_p0_m2")
    parser.add_argument("--exact-tokens", type=int, default=200)
    parser.add_argument("--message-length", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--ngram", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeding", default="hash")
    parser.add_argument("--hash-key", type=int, default=35317)
    parser.add_argument("--scoring-method", choices=["none", "v1", "v2"], default="none")
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
