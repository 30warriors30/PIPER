from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from utils.datasets import SampleFilterError, SampleRecord, prepare_sample, tokenize_without_specials


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT.parent / "models" / "facebook" / "opt-1.3b"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT.parent / "dataset" / "c4" / "c4_realnewslike_validation_t1000.jsonl"
)
DATASET_SERVER = "https://datasets-server.huggingface.co"


def iter_dataset_rows(
    fetch_page: Callable[[int, int], dict[str, Any]],
    *,
    page_size: int = 100,
) -> Iterator[tuple[int, dict[str, Any]]]:
    offset = 0
    while True:
        payload = fetch_page(offset, int(page_size))
        page = payload.get("rows", [])
        for item in page:
            yield int(item["row_idx"]), dict(item["row"])
        offset += len(page)
        total = int(payload.get("num_rows_total", offset))
        if not page or offset >= total:
            return


def iter_parquet_rows(path: Path, *, batch_size: int = 256) -> Iterator[tuple[int, dict[str, Any]]]:
    import pyarrow.parquet as pq

    source_index = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=int(batch_size),
        columns=["text", "timestamp", "url"],
    ):
        for row in batch.to_pylist():
            yield source_index, dict(row)
            source_index += 1


def _special_token_count(tokenizer: Any) -> int:
    builder = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if callable(builder):
        probe = [0]
        return max(0, len(builder(probe)) - len(probe))
    counter = getattr(tokenizer, "num_special_tokens_to_add", None)
    if callable(counter):
        return int(counter(pair=False))
    return 0


def build_processed_record(
    source: dict[str, Any],
    *,
    source_index: int,
    tokenizer: Any,
    prompt_tokens: int,
    continuation_tokens: int,
    model_max_length: int,
) -> dict[str, Any] | None:
    prompt_content_tokens = int(prompt_tokens) - _special_token_count(tokenizer)
    if prompt_content_tokens <= 0:
        raise ValueError("prompt_tokens must exceed the tokenizer's special-token count")
    raw_ids = tokenize_without_specials(tokenizer, str(source.get("text", "")))
    required = prompt_content_tokens + int(continuation_tokens)
    if len(raw_ids) < required:
        return None

    prompt_source_ids = raw_ids[:prompt_content_tokens]
    expected_natural_ids = raw_ids[prompt_content_tokens:required]
    sample_id = f"realnewslike-validation-{int(source_index):08d}"
    candidate = SampleRecord(
        sample_id=sample_id,
        dataset="c4",
        prompt=tokenizer.decode(prompt_source_ids, skip_special_tokens=True),
        natural_completion=tokenizer.decode(expected_natural_ids, skip_special_tokens=True),
    )
    try:
        prepared = prepare_sample(
            candidate,
            tokenizer,
            max_new_tokens=int(continuation_tokens),
            context_width=4,
            model_max_length=int(model_max_length),
        )
    except SampleFilterError:
        return None
    if len(prepared.prompt_token_ids or ()) != int(prompt_tokens):
        return None
    if len(prepared.natural_token_ids or ()) != int(continuation_tokens):
        return None
    return {
        "id": sample_id,
        "source_index": int(source_index),
        "prompt": prepared.prompt,
        "natural_text": prepared.natural_completion,
        "url": str(source.get("url", "")),
        "timestamp": str(source.get("timestamp", "")),
    }


def write_processed_dataset(
    rows: Iterable[tuple[int, dict[str, Any]]],
    *,
    output_path: Path,
    metadata_path: Path,
    tokenizer: Any,
    model_path: str,
    prompt_tokens: int,
    continuation_tokens: int,
    records: int,
    model_max_length: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f"{output_path.name}.tmp")
    scanned = selected = 0
    try:
        with temporary_output.open("w", encoding="utf-8") as handle:
            for source_index, source in rows:
                scanned += 1
                record = build_processed_record(
                    source,
                    source_index=source_index,
                    tokenizer=tokenizer,
                    prompt_tokens=prompt_tokens,
                    continuation_tokens=continuation_tokens,
                    model_max_length=model_max_length,
                )
                if record is None:
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                selected += 1
                if selected >= int(records):
                    break
        if selected < int(records):
            raise RuntimeError(
                f"Dataset exhausted after {scanned} rows with {selected} valid records; "
                f"required {records}"
            )
        os.replace(temporary_output, output_path)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    summary = {
        "schema_version": 1,
        "dataset": "allenai/c4",
        "config": "realnewslike",
        "split": "validation",
        "model_path": str(model_path),
        "prompt_tokens": int(prompt_tokens),
        "continuation_tokens": int(continuation_tokens),
        "requested_records": int(records),
        "selected_records": selected,
        "scanned_rows": scanned,
        "model_max_length": int(model_max_length),
        "sha256": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary_metadata = metadata_path.with_name(f"{metadata_path.name}.tmp")
    temporary_metadata.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_path)
    return summary


def _dataset_viewer_fetcher(*, timeout: float, retries: int):
    def fetch_page(offset: int, length: int) -> dict[str, Any]:
        query = urlencode(
            {
                "dataset": "allenai/c4",
                "config": "realnewslike",
                "split": "validation",
                "offset": int(offset),
                "length": int(length),
            }
        )
        request = Request(
            f"{DATASET_SERVER}/rows?{query}",
            headers={"User-Agent": "PIPER-C4-preparer/1.0"},
        )
        for attempt in range(int(retries) + 1):
            try:
                with urlopen(request, timeout=float(timeout)) as response:
                    payload = json.load(response)
                print(f"Fetched rows {offset}..{offset + len(payload.get('rows', [])) - 1}", flush=True)
                return payload
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt >= int(retries):
                    raise RuntimeError(
                        f"Dataset Viewer request failed at offset {offset} after {retries + 1} attempts"
                    ) from exc
                time.sleep(min(2**attempt, 10))
        raise AssertionError("unreachable")

    return fetch_page


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare long C4 realnewslike validation samples.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--metadata-path")
    parser.add_argument("--source-parquet")
    parser.add_argument("--prompt-tokens", type=int, default=30)
    parser.add_argument("--continuation-tokens", type=int, default=1000)
    parser.add_argument("--records", type=int, default=1200)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prompt_tokens <= 0 or args.continuation_tokens <= 0 or args.records <= 0:
        raise ValueError("token counts and records must be positive")
    if not 1 <= args.page_size <= 100:
        raise ValueError("page-size must be in [1, 100]")
    if args.timeout <= 0 or args.retries < 0:
        raise ValueError("timeout must be positive and retries must be non-negative")

    output_path = Path(args.output_path).resolve()
    metadata_path = (
        Path(args.metadata_path).resolve()
        if args.metadata_path
        else output_path.with_suffix(".metadata.json")
    )
    existing = [path for path in (output_path, metadata_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists: {existing[0]}. Use --overwrite to replace it.")

    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    model_max_length = int(model_config.max_position_embeddings)
    source_parquet = Path(args.source_parquet).resolve() if args.source_parquet else None
    if source_parquet is not None and not source_parquet.is_file():
        raise FileNotFoundError(source_parquet)
    source_label = str(source_parquet) if source_parquet else "allenai/c4 realnewslike validation"
    print(f"Source:       {source_label}", flush=True)
    print(f"Target:       {args.records} records", flush=True)
    print(f"Token shape:  prompt={args.prompt_tokens}, continuation={args.continuation_tokens}", flush=True)
    print(f"Output:       {output_path}", flush=True)

    if source_parquet is not None:
        rows = iter_parquet_rows(source_parquet)
    else:
        fetch_page = _dataset_viewer_fetcher(timeout=args.timeout, retries=args.retries)
        rows = iter_dataset_rows(fetch_page, page_size=args.page_size)
    summary = write_processed_dataset(
        rows,
        output_path=output_path,
        metadata_path=metadata_path,
        tokenizer=tokenizer,
        model_path=args.model_path,
        prompt_tokens=args.prompt_tokens,
        continuation_tokens=args.continuation_tokens,
        records=args.records,
        model_max_length=model_max_length,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
