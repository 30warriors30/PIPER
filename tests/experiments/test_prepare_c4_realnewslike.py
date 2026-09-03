from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments import prepare_c4_realnewslike as prepare
from utils.datasets import SampleRecord, prepare_sample


class _WhitespaceTokenizer:
    bos_token_id = 9999

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        values = [int(value) for value in text.split()] if text else []
        if add_special_tokens:
            values = [self.bos_token_id, *values]
        return {"input_ids": values}

    def decode(self, values, *, skip_special_tokens: bool = True) -> str:
        ids = [int(value) for value in values]
        if skip_special_tokens:
            ids = [value for value in ids if value != self.bos_token_id]
        return " ".join(str(value) for value in ids)

    def build_inputs_with_special_tokens(self, values):
        return [self.bos_token_id, *[int(value) for value in values]]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 1


class _BoundaryChangingTokenizer(_WhitespaceTokenizer):
    def __call__(self, text: str, *, add_special_tokens: bool = False):
        if text.startswith("boundary "):
            values = [int(value) for value in text.removeprefix("boundary ").split()][:-1]
            return {"input_ids": values}
        return super().__call__(text, add_special_tokens=add_special_tokens)

    def decode(self, values, *, skip_special_tokens: bool = True) -> str:
        ids = [int(value) for value in values]
        text = super().decode(ids, skip_special_tokens=skip_special_tokens)
        return f"boundary {text}" if len(ids) == 1000 else text


class _SameLengthBoundaryChangingTokenizer(_WhitespaceTokenizer):
    def __call__(self, text: str, *, add_special_tokens: bool = False):
        if text.startswith("boundary "):
            values = [int(value) for value in text.removeprefix("boundary ").split()]
            values[0] += 10000
            return {"input_ids": values}
        return super().__call__(text, add_special_tokens=add_special_tokens)

    def decode(self, values, *, skip_special_tokens: bool = True) -> str:
        ids = [int(value) for value in values]
        text = super().decode(ids, skip_special_tokens=skip_special_tokens)
        return f"boundary {text}" if len(ids) == 1000 else text


class _OptLikeTokenizer(_WhitespaceTokenizer):
    def build_inputs_with_special_tokens(self, values):
        return [int(value) for value in values]


def test_prepare_c4_cli_exposes_token_length_controls() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.prepare_c4_realnewslike", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--prompt-tokens" in completed.stdout
    assert "--continuation-tokens" in completed.stdout
    assert "--records" in completed.stdout
    assert "--model-path" in completed.stdout
    assert "--output-path" in completed.stdout
    assert "--source-parquet" in completed.stdout
    assert "--overwrite" in completed.stdout


def test_build_processed_record_preserves_exact_prompt_and_continuation_tokens() -> None:
    tokenizer = _WhitespaceTokenizer()
    source = {
        "text": " ".join(str(value) for value in range(1100)),
        "timestamp": "2019-04-19 18:35:24",
        "url": "https://example.test/article",
    }

    record = prepare.build_processed_record(
        source,
        source_index=7,
        tokenizer=tokenizer,
        prompt_tokens=30,
        continuation_tokens=1000,
        model_max_length=2048,
    )

    assert record is not None
    assert record["id"] == "realnewslike-validation-00000007"
    assert record["source_index"] == 7
    assert record["prompt"] == " ".join(str(value) for value in range(29))
    assert record["natural_text"] == " ".join(str(value) for value in range(29, 1029))
    prepared = prepare_sample(
        SampleRecord(
            sample_id=record["id"],
            dataset="c4",
            prompt=record["prompt"],
            natural_completion=record["natural_text"],
        ),
        tokenizer,
        max_new_tokens=1000,
        context_width=4,
        model_max_length=2048,
    )
    assert len(prepared.prompt_token_ids or ()) == 30
    assert len(prepared.natural_token_ids or ()) == 1000


def test_build_processed_record_rejects_short_source_text() -> None:
    tokenizer = _WhitespaceTokenizer()
    source = {"text": " ".join(str(value) for value in range(1028))}

    record = prepare.build_processed_record(
        source,
        source_index=0,
        tokenizer=tokenizer,
        prompt_tokens=30,
        continuation_tokens=1000,
        model_max_length=2048,
    )

    assert record is None


def test_build_processed_record_rejects_tokenizer_round_trip_drift() -> None:
    tokenizer = _BoundaryChangingTokenizer()
    source = {"text": " ".join(str(value) for value in range(1100))}

    record = prepare.build_processed_record(
        source,
        source_index=0,
        tokenizer=tokenizer,
        prompt_tokens=30,
        continuation_tokens=1000,
        model_max_length=2048,
    )

    assert record is None


def test_build_processed_record_accepts_same_length_boundary_retokenization() -> None:
    tokenizer = _SameLengthBoundaryChangingTokenizer()
    source = {"text": " ".join(str(value) for value in range(1100))}

    record = prepare.build_processed_record(
        source,
        source_index=0,
        tokenizer=tokenizer,
        prompt_tokens=30,
        continuation_tokens=1000,
        model_max_length=2048,
    )

    assert record is not None


def test_build_processed_record_uses_actual_special_tokens_added() -> None:
    tokenizer = _OptLikeTokenizer()
    source = {"text": " ".join(str(value) for value in range(1100))}

    record = prepare.build_processed_record(
        source,
        source_index=0,
        tokenizer=tokenizer,
        prompt_tokens=30,
        continuation_tokens=1000,
        model_max_length=2048,
    )

    assert record is not None
    assert record["prompt"] == " ".join(str(value) for value in range(30))
    assert record["natural_text"] == " ".join(str(value) for value in range(30, 1030))


def test_write_processed_dataset_filters_and_records_checksum(tmp_path: Path) -> None:
    tokenizer = _WhitespaceTokenizer()
    rows = [
        (0, {"text": " ".join(str(value) for value in range(100))}),
        (1, {"text": " ".join(str(value) for value in range(1100)), "url": "u1"}),
        (2, {"text": " ".join(str(value) for value in range(1200)), "url": "u2"}),
    ]
    output = tmp_path / "c4_t1000.jsonl"
    metadata = tmp_path / "c4_t1000.metadata.json"

    summary = prepare.write_processed_dataset(
        rows,
        output_path=output,
        metadata_path=metadata,
        tokenizer=tokenizer,
        model_path="fake-opt",
        prompt_tokens=30,
        continuation_tokens=1000,
        records=2,
        model_max_length=2048,
    )

    payload = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    saved_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert [row["source_index"] for row in payload] == [1, 2]
    assert summary["scanned_rows"] == 3
    assert summary["selected_records"] == 2
    assert saved_metadata["sha256"] == digest
    assert saved_metadata["prompt_tokens"] == 30
    assert saved_metadata["continuation_tokens"] == 1000


def test_iter_dataset_rows_paginates_until_total() -> None:
    calls: list[tuple[int, int]] = []

    def fetch_page(offset: int, length: int):
        calls.append((offset, length))
        all_rows = [
            {"row_idx": index, "row": {"text": str(index)}}
            for index in range(5)
        ]
        return {
            "rows": all_rows[offset : offset + length],
            "num_rows_total": len(all_rows),
        }

    rows = list(prepare.iter_dataset_rows(fetch_page, page_size=2))

    assert rows == [(index, {"text": str(index)}) for index in range(5)]
    assert calls == [(0, 2), (2, 2), (4, 2)]


def test_iter_parquet_rows_preserves_source_indices(tmp_path: Path) -> None:
    path = tmp_path / "validation.parquet"
    pq.write_table(
        pa.table(
            {
                "text": ["first", "second"],
                "timestamp": ["t0", "t1"],
                "url": ["u0", "u1"],
            }
        ),
        path,
    )

    rows = list(prepare.iter_parquet_rows(path))

    assert rows == [
        (0, {"text": "first", "timestamp": "t0", "url": "u0"}),
        (1, {"text": "second", "timestamp": "t1", "url": "u1"}),
    ]
