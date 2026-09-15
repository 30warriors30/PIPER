from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

import torch
from tqdm import tqdm

from utils.batched_generation import adaptive_batches, generate_exact_batch, is_cuda_oom
from utils.datasets import SampleFilterError, SampleRecord, load_samples, prepare_sample
from utils.environment import environment_snapshot
from utils.io import RunStore, append_jsonl, write_json
from utils.logging import configure_logging
from utils.model import excluded_token_ids, load_model_and_tokenizer, model_max_length
from utils.seeds import derive_seed, message_bits
from watermark.config import ExperimentConfig
from watermark.ecc import BCHCodec
from watermark.logits_processor import DualLayerLogitsProcessor


@contextmanager
def paired_rng(seed: int, device: torch.device) -> Iterator[None]:
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield


@dataclass(frozen=True)
class GenerationPair:
    prompt_ids: tuple[int, ...]
    message_bits: tuple[int, ...]
    encoded_bits: tuple[int, ...]
    generation_seed: int
    watermarked_ids: tuple[int, ...]
    unwatermarked_ids: tuple[int, ...]
    watermarked_text: str
    unwatermarked_text: str
    watermarked_seconds: float
    unwatermarked_seconds: float
    ecc_encode_seconds: float
    generation_batch_size: int = 1


class PairedGenerator:
    def __init__(
        self,
        config: ExperimentConfig,
        codec: BCHCodec,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.codec = codec
        if model is None or tokenizer is None or device is None:
            loaded_model, loaded_tokenizer, loaded_device = load_model_and_tokenizer(
                config.model
            )
            self.model = loaded_model if model is None else model
            self.tokenizer = loaded_tokenizer if tokenizer is None else tokenizer
            self.device = loaded_device if device is None else device
        else:
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
        self.vocab_size = int(
            getattr(self.model.config, "vocab_size", len(self.tokenizer))
        )
        self.excluded_ids = excluded_token_ids(
            self.tokenizer,
            exclude_special_tokens=config.watermark.exclude_special_tokens,
            exclude_eos=config.watermark.exclude_eos,
        )

    def _generation_kwargs(self) -> dict[str, Any]:
        exact_length = int(self.config.generation.max_new_tokens)
        kwargs: dict[str, Any] = {
            "min_new_tokens": exact_length,
            "max_new_tokens": exact_length,
            "do_sample": self.config.generation.do_sample,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        if self.config.generation.do_sample:
            kwargs["temperature"] = self.config.generation.temperature
            kwargs["top_p"] = self.config.generation.top_p
            kwargs["top_k"] = self.config.generation.top_k
        return {key: value for key, value in kwargs.items() if value is not None}

    def _generate_once(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        seed: int,
        processor: Any | None,
    ) -> tuple[torch.Tensor, float]:
        kwargs = self._generation_kwargs()
        if processor is not None:
            try:
                from transformers import LogitsProcessorList

                kwargs["logits_processor"] = LogitsProcessorList([processor])
            except ImportError:  # Allows dependency-light injected-model tests.
                kwargs["logits_processor"] = [processor]
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        started = time.perf_counter()
        with paired_rng(seed, self.device), torch.inference_mode():
            output = self.model.generate(input_ids=input_ids, **kwargs)
        elapsed = time.perf_counter() - started
        if not isinstance(output, torch.Tensor):
            output = output.sequences
        return output, elapsed

    def generate(
        self, sample: SampleRecord, payload: tuple[int, ...]
    ) -> GenerationPair:
        if len(payload) != self.codec.k:
            raise ValueError(f"payload must contain exactly {self.codec.k} bits")
        if sample.prompt_token_ids is None:
            raise ValueError(
                "sample.prompt_token_ids must be prepared before generation"
            )

        started = time.perf_counter()
        encoded = self.codec.encode(payload)
        ecc_seconds = time.perf_counter() - started

        input_ids = torch.tensor(
            [sample.prompt_token_ids], dtype=torch.long, device=self.device
        )
        attention_mask = torch.ones_like(input_ids)
        prompt_ids = tuple(int(value) for value in sample.prompt_token_ids)
        seed = derive_seed(
            self.config.generation.global_seed, sample.dataset, sample.sample_id
        )
        processor = DualLayerLogitsProcessor(
            secret_key=self.config.watermark.secret_key.encode("utf-8"),
            context_width=self.config.watermark.context_width,
            encoded_bits=encoded,
            vocab_size=self.vocab_size,
            excluded_token_ids=self.excluded_ids,
            presence_mode=self.config.watermark.presence_mode,
            delta_presence=self.config.watermark.delta_presence,
            delta_payload=self.config.watermark.delta_payload,
            prf_mode=self.config.watermark.prf_mode,
            partition_engine=self.config.watermark.partition_engine,
            seeding_scheme=self.config.watermark.seeding_scheme,
            candidate_top_k=self.config.watermark.candidate_top_k,
        )
        unwm_output, unwm_seconds = self._generate_once(
            input_ids, attention_mask, seed, None
        )
        wm_output, wm_seconds = self._generate_once(
            input_ids, attention_mask, seed, processor
        )
        prompt_length = input_ids.shape[1]
        unwm_ids = tuple(
            int(value) for value in unwm_output[0, prompt_length:].tolist()
        )
        wm_ids = tuple(int(value) for value in wm_output[0, prompt_length:].tolist())
        expected = int(self.config.generation.max_new_tokens)
        if len(wm_ids) != expected or len(unwm_ids) != expected:
            raise RuntimeError(
                "Exact-length generation invariant failed: "
                f"watermarked={len(wm_ids)}, unwatermarked={len(unwm_ids)}, expected={expected}"
            )
        return GenerationPair(
            prompt_ids=prompt_ids,
            message_bits=payload,
            encoded_bits=encoded,
            generation_seed=seed,
            watermarked_ids=wm_ids,
            unwatermarked_ids=unwm_ids,
            watermarked_text=self.tokenizer.decode(wm_ids, skip_special_tokens=True),
            unwatermarked_text=self.tokenizer.decode(
                unwm_ids, skip_special_tokens=True
            ),
            watermarked_seconds=wm_seconds,
            unwatermarked_seconds=unwm_seconds,
            ecc_encode_seconds=ecc_seconds,
            generation_batch_size=1,
        )

    def generate_batch(
        self,
        samples: list[SampleRecord] | tuple[SampleRecord, ...],
        payloads: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
    ) -> tuple[GenerationPair, ...]:
        if len(samples) != len(payloads):
            raise ValueError("samples and payloads must have the same length")
        if not samples:
            return ()
        if len(samples) == 1 and not callable(self.model):
            return (self.generate(samples[0], payloads[0]),)

        encoded_batch: list[tuple[int, ...]] = []
        ecc_seconds: list[float] = []
        seeds: list[int] = []
        prompts: list[tuple[int, ...]] = []
        for sample, payload in zip(samples, payloads, strict=True):
            if len(payload) != self.codec.k:
                raise ValueError(f"payload must contain exactly {self.codec.k} bits")
            if sample.prompt_token_ids is None:
                raise ValueError(
                    "sample.prompt_token_ids must be prepared before generation"
                )
            started = time.perf_counter()
            encoded_batch.append(self.codec.encode(payload))
            ecc_seconds.append(time.perf_counter() - started)
            prompts.append(tuple(int(value) for value in sample.prompt_token_ids))
            seeds.append(
                derive_seed(
                    self.config.generation.global_seed, sample.dataset, sample.sample_id
                )
            )

        common = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "device": self.device,
            "prompt_token_ids": prompts,
            "seeds": seeds,
            "exact_tokens": int(self.config.generation.max_new_tokens),
            "temperature": self.config.generation.temperature,
            "top_p": self.config.generation.top_p,
            "top_k": self.config.generation.top_k,
            "do_sample": self.config.generation.do_sample,
        }
        unwatermarked = generate_exact_batch(processor=None, **common)
        processor = DualLayerLogitsProcessor(
            secret_key=self.config.watermark.secret_key.encode("utf-8"),
            context_width=self.config.watermark.context_width,
            encoded_bits_by_row=tuple(encoded_batch),
            vocab_size=self.vocab_size,
            excluded_token_ids=self.excluded_ids,
            presence_mode=self.config.watermark.presence_mode,
            delta_presence=self.config.watermark.delta_presence,
            delta_payload=self.config.watermark.delta_payload,
            prf_mode=self.config.watermark.prf_mode,
            partition_engine=self.config.watermark.partition_engine,
            capture_traces=False,
            seeding_scheme=self.config.watermark.seeding_scheme,
            candidate_top_k=self.config.watermark.candidate_top_k,
        )
        watermarked = generate_exact_batch(processor=processor, **common)

        pairs = []
        for index, (sample, payload) in enumerate(zip(samples, payloads, strict=True)):
            wm = watermarked[index]
            unwm = unwatermarked[index]
            pairs.append(
                GenerationPair(
                    prompt_ids=prompts[index],
                    message_bits=payload,
                    encoded_bits=encoded_batch[index],
                    generation_seed=seeds[index],
                    watermarked_ids=wm.token_ids,
                    unwatermarked_ids=unwm.token_ids,
                    watermarked_text=wm.text,
                    unwatermarked_text=unwm.text,
                    watermarked_seconds=wm.seconds,
                    unwatermarked_seconds=unwm.seconds,
                    ecc_encode_seconds=ecc_seconds[index],
                    generation_batch_size=wm.batch_size,
                )
            )
        return tuple(pairs)


def generation_summary(config: ExperimentConfig) -> str:
    return "\n".join(
        [
            "=" * 68,
            "Dual-Layer Multi-bit Watermark Generation",
            "=" * 68,
            f"Model:          {config.model.path}",
            f"Device/dtype:   {config.model.device} / {config.model.dtype}",
            f"Dataset:        {config.dataset.name} / {config.dataset.config} / {config.dataset.split}",
            f"Samples:        {config.dataset.max_samples} completed triples",
            f"Exact tokens:   {config.generation.max_new_tokens} per text class",
            f"Presence:       {config.watermark.presence_mode}",
            f"PRF:            {config.watermark.prf_mode}",
            f"Partition:      {config.watermark.partition_mode}",
            f"Partition seed: {config.watermark.seeding_scheme}",
            f"Engine:         {config.watermark.partition_engine}",
            f"Raw candidates: {config.watermark.candidate_top_k}",
            f"Sampling top-k: {config.generation.top_k}",
            f"Sampling top-p: {config.generation.top_p}",
            f"Allocation:     {config.watermark.allocation_mode}",
            f"BCH:            ({config.ecc.n}, {config.ecc.k}, t={config.ecc.t})",
            f"Context width:  {config.watermark.context_width}",
            f"Output:         {config.run_dir}",
            "=" * 68,
        ]
    )


def _completed_record(
    sample: SampleRecord,
    pair: GenerationPair,
    *,
    exact_length: int,
) -> dict[str, Any]:
    if sample.natural_token_ids is None:
        raise RuntimeError("Prepared sample has no natural_token_ids")
    lengths = {
        "watermarked": len(pair.watermarked_ids),
        "unwatermarked": len(pair.unwatermarked_ids),
        "natural": len(sample.natural_token_ids),
    }
    if any(length != exact_length for length in lengths.values()):
        raise RuntimeError(
            f"Three-class exact-length invariant failed: {lengths}, expected={exact_length}"
        )
    return {
        "schema_version": 1,
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "status": "completed",
        "prompt": sample.prompt,
        "prompt_token_ids": list(pair.prompt_ids),
        "message_bits": "".join(map(str, pair.message_bits)),
        "encoded_bits": "".join(map(str, pair.encoded_bits)),
        "generation_seed": pair.generation_seed,
        "watermarked": {
            "text": pair.watermarked_text,
            "token_ids": list(pair.watermarked_ids),
            "num_tokens": len(pair.watermarked_ids),
            "generation_time_seconds": pair.watermarked_seconds,
        },
        "unwatermarked": {
            "text": pair.unwatermarked_text,
            "token_ids": list(pair.unwatermarked_ids),
            "num_tokens": len(pair.unwatermarked_ids),
            "generation_time_seconds": pair.unwatermarked_seconds,
        },
        "natural": {
            "text": sample.natural_completion or "",
            "token_ids": list(sample.natural_token_ids),
            "num_tokens": len(sample.natural_token_ids),
        },
        "ecc_encode_seconds": pair.ecc_encode_seconds,
        "metadata": sample.metadata,
    }


def run_generation(
    config: ExperimentConfig,
    *,
    resume: bool = False,
    overwrite: bool = False,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: torch.device | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    target = int(config.dataset.max_samples)
    exact_length = int(config.generation.max_new_tokens)
    if target <= 0:
        raise ValueError("max_samples must be positive")
    if exact_length <= 0:
        raise ValueError("max_new_tokens must be positive")
    if batch_size is None:
        batch_size = int(config.execution.generation_batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    codec = BCHCodec(config.ecc.n, config.ecc.k, config.ecc.t)
    experiment = config.to_dict()
    experiment["generation_runtime"] = {
        "exact_continuation_tokens": exact_length,
        "force_exact_length": True,
    }
    experiment["ecc_runtime"] = {
        "n": codec.n,
        "k": codec.k,
        "t": codec.t,
        "d": codec.d,
        "parent_n": codec.parent_n,
        "parent_k": codec.parent_k,
        "backend": codec.backend,
    }
    store = RunStore.initialize(
        config.run_dir, experiment, resume=resume, overwrite=overwrite
    )
    write_json(config.run_dir / "environment.json", environment_snapshot())
    logger = configure_logging(
        store.logs_dir / "generation.log", f"generation.{config.output.run_id}"
    )
    completed = store.completed_sample_ids()
    if len(completed) >= target:
        return {
            "run_dir": str(config.run_dir),
            "processed": 0,
            "skipped": 0,
            "filtered": 0,
            "failed": 0,
        }

    generator = PairedGenerator(
        config, codec, model=model, tokenizer=tokenizer, device=device
    )
    max_length = model_max_length(generator.model, generator.tokenizer)
    if max_length < config.watermark.context_width + exact_length:
        raise ValueError(
            f"Model context length {max_length} cannot fit context width "
            f"{config.watermark.context_width} plus {exact_length} continuation tokens"
        )

    processed = skipped = filtered = failed = candidates_seen = 0
    pending: list[SampleRecord] = []
    for original in load_samples(config.dataset):
        if len(completed) + len(pending) >= target:
            break
        candidates_seen += 1
        if original.sample_id in completed:
            skipped += 1
            continue
        try:
            sample = prepare_sample(
                original,
                generator.tokenizer,
                max_new_tokens=exact_length,
                context_width=config.watermark.context_width,
                model_max_length=max_length,
            )
        except SampleFilterError as exc:
            filtered += 1
            append_jsonl(
                store.errors_dir / "filtered_samples.jsonl",
                {
                    "schema_version": 1,
                    "sample_id": original.sample_id,
                    "dataset": original.dataset,
                    "status": "filtered",
                    "reason": exc.reason,
                    "raw_token_count": exc.raw_token_count,
                    "required_token_count": exc.required_token_count,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            continue
        pending.append(sample)

    progress = tqdm(
        total=target,
        initial=min(len(completed), target),
        desc="Generating",
        unit="sample",
    )

    def generate_batch(chunk: list[SampleRecord] | tuple[SampleRecord, ...]):
        payloads = tuple(
            message_bits(config.generation.message_seed, sample.sample_id, codec.k)
            for sample in chunk
        )
        try:
            return generator.generate_batch(tuple(chunk), payloads)
        except RuntimeError as exc:
            if is_cuda_oom(exc):
                raise
        except Exception:
            pass

        outcomes: list[GenerationPair | Exception] = []
        for sample, payload in zip(chunk, payloads, strict=True):
            try:
                outcomes.append(generator.generate(sample, payload))
            except Exception as exc:  # Preserve per-sample error reporting.
                outcomes.append(exc)
        return outcomes

    try:
        for batch in adaptive_batches(
            pending, batch_size=batch_size, run=generate_batch
        ):
            for sample, outcome in zip(batch.items, batch.results, strict=True):
                try:
                    if isinstance(outcome, Exception):
                        raise outcome
                    pair = outcome
                    if not isinstance(pair, GenerationPair):
                        raise TypeError(
                            f"Unexpected generation result: {type(pair).__name__}"
                        )
                    record = _completed_record(sample, pair, exact_length=exact_length)
                    record["generation_batch_size_configured"] = batch_size
                    record["generation_batch_size_actual"] = batch.batch_size
                    append_jsonl(store.samples_path, record)
                    processed += 1
                    progress.update(1)
                except Exception as exc:
                    failed += 1
                    error = {
                        "schema_version": 1,
                        "sample_id": sample.sample_id,
                        "dataset": sample.dataset,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback_tail": "\n".join(
                            traceback.format_exc().splitlines()[-20:]
                        ),
                    }
                    append_jsonl(store.samples_path, error)
                    append_jsonl(store.errors_dir / "generation_errors.jsonl", error)
                    logger.exception("Sample %s failed", sample.sample_id)
                progress.set_postfix(filtered=filtered, failed=failed, skipped=skipped)
    finally:
        progress.close()

    completed_total = len(completed) + processed
    if completed_total < target:
        raise RuntimeError(
            "Dataset exhausted before reaching the requested completed-sample target: "
            f"requested={target}, completed={completed_total}, filtered={filtered}, "
            f"failed={failed}, candidates_seen={candidates_seen}"
        )
    return {
        "run_dir": str(config.run_dir),
        "processed": processed,
        "skipped": skipped,
        "filtered": filtered,
        "failed": failed,
    }
