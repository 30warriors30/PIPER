from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from utils.batched_generation import (
    adaptive_batches,
    generate_exact_batch,
)


class FakeCausalModel:
    """Small deterministic decoder-only model with a usable KV-cache contract."""

    def __init__(self, vocab_size: int = 12) -> None:
        self.config = SimpleNamespace(vocab_size=vocab_size, is_encoder_decoder=False)
        self.forward_batch_sizes: list[int] = []

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values=None,
        use_cache: bool,
        return_dict: bool,
    ):
        del attention_mask, use_cache, return_dict
        self.forward_batch_sizes.append(int(input_ids.shape[0]))
        batch, length = input_ids.shape
        base = torch.arange(self.config.vocab_size, dtype=torch.float32)
        logits = base.repeat(batch, length, 1)
        return SimpleNamespace(logits=logits, past_key_values=(past_key_values, length))


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(value)) for value in ids)


def _generate(model: FakeCausalModel, prompts, seeds):
    return generate_exact_batch(
        model=model,
        tokenizer=FakeTokenizer(),
        device=torch.device("cpu"),
        prompt_token_ids=prompts,
        seeds=seeds,
        exact_tokens=5,
        temperature=1.0,
        top_p=0.95,
        processor=None,
    )


def test_generate_exact_batch_uses_one_model_batch_and_exact_lengths() -> None:
    model = FakeCausalModel()

    results = _generate(model, [[2, 3], [4, 5, 6]], [10, 20])

    assert len(results) == 2
    assert all(len(result.token_ids) == 5 for result in results)
    assert model.forward_batch_sizes == [2, 2, 2, 2, 2]
    assert all(result.batch_size == 2 for result in results)


def test_per_sample_rng_is_independent_of_batch_grouping() -> None:
    batched = _generate(FakeCausalModel(), [[2, 3], [4, 5, 6]], [10, 20])
    first = _generate(FakeCausalModel(), [[2, 3]], [10])
    second = _generate(FakeCausalModel(), [[4, 5, 6]], [20])

    assert batched[0].token_ids == first[0].token_ids
    assert batched[1].token_ids == second[0].token_ids


def test_adaptive_batches_halves_only_for_cuda_oom() -> None:
    calls: list[int] = []

    def run(chunk):
        calls.append(len(chunk))
        if len(chunk) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return [value * 10 for value in chunk]

    completed = list(adaptive_batches([1, 2, 3, 4, 5], batch_size=4, run=run))

    assert calls == [4, 2, 2, 1]
    assert [item for batch in completed for item in batch.results] == [
        10,
        20,
        30,
        40,
        50,
    ]
    assert [batch.batch_size for batch in completed] == [2, 2, 1]


def test_adaptive_batches_does_not_hide_non_oom_errors() -> None:
    with pytest.raises(RuntimeError, match="bad logits"):
        list(
            adaptive_batches(
                [1, 2],
                batch_size=2,
                run=lambda chunk: (_ for _ in ()).throw(RuntimeError("bad logits")),
            )
        )
