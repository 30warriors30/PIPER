import math
from types import SimpleNamespace

import pytest
import torch

from experiments.external_quality import (
    batched_continuation_nll,
    evaluator_slug,
    verify_tokenizer_compatibility,
)


class UniformCausalModel:
    def __init__(self, vocab_size: int = 7):
        self.config = SimpleNamespace(vocab_size=vocab_size, max_position_embeddings=128)

    def __call__(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(batch, length, self.config.vocab_size, device=input_ids.device))


class FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = dict(vocab)
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 2
        self.unk_token_id = 3

    def get_vocab(self):
        return dict(self._vocab)


def test_batched_continuation_nll_excludes_prompt_and_handles_variable_lengths() -> None:
    model = UniformCausalModel(vocab_size=7)
    examples = [
        {"sample_id": "a", "prompt_token_ids": [1, 2, 3], "continuation_token_ids": [4, 5]},
        {"sample_id": "b", "prompt_token_ids": [1], "continuation_token_ids": [2, 3, 4, 5]},
    ]

    rows = batched_continuation_nll(model, examples, device=torch.device("cpu"), pad_token_id=0)

    assert [row["sample_id"] for row in rows] == ["a", "b"]
    assert [row["token_count"] for row in rows] == [2, 4]
    assert math.isclose(rows[0]["nll"], 2 * math.log(7), rel_tol=1e-6)
    assert math.isclose(rows[1]["nll"], 4 * math.log(7), rel_tol=1e-6)
    assert math.isclose(rows[0]["mean_nll"], math.log(7), rel_tol=1e-6)
    assert math.isclose(rows[1]["ppl"], 7.0, rel_tol=1e-6)


def test_batched_continuation_nll_matches_single_item_batches() -> None:
    model = UniformCausalModel(vocab_size=5)
    examples = [
        {"sample_id": "a", "prompt_token_ids": [1, 2], "continuation_token_ids": [3, 4, 1]},
        {"sample_id": "b", "prompt_token_ids": [1, 2, 3, 4], "continuation_token_ids": [2, 2]},
    ]

    batch = batched_continuation_nll(model, examples, device=torch.device("cpu"), pad_token_id=0)
    singles = [
        batched_continuation_nll(model, [example], device=torch.device("cpu"), pad_token_id=0)[0]
        for example in examples
    ]

    assert [row["token_count"] for row in batch] == [row["token_count"] for row in singles]
    assert [row["nll"] for row in batch] == pytest.approx([row["nll"] for row in singles])


def test_verify_tokenizer_compatibility_requires_same_token_id_mapping() -> None:
    generation = FakeTokenizer({"a": 0, "b": 1, "c": 2})
    evaluator = FakeTokenizer({"a": 0, "b": 1, "c": 2})
    verify_tokenizer_compatibility(generation, evaluator)

    incompatible = FakeTokenizer({"a": 0, "b": 2, "c": 1})
    with pytest.raises(ValueError, match="token-ID compatible"):
        verify_tokenizer_compatibility(generation, incompatible)


def test_evaluator_slug_is_stable_and_filesystem_safe() -> None:
    assert evaluator_slug("/models/facebook/opt-6.7b") == "opt-6_7b"
    assert evaluator_slug("/models/facebook/opt-13b", "OPT 13B evaluator") == "opt-13b-evaluator"
