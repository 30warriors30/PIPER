import pytest
import torch

from watermark.logits_processor import DualLayerLogitsProcessor


class CountingCpuTensor(torch.Tensor):
    cpu_calls = 0

    def cpu(self, *args: object, **kwargs: object) -> torch.Tensor:
        type(self).cpu_calls += 1
        return super().cpu(*args, **kwargs)


class NoHostCopyTensor(torch.Tensor):
    def cpu(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("region tensor must not be copied to CPU")

    def tolist(self) -> list[object]:
        raise AssertionError("region tensor must not be converted to a host list")


def processor_kwargs() -> dict[str, object]:
    return {
        "secret_key": b"secret",
        "context_width": 2,
        "vocab_size": 12,
        "excluded_token_ids": {0},
        "presence_mode": "soft",
        "delta_presence": 2.0,
        "delta_payload": 3.0,
        "prf_mode": "paper_shared",
    }


def make_processor(mode: str) -> DualLayerLogitsProcessor:
    return DualLayerLogitsProcessor(
        secret_key=b"secret",
        context_width=2,
        encoded_bits=(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
        vocab_size=12,
        excluded_token_ids={0},
        presence_mode=mode,
        delta_presence=2.0,
        delta_payload=3.0,
        prf_mode="paper_shared",
    )


def make_v2_processor(
    *,
    encoded_bits_by_row: tuple[tuple[int, ...], ...],
    capture_traces: bool = True,
    mode: str = "soft",
) -> DualLayerLogitsProcessor:
    return DualLayerLogitsProcessor(
        secret_key=b"secret",
        context_width=2,
        encoded_bits_by_row=encoded_bits_by_row,
        vocab_size=12,
        excluded_token_ids={0},
        presence_mode=mode,
        delta_presence=2.0,
        delta_payload=3.0,
        prf_mode="paper_shared",
        partition_engine="v2",
        capture_traces=capture_traces,
    )


def test_hard_mode_masks_lower_and_special_tokens() -> None:
    processor = make_processor("hard")
    output = processor(torch.tensor([[2, 3]]), torch.zeros((1, 12)))
    trace = processor.last_trace
    assert trace is not None
    assert torch.isneginf(output[0, list(trace.lower_ids)]).all()
    assert torch.isneginf(output[0, 0])
    assert torch.all(output[0, list(trace.target_ids)] == 3.0)


def test_soft_mode_biases_upper_and_target() -> None:
    processor = make_processor("soft")
    output = processor(torch.tensor([[2, 3]]), torch.zeros((1, 12)))
    trace = processor.last_trace
    assert trace is not None
    assert torch.all(output[0, list(trace.non_target_upper_ids)] == 2.0)
    assert torch.all(output[0, list(trace.target_ids)] == 5.0)


def test_batch_rows_use_independent_codewords_and_contexts() -> None:
    codewords = ((0, 1, 0, 1), (1, 0, 1, 0))
    processor = make_v2_processor(encoded_bits_by_row=codewords)

    output = processor(
        torch.tensor([[2, 3], [8, 9]]),
        torch.zeros((2, 12)),
    )

    assert output.shape == (2, 12)
    assert len(processor.last_traces) == 2
    assert processor.last_traces[0].context_ids == (2, 3)
    assert processor.last_traces[1].context_ids == (8, 9)
    for row, trace in enumerate(processor.last_traces):
        assert trace.embedded_bit == codewords[row][trace.code_bit_index]
        assert torch.all(output[row, list(trace.target_ids)] == 5.0)


def test_batch_size_must_match_codeword_rows() -> None:
    processor = make_v2_processor(encoded_bits_by_row=((0, 1),))

    with pytest.raises(ValueError, match="2 score rows but 1 encoded codeword"):
        processor(
            torch.tensor([[2, 3], [4, 5]]),
            torch.zeros((2, 12)),
        )


@pytest.mark.parametrize(
    "constructor_overrides",
    [
        {},
        {
            "encoded_bits": (0, 1),
            "encoded_bits_by_row": ((0, 1),),
        },
    ],
)
def test_constructor_requires_exactly_one_codeword_source(
    constructor_overrides: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError,
        match="exactly one of encoded_bits and encoded_bits_by_row",
    ):
        DualLayerLogitsProcessor(**processor_kwargs(), **constructor_overrides)


@pytest.mark.parametrize(
    ("codewords", "message"),
    [
        ((), "at least one encoded codeword is required"),
        (((),), "encoded codewords must be non-empty"),
        (((0, 1), (1,)), "encoded codewords must have equal length"),
        (((0, 2),), "encoded codewords must be binary"),
    ],
)
def test_per_row_codewords_are_uniform_nonempty_binary_sequences(
    codewords: tuple[tuple[int, ...], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DualLayerLogitsProcessor(
            **processor_kwargs(),
            encoded_bits_by_row=codewords,
            partition_engine="v2",
        )


def test_input_and_score_row_counts_must_match() -> None:
    processor = make_v2_processor(encoded_bits_by_row=((0, 1), (1, 0)))

    with pytest.raises(ValueError, match="input_ids has 1 row but scores has 2 rows"):
        processor(torch.tensor([[2, 3]]), torch.zeros((2, 12)))


def test_partition_engine_is_validated() -> None:
    with pytest.raises(ValueError, match="partition_engine must be v1 or v2"):
        DualLayerLogitsProcessor(
            **processor_kwargs(),
            encoded_bits=(0, 1),
            partition_engine="future",
        )


def test_legacy_single_row_api_and_default_partition_remain_available() -> None:
    processor = make_processor("soft")

    output = processor(torch.tensor([[2, 3]]), torch.zeros((1, 12)))

    assert processor.last_trace is not None
    assert processor.last_traces == (processor.last_trace,)
    assert output.tolist() == [
        [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]
    ]


def test_multi_row_call_has_no_legacy_last_trace() -> None:
    processor = make_v2_processor(
        encoded_bits_by_row=((0, 1, 0, 1), (1, 0, 1, 0)),
    )

    processor(torch.tensor([[2, 3], [8, 9]]), torch.zeros((2, 12)))

    assert processor.last_trace is None


def test_paper_shared_uses_allocator_base_seed_before_round_expansion() -> None:
    processor = make_v2_processor(encoded_bits_by_row=((0, 1, 0, 1),))

    processor(torch.tensor([[2, 3]]), torch.zeros((1, 12)))

    trace = processor.last_trace
    assert trace is not None
    assert trace.partition_seed == trace.position_seed


def test_v2_hard_mode_preserves_excluded_token_behavior() -> None:
    processor = make_v2_processor(
        encoded_bits_by_row=((0, 1, 0, 1),),
        mode="hard",
    )

    output = processor(torch.tensor([[2, 3]]), torch.zeros((1, 12)))

    trace = processor.last_trace
    assert trace is not None
    assert torch.isneginf(output[0, list(trace.lower_ids)]).all()
    assert torch.isneginf(output[0, 0])
    assert torch.all(output[0, list(trace.target_ids)] == 3.0)


def test_capture_traces_false_keeps_regions_on_score_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = make_v2_processor(
        encoded_bits_by_row=((0, 1, 0, 1),),
        capture_traces=False,
    )
    real_regions_for_vocab = processor.partitioner.regions_for_vocab
    requested_devices: list[torch.device] = []

    def no_host_regions(
        *,
        round_keys_by_row: object,
        device: torch.device,
    ) -> torch.Tensor:
        requested_devices.append(device)
        regions = real_regions_for_vocab(
            round_keys_by_row=round_keys_by_row,
            device=device,
        )
        return regions.as_subclass(NoHostCopyTensor)

    monkeypatch.setattr(processor.partitioner, "regions_for_vocab", no_host_regions)
    scores = torch.zeros((1, 12))

    output = processor(torch.tensor([[2, 3]]), scores)

    assert requested_devices == [scores.device]
    assert processor.last_traces == ()
    assert processor.last_trace is None
    assert torch.count_nonzero(output == 5.0).item() == 2
    assert torch.count_nonzero(output == 2.0).item() == 2
    assert output[0, 0].item() == 0.0


def test_context_rows_use_one_cpu_transfer() -> None:
    processor = make_v2_processor(
        encoded_bits_by_row=((0, 1, 0, 1), (1, 0, 1, 0)),
        capture_traces=False,
    )
    input_ids = torch.tensor([[2, 3], [8, 9]]).as_subclass(CountingCpuTensor)
    CountingCpuTensor.cpu_calls = 0

    processor(input_ids, torch.zeros((2, 12)))

    assert CountingCpuTensor.cpu_calls == 1
