import torch
from watermark.logits_processor import DualLayerLogitsProcessor


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
