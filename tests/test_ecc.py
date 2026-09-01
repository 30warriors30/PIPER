import pytest

from watermark.ecc import BCHCodec


MESSAGE = (1, 0, 1, 1, 0, 0, 1, 0)


def test_bch_round_trip_and_three_error_correction() -> None:
    codec = BCHCodec(23, 8, 3)
    codeword = list(codec.encode(MESSAGE))
    for index in (0, 8, 17):
        codeword[index] ^= 1
    result = codec.decode(codeword)
    assert result.status == "decoded"
    assert result.message_bits == MESSAGE
    assert result.codeword_bits == codec.encode(MESSAGE)


@pytest.mark.parametrize(
    ("error_positions", "erasure_positions"),
    [
        ((0, 8), (2, 12)),
        ((0,), (2, 6, 12, 20)),
        ((), (1, 4, 7, 10, 13, 19)),
    ],
)
def test_bounded_error_erasure_decoder_recovers_inside_unique_radius(
    error_positions: tuple[int, ...],
    erasure_positions: tuple[int, ...],
) -> None:
    codec = BCHCodec(23, 8, 3)
    decisions: list[int | None] = list(codec.encode(MESSAGE))
    for index in error_positions:
        decisions[index] = 1 - int(decisions[index])
    for index in erasure_positions:
        decisions[index] = None

    result = codec.decode_errors_and_erasures(decisions, max_assignments=64)

    assert result.status == "decoded"
    assert result.message_bits == MESSAGE
    assert result.codeword_bits == codec.encode(MESSAGE)
    assert result.erasure_count == len(erasure_positions)
    assert result.known_position_errors == len(error_positions)
    assert result.distance_cost == 2 * len(error_positions) + len(erasure_positions)
    assert result.candidate_count == 1
    assert result.assignments_tested == 2 ** len(erasure_positions)


def test_error_erasure_decoder_rejects_search_outside_distance_bound() -> None:
    codec = BCHCodec(23, 8, 3)
    decisions: list[int | None] = list(codec.encode(MESSAGE))
    for index in range(codec.d):
        decisions[index] = None

    result = codec.decode_errors_and_erasures(decisions, max_assignments=128)

    assert result.status == "too_many_erasures"
    assert result.erasure_count == codec.d
    assert result.assignments_tested == 0


def test_error_erasure_decoder_obeys_assignment_cap() -> None:
    codec = BCHCodec(23, 8, 3)
    decisions: list[int | None] = list(codec.encode(MESSAGE))
    for index in range(6):
        decisions[index] = None

    result = codec.decode_errors_and_erasures(decisions, max_assignments=32)

    assert result.status == "too_many_erasures"
    assert result.candidate_count == 0
    assert result.assignments_tested == 0


def test_error_erasure_decoder_uses_codebook_without_backend_decode(monkeypatch) -> None:
    codec = BCHCodec(23, 8, 3)
    decisions: list[int | None] = list(codec.encode(MESSAGE))
    decisions[0] = 1 - int(decisions[0])
    decisions[5] = None

    def fail_decode(_codeword):
        raise AssertionError("error-erasure decoding should not call the BCH backend decoder")

    monkeypatch.setattr(codec, "decode", fail_decode)

    result = codec.decode_errors_and_erasures(decisions, max_assignments=64)

    assert result.status == "decoded"
    assert result.message_bits == MESSAGE
    assert result.codeword_bits == codec.encode(MESSAGE)


def test_bch_rejects_mismatched_t() -> None:
    with pytest.raises(ValueError):
        BCHCodec(23, 8, 2)
