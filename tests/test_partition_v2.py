import pytest
import torch

from watermark.partition_v2 import StatelessExactPartitioner, partition_round_keys


@pytest.mark.parametrize(
    ("vocab_size", "excluded"),
    [(12, {0}), (17, {0, 1}), (5, set())],
)
def test_stateless_partition_is_an_exact_bijection(
    vocab_size: int,
    excluded: set[int],
) -> None:
    partitioner = StatelessExactPartitioner(vocab_size, excluded)
    keys = partition_round_keys(123456789)

    ranks = [
        partitioner.rank_for_token(round_keys=keys, token_id=token)
        for token in range(vocab_size)
        if token not in excluded
    ]

    assert sorted(ranks) == list(range(vocab_size - len(excluded)))


def test_scalar_ranks_match_golden_cycle_walked_feistel_vector() -> None:
    partitioner = StatelessExactPartitioner(13, {0, 4})

    assert [
        partitioner.rank_for_token(round_keys=partition_round_keys(7), token_id=token)
        for token in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12)
    ] == [4, 1, 9, 7, 8, 10, 0, 3, 6, 5, 2]


@pytest.mark.parametrize(
    ("eligible_count", "expected_sizes"),
    [
        (4, (1, 1, 1, 1)),
        (5, (1, 1, 1, 2)),
        (6, (1, 1, 2, 2)),
        (7, (1, 1, 2, 3)),
        (12, (3, 3, 3, 3)),
    ],
)
def test_region_cardinalities_follow_existing_quarter_rule(
    eligible_count: int,
    expected_sizes: tuple[int, int, int, int],
) -> None:
    partitioner = StatelessExactPartitioner(eligible_count + 1, {0})

    values = partitioner.partition_for_context(round_keys=partition_round_keys(7))

    assert tuple(
        map(len, (values.bit0, values.bit1, values.lower_a, values.lower_b))
    ) == expected_sizes


def test_materialized_partition_matches_golden_rank_order() -> None:
    partitioner = StatelessExactPartitioner(13, {0, 4})

    values = partitioner.partition_for_context(round_keys=partition_round_keys(7))

    assert values.bit0 == (8, 2)
    assert values.bit1 == (12, 9)
    assert values.lower_a == (1, 11, 10)
    assert values.lower_b == (5, 6, 3, 7)


def test_region_for_token_covers_all_four_regions() -> None:
    partitioner = StatelessExactPartitioner(13, {0, 4})
    keys = partition_round_keys(7)

    assert [
        partitioner.region_for_token(round_keys=keys, token_id=token)
        for token in (8, 12, 1, 5)
    ] == [0, 1, 2, 3]


@pytest.mark.parametrize("token_id", [-1, 0, 4, 13])
def test_excluded_and_out_of_range_tokens_have_no_rank_or_region(token_id: int) -> None:
    partitioner = StatelessExactPartitioner(13, {0, 4})
    keys = partition_round_keys(7)

    assert partitioner.rank_for_token(round_keys=keys, token_id=token_id) is None
    assert partitioner.region_for_token(round_keys=keys, token_id=token_id) is None


@pytest.mark.parametrize(
    ("vocab_size", "excluded", "message"),
    [
        (0, set(), "vocab_size must be positive"),
        (4, {-1}, "excluded token ID is outside the vocabulary"),
        (4, {4}, "excluded token ID is outside the vocabulary"),
        (4, {0}, "At least four eligible vocabulary tokens are required"),
    ],
)
def test_invalid_vocabulary_configuration_is_rejected(
    vocab_size: int,
    excluded: set[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        StatelessExactPartitioner(vocab_size, excluded)


def test_exactly_six_round_keys_are_required() -> None:
    partitioner = StatelessExactPartitioner(8, set())

    with pytest.raises(ValueError, match="exactly six round keys"):
        partitioner.rank_for_token(round_keys=(1, 2, 3, 4, 5), token_id=0)


def test_tensor_regions_match_golden_vector_on_cpu() -> None:
    partitioner = StatelessExactPartitioner(12, set())

    regions = partitioner.regions_for_vocab(
        round_keys_by_row=[partition_round_keys(7)],
        device=torch.device("cpu"),
    )

    assert regions.dtype == torch.int64
    assert regions.shape == (1, 12)
    assert regions[0].tolist() == [1, 0, 3, 2, 2, 3, 3, 1, 2, 1, 0, 0]


def test_tensor_regions_match_scalar_regions_on_cpu() -> None:
    partitioner = StatelessExactPartitioner(31, {0, 3})
    keys = [partition_round_keys(11), partition_round_keys(29)]

    tensor = partitioner.regions_for_vocab(
        round_keys_by_row=keys,
        device=torch.device("cpu"),
    )

    for row, row_keys in enumerate(keys):
        expected = [
            partitioner.region_for_token(round_keys=row_keys, token_id=token)
            for token in range(31)
        ]
        assert tensor[row].tolist() == [
            -1 if value is None else value for value in expected
        ]


def test_tensor_regions_have_exact_uneven_cardinalities() -> None:
    partitioner = StatelessExactPartitioner(50272, {0, 1})

    regions = partitioner.regions_for_vocab(
        round_keys_by_row=[partition_round_keys(42)],
        device=torch.device("cpu"),
    )

    assert regions[0, :2].tolist() == [-1, -1]
    assert torch.bincount(regions[0, 2:], minlength=4).tolist() == [
        12567,
        12567,
        12568,
        12568,
    ]


def test_eligible_id_tensor_is_reused_without_caching_regions() -> None:
    partitioner = StatelessExactPartitioner(31, {0, 3})
    device = torch.device("cpu")

    first_regions = partitioner.regions_for_vocab(
        round_keys_by_row=[partition_round_keys(11)],
        device=device,
    )
    cached_ids = partitioner._eligible_ids_by_device[device]
    second_regions = partitioner.regions_for_vocab(
        round_keys_by_row=[partition_round_keys(29)],
        device=device,
    )

    assert partitioner._eligible_ids_by_device[device] is cached_ids
    assert not torch.equal(first_regions, second_regions)


def test_regions_for_tokens_support_candidate_specific_partition_keys() -> None:
    partitioner = StatelessExactPartitioner(12, {0})
    token_ids = torch.tensor([[1, 4, 9], [2, 5, 10]])
    rows = token_ids.tolist()
    round_keys_by_token = tuple(
        tuple(partition_round_keys(seed + token_id) for token_id in row)
        for seed, row in zip((10, 20), rows, strict=True)
    )

    regions = partitioner.regions_for_tokens(
        round_keys_by_token=round_keys_by_token,
        token_ids=token_ids,
    )

    assert regions.tolist() == [[0, 3, 3], [0, 2, 3]]


def test_regions_for_tokens_marks_excluded_candidates_ineligible() -> None:
    partitioner = StatelessExactPartitioner(12, {0})
    token_ids = torch.tensor([[0, 1]])
    round_keys_by_token = ((partition_round_keys(10), partition_round_keys(11)),)

    regions = partitioner.regions_for_tokens(
        round_keys_by_token=round_keys_by_token,
        token_ids=token_ids,
    )

    assert regions.tolist() == [[-1, 0]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_regions_match_cpu_regions() -> None:
    partitioner = StatelessExactPartitioner(50272, {0, 1})
    keys = [partition_round_keys(42), partition_round_keys(99)]

    cpu = partitioner.regions_for_vocab(
        round_keys_by_row=keys,
        device=torch.device("cpu"),
    )
    cuda = partitioner.regions_for_vocab(
        round_keys_by_row=keys,
        device=torch.device("cuda"),
    )

    assert cuda.device.type == "cuda"
    assert torch.equal(cpu, cuda.cpu())
