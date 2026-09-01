from watermark.partition import ExactPermutationPartitioner


def test_partition_is_disjoint_and_covers_allowed_tokens() -> None:
    part = ExactPermutationPartitioner().partition(seed=17, vocab_size=11, excluded_ids={0})
    regions = [set(part.bit0), set(part.bit1), set(part.lower_a), set(part.lower_b)]
    assert all(regions[i].isdisjoint(regions[j]) for i in range(4) for j in range(i + 1, 4))
    assert set.union(*regions) == set(range(1, 11))
    assert len(part.bit0) == len(part.bit1)


def test_region_for_token_matches_materialized_partition() -> None:
    partitioner = ExactPermutationPartitioner()
    part = partitioner.partition(seed=17, vocab_size=11, excluded_ids={0})

    for token_id in range(11):
        if token_id in part.bit0:
            expected = 0
        elif token_id in part.bit1:
            expected = 1
        else:
            expected = None
        assert partitioner.region_for_token(
            seed=17,
            vocab_size=11,
            excluded_ids={0},
            token_id=token_id,
        ) == expected


def test_region_for_token_returns_none_for_excluded_or_out_of_range_token() -> None:
    partitioner = ExactPermutationPartitioner()

    assert partitioner.region_for_token(seed=17, vocab_size=11, excluded_ids={0}, token_id=0) is None
    assert partitioner.region_for_token(seed=17, vocab_size=11, excluded_ids={0}, token_id=-1) is None
    assert partitioner.region_for_token(seed=17, vocab_size=11, excluded_ids={0}, token_id=11) is None
