import pytest

from watermark.prf import KeyedPRF, partition_round_keys


def test_prf_is_deterministic_and_domain_separated() -> None:
    prf = KeyedPRF(b"secret")
    context = [4, 8, 15, 16]
    assert prf.seed(context, "shared") == prf.seed(context, "shared")
    assert prf.seed(context, "partition") != prf.seed(context, "position")


def test_prf_seed_keeps_legacy_golden_values() -> None:
    prf = KeyedPRF(b"secret")
    context = [4, 8, 15, 16]

    assert prf.seed(context, "shared") == 15815698519238049179
    assert prf.seed(context, "partition") == 2577210898764477968
    assert prf.seed(context, "position") == 926580564545288181


@pytest.mark.parametrize(
    ("base_seed", "expected"),
    [
        (
            0,
            (
                16852473371444490038,
                16125043466793272905,
                9992746046255836306,
                5533602333010859367,
                8698330073434634262,
                2132173345291297387,
            ),
        ),
        (
            123456789,
            (
                4172726930665476555,
                15477620728544260555,
                7779875036031293869,
                8828303373517640590,
                2858718267555524112,
                9752937822159868207,
            ),
        ),
    ],
)
def test_partition_round_keys_match_golden_vectors(
    base_seed: int,
    expected: tuple[int, ...],
) -> None:
    assert partition_round_keys(base_seed) == expected


def test_partition_round_keys_treat_base_seed_as_uint64() -> None:
    assert partition_round_keys((1 << 64) + 7) == partition_round_keys(7)
    assert partition_round_keys(-1) == partition_round_keys((1 << 64) - 1)
