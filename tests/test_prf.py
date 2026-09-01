from watermark.prf import KeyedPRF


def test_prf_is_deterministic_and_domain_separated() -> None:
    prf = KeyedPRF(b"secret")
    context = [4, 8, 15, 16]
    assert prf.seed(context, "shared") == prf.seed(context, "shared")
    assert prf.seed(context, "partition") != prf.seed(context, "position")
