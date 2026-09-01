import pytest
from watermark.allocator import build_allocator


def test_hash_mod_and_reserved_allocator() -> None:
    assert build_allocator("hash_mod").allocate(seed=19, code_length=7) == 5
    with pytest.raises(NotImplementedError):
        build_allocator("balanced_permutation")
