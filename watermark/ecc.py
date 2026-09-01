from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

from watermark.result_types import DecodeResult

# One verified primitive polynomial for GF(2^m), represented with the x^m bit included.
_PRIMITIVE_POLYNOMIALS: dict[int, int] = {
    2: 0x7,
    3: 0xB,
    4: 0x13,
    5: 0x25,
    6: 0x43,
    7: 0x89,
    8: 0x11D,
    9: 0x211,
    10: 0x409,
}


def _validate_bits(bits: Sequence[int], expected: int, label: str) -> tuple[int, ...]:
    values = tuple(int(bit) for bit in bits)
    if len(values) != expected:
        raise ValueError(f"{label} must contain exactly {expected} bits")
    if any(bit not in (0, 1) for bit in values):
        raise ValueError(f"{label} symbols must be 0 or 1")
    return values


def _bits_to_int(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _int_to_bits(value: int, width: int) -> tuple[int, ...]:
    return tuple((value >> shift) & 1 for shift in range(width - 1, -1, -1))


def _poly_mod(dividend: int, divisor: int) -> int:
    divisor_degree = divisor.bit_length() - 1
    while dividend and dividend.bit_length() - 1 >= divisor_degree:
        dividend ^= divisor << (dividend.bit_length() - 1 - divisor_degree)
    return dividend


class _BinaryField:
    def __init__(self, m: int, primitive_poly: int) -> None:
        self.m = m
        self.order = 1 << m
        self.period = self.order - 1
        self.exp = [0] * (2 * self.period)
        self.log = [-1] * self.order
        value = 1
        seen: set[int] = set()
        for index in range(self.period):
            if value in seen or value == 0:
                raise ValueError(f"Polynomial 0x{primitive_poly:x} is not primitive for m={m}")
            seen.add(value)
            self.exp[index] = value
            self.log[value] = index
            value <<= 1
            if value & self.order:
                value ^= primitive_poly
        if value != 1 or len(seen) != self.period:
            raise ValueError(f"Polynomial 0x{primitive_poly:x} is not primitive for m={m}")
        for index in range(self.period, 2 * self.period):
            self.exp[index] = self.exp[index - self.period]

    def mul(self, left: int, right: int) -> int:
        if left == 0 or right == 0:
            return 0
        return self.exp[(self.log[left] + self.log[right]) % self.period]

    def alpha(self, exponent: int) -> int:
        return self.exp[exponent % self.period]


def _field_poly_mul(left: list[int], right: list[int], field: _BinaryField) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] ^= field.mul(a, b)
    return result


def _binary_poly_mul(left: int, right: int) -> int:
    result = 0
    shift = 0
    while right:
        if right & 1:
            result ^= left << shift
        right >>= 1
        shift += 1
    return result


def _cyclotomic_coset(exponent: int, length: int) -> tuple[int, ...]:
    values: list[int] = []
    current = exponent % length
    while current not in values:
        values.append(current)
        current = (2 * current) % length
    return tuple(values)


def _generator_polynomial(m: int, t: int) -> int:
    if t <= 0:
        raise ValueError("t must be positive")
    primitive_poly = _PRIMITIVE_POLYNOMIALS.get(m)
    if primitive_poly is None:
        raise ValueError(f"No fallback primitive polynomial is registered for m={m}")
    field = _BinaryField(m, primitive_poly)
    length = field.period
    seen_cosets: set[tuple[int, ...]] = set()
    generator = 1
    for exponent in range(1, 2 * t + 1):
        coset = _cyclotomic_coset(exponent, length)
        canonical = tuple(sorted(coset))
        if canonical in seen_cosets:
            continue
        seen_cosets.add(canonical)
        minimal: list[int] = [1]
        for root_exponent in coset:
            minimal = _field_poly_mul(minimal, [field.alpha(root_exponent), 1], field)
        if any(coefficient not in (0, 1) for coefficient in minimal):
            raise RuntimeError("Minimal polynomial did not reduce to GF(2)")
        minimal_binary = sum(
            int(coefficient) << degree
            for degree, coefficient in enumerate(minimal)
        )
        generator = _binary_poly_mul(generator, minimal_binary)
    return generator


def _find_structure(n: int, k: int) -> tuple[int, int, int, int, int]:
    """Return m, parent_n, parent_k, shortening, constructed_t for a valid code shape."""
    candidates: list[tuple[int, int, int, int, int]] = []
    for m in sorted(_PRIMITIVE_POLYNOMIALS):
        parent_n = (1 << m) - 1
        if parent_n < n:
            continue
        shortening = parent_n - n
        desired_parent_k = k + shortening
        if desired_parent_k <= 0 or desired_parent_k >= parent_n:
            continue
        previous_degree = -1
        for candidate_t in range(1, (parent_n - 1) // 2 + 1):
            generator = _generator_polynomial(m, candidate_t)
            degree = generator.bit_length() - 1
            if degree == previous_degree:
                continue
            previous_degree = degree
            parent_k = parent_n - degree
            if parent_k == desired_parent_k:
                candidates.append((m, parent_n, parent_k, shortening, candidate_t))
                break
            if parent_k < desired_parent_k:
                break
    if not candidates:
        raise ValueError(
            f"Invalid BCH configuration (n={n}, k={k}). Choose a shortened primitive binary "
            "BCH code whose parent dimension matches k + shortening."
        )
    return candidates[0]


class _FallbackBCH:
    """Small research fallback used only when the optional galois package is unavailable."""

    def __init__(self, n: int, k: int, t: int) -> None:
        m, parent_n, parent_k, shortening, constructed_t = _find_structure(n, k)
        if constructed_t != t:
            raise ValueError(
                f"BCH(n={n}, k={k}) has constructed t={constructed_t}, not requested t={t}"
            )
        self.n = n
        self.k = k
        self.t = t
        self.m = m
        self.parent_n = parent_n
        self.parent_k = parent_k
        self.shortening = shortening
        self.generator = _generator_polynomial(m, t)
        self.d = 2 * t + 1
        self.is_systematic = True
        self.field_order = 1 << m
        self._syndrome_table: dict[int, int] | None = None

    def encode(self, message_bits: Sequence[int]) -> tuple[int, ...]:
        message = _validate_bits(message_bits, self.k, "message")
        parent_message = (0,) * self.shortening + message
        parity_width = self.parent_n - self.parent_k
        dividend = _bits_to_int(parent_message) << parity_width
        remainder = _poly_mod(dividend, self.generator)
        parent_codeword = _int_to_bits(dividend ^ remainder, self.parent_n)
        return parent_codeword[self.shortening :]

    def _build_syndrome_table(self) -> dict[int, int]:
        combinations = sum(math.comb(self.n, errors) for errors in range(1, self.t + 1))
        if combinations > 1_000_000:
            raise RuntimeError(
                "Fallback BCH syndrome table would be too large; install the 'galois' package "
                "for this BCH configuration"
            )
        table: dict[int, int] = {}
        transmitted_positions = range(self.shortening, self.parent_n)
        for errors in range(1, self.t + 1):
            for indices in itertools.combinations(transmitted_positions, errors):
                mask = 0
                for index in indices:
                    mask |= 1 << (self.parent_n - 1 - index)
                syndrome = _poly_mod(mask, self.generator)
                existing = table.get(syndrome)
                if existing is not None and existing != mask:
                    raise RuntimeError("Syndrome collision within the configured correction radius")
                table[syndrome] = mask
        return table

    def decode(self, codeword_bits: Sequence[int]) -> DecodeResult:
        codeword = _validate_bits(codeword_bits, self.n, "codeword")
        parent_bits = (0,) * self.shortening + codeword
        received = _bits_to_int(parent_bits)
        syndrome = _poly_mod(received, self.generator)
        corrected_errors = 0
        if syndrome:
            if self._syndrome_table is None:
                self._syndrome_table = self._build_syndrome_table()
            error_mask = self._syndrome_table.get(syndrome)
            if error_mask is None:
                return DecodeResult(
                    status="decode_failed",
                    message_bits=None,
                    corrected_errors=None,
                    error=f"Uncorrectable BCH syndrome 0x{syndrome:x}",
                )
            received ^= error_mask
            corrected_errors = error_mask.bit_count()
            if _poly_mod(received, self.generator) != 0:
                return DecodeResult(
                    status="decode_failed",
                    message_bits=None,
                    corrected_errors=None,
                    error="BCH correction did not produce a valid codeword",
                )
        corrected = _int_to_bits(received, self.parent_n)
        parent_message = corrected[: self.parent_k]
        message = parent_message[self.shortening :]
        corrected_codeword = tuple(corrected[self.shortening :])
        return DecodeResult(
            status="decoded",
            message_bits=tuple(message),
            corrected_errors=corrected_errors,
            codeword_bits=corrected_codeword,
        )


class BCHCodec:
    """Validated shortened primitive binary BCH codec.

    The official ``galois`` backend is used when installed. A bounded pure-Python
    fallback keeps offline unit tests executable in restricted environments.
    """

    def __init__(self, n: int, k: int, t: int) -> None:
        if n <= k or k <= 0 or t <= 0:
            raise ValueError("ECC parameters must satisfy n > k > 0 and t > 0")
        m, parent_n, parent_k, shortening, constructed_t = _find_structure(n, k)
        if constructed_t != t:
            raise ValueError(
                f"BCH(n={n}, k={k}) has constructed t={constructed_t}, not requested t={t}"
            )
        self.n = int(n)
        self.k = int(k)
        self.t = int(t)
        self.parent_n = parent_n
        self.parent_k = parent_k
        self.shortening = shortening
        self._backend: str
        self._code: Any
        self._codebook: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] | None = None
        try:
            import galois  # type: ignore[import-not-found]

            code = galois.BCH(parent_n, parent_k)
            actual_t = int(code.t)
            if actual_t != t:
                raise ValueError(
                    f"BCH(n={n}, k={k}) has constructed t={actual_t}, not requested t={t}"
                )
            self._backend = "galois"
            self._code = code
            self.d = int(code.d)
            self.systematic = bool(code.is_systematic)
            self.field_order = int(code.field.order)
        except Exception:
            fallback = _FallbackBCH(n, k, t)
            self._backend = "python-fallback"
            self._code = fallback
            self.d = fallback.d
            self.systematic = fallback.is_systematic
            self.field_order = fallback.field_order

    @property
    def backend(self) -> str:
        return self._backend

    def encode(self, message_bits: Sequence[int]) -> tuple[int, ...]:
        message = _validate_bits(message_bits, self.k, "message")
        if self._backend == "python-fallback":
            return self._code.encode(message)
        encoded = self._code.encode(list(message))
        values = tuple(int(value) for value in encoded.tolist())
        if len(values) != self.n:
            raise RuntimeError(f"BCH backend returned {len(values)} bits, expected {self.n}")
        return values

    def _all_codewords(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        if self._codebook is not None:
            return self._codebook
        rows: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for message_value in range(1 << self.k):
            message = _int_to_bits(message_value, self.k)
            rows.append((message, self.encode(message)))
        self._codebook = tuple(rows)
        return self._codebook

    def decode(self, codeword_bits: Sequence[int]) -> DecodeResult:
        codeword = _validate_bits(codeword_bits, self.n, "codeword")
        if self._backend == "python-fallback":
            return self._code.decode(codeword)
        try:
            decoded, errors = self._code.decode(list(codeword), errors=True)
            corrected_errors = int(errors)
            if corrected_errors < 0:
                return DecodeResult(
                    status="decode_failed",
                    message_bits=None,
                    corrected_errors=None,
                    error="BCH decoder reported an uncorrectable codeword",
                )
            message = tuple(int(value) for value in decoded.tolist())
            if len(message) != self.k:
                raise RuntimeError(
                    f"BCH backend returned {len(message)} message bits, expected {self.k}"
                )
            canonical_codeword = self.encode(message)
            return DecodeResult(
                status="decoded",
                message_bits=message,
                corrected_errors=corrected_errors,
                codeword_bits=canonical_codeword,
            )
        except Exception as exc:  # pragma: no cover - backend-specific failure surface
            return DecodeResult(
                status="decode_failed",
                message_bits=None,
                corrected_errors=None,
                error=str(exc),
            )

    def decode_errors_and_erasures(
        self,
        decisions: Sequence[int | None],
        *,
        max_assignments: int = 64,
    ) -> DecodeResult:
        """Decode a ternary received word using bounded codebook search.

        Non-erased disagreements cost two distance units and erasures cost one.
        A candidate is retained only when ``2 * v + e < d``.
        """

        values = tuple(decisions)
        if len(values) != self.n:
            raise ValueError(f"decisions must contain exactly {self.n} symbols")
        if any(value not in (0, 1, None) for value in values):
            raise ValueError("decision symbols must be 0, 1, or None")
        if max_assignments <= 0:
            raise ValueError("max_assignments must be positive")

        erasure_positions = tuple(index for index, value in enumerate(values) if value is None)
        erasure_count = len(erasure_positions)
        assignments = 1 << erasure_count
        if erasure_count >= self.d:
            return DecodeResult(
                status="too_many_erasures",
                message_bits=None,
                corrected_errors=None,
                error=f"{erasure_count} erasures cannot satisfy 2*v+e < d={self.d}",
                erasure_count=erasure_count,
                candidate_count=0,
                assignments_tested=0,
            )
        if assignments > max_assignments:
            return DecodeResult(
                status="too_many_erasures",
                message_bits=None,
                corrected_errors=None,
                error=f"{assignments} erasure assignments exceed max_assignments={max_assignments}",
                erasure_count=erasure_count,
                candidate_count=0,
                assignments_tested=0,
            )

        retained: dict[tuple[int, ...], tuple[tuple[int, ...], int | None, int, int]] = {}
        tested = assignments
        for message, canonical in self._all_codewords():
            known_errors = sum(
                int(canonical[index] != int(value))
                for index, value in enumerate(values)
                if value is not None
            )
            distance_cost = 2 * known_errors + erasure_count
            if distance_cost >= self.d:
                continue
            retained[canonical] = (message, known_errors, known_errors, distance_cost)

        candidate_count = len(retained)
        if candidate_count == 0:
            return DecodeResult(
                status="decode_failed",
                message_bits=None,
                corrected_errors=None,
                error="No BCH codeword satisfies the bounded error-erasure radius",
                erasure_count=erasure_count,
                candidate_count=0,
                assignments_tested=tested,
            )
        if candidate_count > 1:
            return DecodeResult(
                status="ambiguous",
                message_bits=None,
                corrected_errors=None,
                error=f"{candidate_count} distinct BCH candidates satisfy the bounded radius",
                erasure_count=erasure_count,
                candidate_count=candidate_count,
                assignments_tested=tested,
            )

        canonical, (message, corrected_errors, known_errors, distance_cost) = next(iter(retained.items()))
        return DecodeResult(
            status="decoded",
            message_bits=message,
            corrected_errors=corrected_errors,
            codeword_bits=canonical,
            erasure_count=erasure_count,
            known_position_errors=known_errors,
            distance_cost=distance_cost,
            candidate_count=1,
            assignments_tested=tested,
        )
