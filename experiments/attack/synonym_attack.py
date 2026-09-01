from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

AttackKind = Literal["replacement", "deletion", "insertion"]

_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")


@dataclass(frozen=True)
class WordSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Candidate:
    word: str
    replacement: str
    delta_tokens: int
    attacked_text: str


class SynonymProvider(Protocol):
    def synonyms(self, word: str) -> Sequence[str]:
        ...


@dataclass(frozen=True)
class AppliedEdit:
    word: str
    replacement: str
    start: int
    end: int
    delta_tokens: int


@dataclass(frozen=True)
class AttackResult:
    original_text: str
    attacked_text: str
    attacked_token_ids: list[int]
    original_token_count: int
    attacked_token_count: int
    editable_word_count: int
    target_edit_count: int
    achieved_edit_count: int
    achieved_attack_rate: float
    token_length_delta: int
    edits: list[AppliedEdit]


def collect_word_spans(text: str) -> list[WordSpan]:
    return [WordSpan(match.group(0), match.start(), match.end()) for match in _WORD_RE.finditer(text)]


def replace_span(text: str, span: WordSpan, replacement: str) -> str:
    return text[: span.start] + replacement + text[span.end :]


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _candidate_token_delta(
    text: str,
    span: WordSpan,
    replacement: str,
    tokenizer: Any,
    *,
    window_chars: int = 64,
) -> int:
    window_start = max(0, span.start - int(window_chars))
    window_end = min(len(text), span.end + int(window_chars))
    before = text[window_start:window_end]
    local_span = WordSpan(
        text=span.text,
        start=span.start - window_start,
        end=span.end - window_start,
    )
    after = replace_span(before, local_span, replacement)
    return _token_count(tokenizer, after) - _token_count(tokenizer, before)


def candidate_matches_attack(delta_tokens: int, attack: AttackKind) -> bool:
    if attack == "replacement":
        return delta_tokens == 0
    if attack == "deletion":
        return delta_tokens < 0
    if attack == "insertion":
        return delta_tokens > 0
    raise ValueError(f"Unsupported attack kind: {attack}")


def candidate_for_span(
    text: str,
    span: WordSpan,
    replacement: str,
    tokenizer: Any,
    *,
    original_token_count: int | None = None,
) -> Candidate:
    attacked_text = replace_span(text, span, replacement)
    try:
        delta_tokens = _candidate_token_delta(text, span, replacement, tokenizer)
    except Exception:
        before = _token_count(tokenizer, text) if original_token_count is None else int(original_token_count)
        after = _token_count(tokenizer, attacked_text)
        delta_tokens = after - before
    return Candidate(
        word=span.text,
        replacement=replacement,
        delta_tokens=delta_tokens,
        attacked_text=attacked_text,
    )


def filter_candidates(
    text: str,
    span: WordSpan,
    replacements: Sequence[str],
    tokenizer: Any,
    attack: AttackKind,
    *,
    original_token_count: int | None = None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for replacement in replacements:
        normalized = replacement.strip().replace("_", " ")
        if not normalized or normalized.lower() == span.text.lower() or normalized in seen:
            continue
        seen.add(normalized)
        candidate = candidate_for_span(
            text,
            span,
            normalized,
            tokenizer,
            original_token_count=original_token_count,
        )
        if candidate_matches_attack(candidate.delta_tokens, attack):
            candidates.append(candidate)
    return candidates


def target_edit_count(editable_word_count: int, attack_rate: float) -> int:
    if editable_word_count <= 0:
        return 0
    if not 0.0 <= attack_rate <= 1.0:
        raise ValueError("attack_rate must be in [0, 1]")
    if attack_rate == 0.0:
        return 0
    return max(1, int(round(editable_word_count * attack_rate)))


def _preserve_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def attack_text(
    text: str,
    tokenizer: Any,
    provider: SynonymProvider,
    attack: AttackKind,
    *,
    attack_rate: float,
    seed: int,
    require_full_rate: bool = False,
) -> AttackResult:
    spans = collect_word_spans(text)
    original_token_count = _token_count(tokenizer, text)
    target = target_edit_count(len(spans), attack_rate)
    rng = random.Random(int(seed))
    shuffled = list(spans)
    rng.shuffle(shuffled)

    current_text = text
    edits: list[AppliedEdit] = []
    edited_ranges: set[tuple[int, int]] = set()
    character_deltas: dict[tuple[int, int], int] = {}

    for original_span in shuffled:
        if len(edits) >= target:
            break
        if (original_span.start, original_span.end) in edited_ranges:
            continue
        original_range = (original_span.start, original_span.end)
        position_delta = sum(
            delta
            for (start, _end), delta in character_deltas.items()
            if start < original_span.start
        )
        span_start = original_span.start + position_delta
        span_end = original_span.end + position_delta
        span = WordSpan(text=current_text[span_start:span_end], start=span_start, end=span_end)
        replacements = [_preserve_case(span.text, value) for value in provider.synonyms(original_span.text)]
        try:
            candidates = filter_candidates(
                current_text,
                span,
                replacements,
                tokenizer,
                attack,
                original_token_count=_token_count(tokenizer, current_text),
            )
        except KeyError:
            candidates = []
        if not candidates:
            continue
        chosen = candidates[rng.randrange(len(candidates))]
        current_text = chosen.attacked_text
        edited_ranges.add(original_range)
        character_deltas[original_range] = len(chosen.replacement) - len(span.text)
        edits.append(
            AppliedEdit(
                word=span.text,
                replacement=chosen.replacement,
                start=span.start,
                end=span.end,
                delta_tokens=chosen.delta_tokens,
            )
        )

    if require_full_rate and len(edits) < target:
        raise RuntimeError(f"Could only apply {len(edits)} synonym edits out of requested {target}")

    attacked_token_ids = [int(value) for value in tokenizer.encode(current_text, add_special_tokens=False)]
    attacked_token_count = len(attacked_token_ids)
    return AttackResult(
        original_text=text,
        attacked_text=current_text,
        attacked_token_ids=attacked_token_ids,
        original_token_count=original_token_count,
        attacked_token_count=attacked_token_count,
        editable_word_count=len(spans),
        target_edit_count=target,
        achieved_edit_count=len(edits),
        achieved_attack_rate=0.0 if not spans else len(edits) / len(spans),
        token_length_delta=attacked_token_count - original_token_count,
        edits=edits,
    )


class WordNetSynonymProvider:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, ...]] = {}

    def _synsets(self, word: str):
        from nltk.corpus import wordnet as wn

        return wn.synsets(word)

    def synonyms(self, word: str) -> list[str]:
        cache_key = word.lower()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return list(cached)
        try:
            synsets = self._synsets(word)
        except LookupError as exc:
            raise RuntimeError(
                "NLTK WordNet data is required for paper-aligned synonym attacks. "
                "Run: python -m nltk.downloader wordnet omw-1.4"
            ) from exc
        values: list[str] = []
        seen: set[str] = set()
        for synset in synsets:
            for lemma in synset.lemma_names():
                value = lemma.replace("_", " ").strip()
                key = value.lower()
                if value and key != word.lower() and key not in seen:
                    seen.add(key)
                    values.append(value)
        self._cache[cache_key] = tuple(values)
        return values
