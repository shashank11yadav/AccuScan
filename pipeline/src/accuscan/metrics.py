"""Character and word error rates (edit distance over reference length)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def edit_distance(a: Sequence[object], b: Sequence[object]) -> int:
    """Levenshtein distance with unit costs."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


@dataclass
class ErrorRates:
    char_errors: int = 0
    chars: int = 0
    word_errors: int = 0
    words: int = 0
    exact: int = 0
    samples: int = 0

    def add(self, prediction: str, reference: str) -> None:
        self.char_errors += edit_distance(prediction, reference)
        self.chars += len(reference)
        ref_words = reference.split()
        self.word_errors += edit_distance(prediction.split(), ref_words)
        self.words += len(ref_words)
        self.exact += prediction == reference
        self.samples += 1

    @property
    def cer(self) -> float:
        return self.char_errors / max(1, self.chars)

    @property
    def wer(self) -> float:
        return self.word_errors / max(1, self.words)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "exact_match": round(self.exact / max(1, self.samples), 4),
            "samples": self.samples,
        }
