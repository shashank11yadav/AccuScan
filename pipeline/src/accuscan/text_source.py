"""Text for synthetic lines: book fragments, single words, numbers and symbols.

Each bundled book is split into paragraphs; the last `heldout_fraction` of them
is reserved for the held-out canvas test set so its text is unseen in training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .charset import CHARSET, normalise_text

MAX_CHARS = 28
_TOKEN = re.compile(r"\S+")
_JUNK = re.compile(r"[\[\]_*]|^chapter\b|illustration", re.IGNORECASE)
_SYMBOLS = "".join(CHARSET[1:])
_SHOPPING = (
    "milk", "eggs", "bread", "butter", "apples", "rice", "tea", "coffee", "flour",
    "sugar", "cheese", "pasta", "onions", "soap", "tomatoes", "oats", "jam", "salt",
)  # fmt: skip
_UNITS = ("kg", "g", "ml", "L", "x", "pcs", "lb", "oz", "cm", "km", "%")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True)
class TextSource:
    """Tokenised paragraphs of clean, in-charset text."""

    paragraphs: tuple[tuple[str, ...], ...]
    words: tuple[str, ...]

    @classmethod
    def from_texts(cls, texts: list[str], part: str, heldout_fraction: float) -> TextSource:
        if part not in ("train", "heldout"):
            raise ValueError(part)
        paragraphs: list[tuple[str, ...]] = []
        for text in texts:
            paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
            cut = int(len(paras) * (1.0 - heldout_fraction))
            chosen = paras[:cut] if part == "train" else paras[cut:]
            for para in chosen:
                clean = normalise_text(para)
                if not clean or _JUNK.search(clean):
                    continue
                tokens = tuple(_TOKEN.findall(clean))
                if len(tokens) >= 2:
                    paragraphs.append(tokens)
        words = tuple(t for p in paragraphs for t in p)
        if not words:
            raise ValueError("no usable text")
        return cls(tuple(paragraphs), words)


def _case(word: str, rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.12:
        return word.upper()
    if r < 0.32:
        return word[:1].upper() + word[1:]
    return word


def _number_text(rng: np.random.Generator) -> str:
    kind = int(rng.integers(0, 9))
    n = int(rng.integers(0, 1000))
    if kind == 0:
        return str(int(rng.integers(0, 100000)))
    if kind == 1:
        day, month = int(rng.integers(1, 29)), int(rng.integers(1, 13))
        return f"{day}/{month:02d}/{int(rng.integers(1990, 2031))}"
    if kind == 2:
        suffix = ("am", "pm", "")[int(rng.integers(0, 3))]
        return f"{int(rng.integers(1, 13))}:{int(rng.integers(0, 60)):02d}{suffix}"
    if kind == 3:
        return f"${n}.{int(rng.integers(0, 100)):02d}"
    if kind == 4:
        return f"{int(rng.integers(1, 13))} {_UNITS[int(rng.integers(0, len(_UNITS)))]}"
    if kind == 5:
        item = _SHOPPING[int(rng.integers(0, len(_SHOPPING)))]
        return f"{int(rng.integers(1, 10))}. {item}"
    if kind == 6:
        item = _SHOPPING[int(rng.integers(0, len(_SHOPPING)))]
        return f"{int(rng.integers(1, 6))} x {item}"
    if kind == 7:
        month_name = _MONTHS[int(rng.integers(0, 12))]
        return f"{int(rng.integers(1, 32))} {month_name}"
    return f"{int(rng.integers(100, 1000))}-{int(rng.integers(1000, 10000))}"


def _symbols_text(rng: np.random.Generator) -> str:
    length = int(rng.integers(1, 9))
    return "".join(_SYMBOLS[int(i)] for i in rng.integers(1, len(_SYMBOLS), size=length))


def sample_text(source: TextSource, rng: np.random.Generator) -> str:
    """One line of text, at most MAX_CHARS long, with a varied mix of content."""
    r = rng.random()
    if r < 0.30:
        text = _case(source.words[int(rng.integers(0, len(source.words)))], rng)
    elif r < 0.75:
        para = source.paragraphs[int(rng.integers(0, len(source.paragraphs)))]
        n = int(rng.integers(2, 6))
        start = int(rng.integers(0, max(1, len(para) - n + 1)))
        text = " ".join(para[start : start + n])
    elif r < 0.92:
        text = _number_text(rng)
    else:
        text = _symbols_text(rng)
    text = text.strip()
    if len(text) > MAX_CHARS:
        cut = text.rfind(" ", 0, MAX_CHARS)
        text = text[: cut if cut > 0 else MAX_CHARS].strip()
    return text or "a"
