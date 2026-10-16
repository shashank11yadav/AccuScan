"""Output alphabet: CTC blank at index 0, then printable ASCII 32-126."""

from __future__ import annotations

import json
import re
from pathlib import Path

BLANK = 0
BLANK_SYMBOL = "<blank>"
CHARSET: tuple[str, ...] = (BLANK_SYMBOL, *(chr(c) for c in range(32, 127)))
NUM_CLASSES = len(CHARSET)
_INDEX = {ch: i for i, ch in enumerate(CHARSET) if i != BLANK}

# Characters GNHK and book text use that have a plain ASCII equivalent.
_REPLACEMENTS = {
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2026": "...",
    "\u00a0": " ",  # no-break space
}
_WHITESPACE = re.compile(r"\s+")


def normalise_text(text: str) -> str | None:
    """Map typographic punctuation to ASCII; None if anything else is out of charset."""
    mapped = "".join(_REPLACEMENTS.get(ch, ch) for ch in text)
    mapped = _WHITESPACE.sub(" ", mapped).strip()
    if any(ch not in _INDEX for ch in mapped):
        return None
    return mapped


def encode(text: str) -> list[int]:
    return [_INDEX[ch] for ch in text]


def decode_ids(ids: list[int]) -> str:
    return "".join(CHARSET[i] for i in ids if i != BLANK)


def write_charset(path: Path) -> None:
    payload = {"version": 1, "blank": BLANK, "symbols": list(CHARSET)}
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
