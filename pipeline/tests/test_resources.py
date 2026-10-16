from collections import Counter
from pathlib import Path

import pytest

from accuscan.paths import LEXICON_PATH
from accuscan.resources import (
    build_lexicon,
    read_lexicon,
    read_supplement,
    strip_boilerplate,
    write_lexicon,
)


def test_strip_boilerplate_keeps_only_the_body() -> None:
    text = (
        "Header\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\nBody line.\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\r\nLicence"
    )
    assert strip_boilerplate(text) == "Body line.\n"


def test_strip_boilerplate_requires_markers() -> None:
    with pytest.raises(ValueError, match="markers"):
        strip_boilerplate("no markers here")


def test_build_lexicon_ranks_by_count_and_adds_inflections() -> None:
    counts = Counter({"walk": 5, "the": 50, "zzz": 9})
    vocab = {"walk", "walks", "walked", "walking", "the"}
    entries = build_lexicon(counts, vocab, ["email"], size=10)
    words = [w for w, _ in entries]
    assert words[0] == "the"
    assert {"walks", "walked", "walking", "email"} <= set(words)
    assert "zzz" not in words  # not a dictionary word


def test_lexicon_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "w.txt.gz"
    write_lexicon([("the", 3), ("cat", 1)], path)
    assert read_lexicon(path) == {"the": 3, "cat": 1}


def test_bundled_lexicon_is_about_50k_words_with_modern_terms() -> None:
    lex = read_lexicon(LEXICON_PATH)
    assert 40_000 <= len(lex) <= 50_000
    assert {"the", "email", "don't", "hello"} <= set(lex)
    assert all(w == w.lower() for w in list(lex)[:1000])
    assert set(read_supplement()) <= set(lex)
