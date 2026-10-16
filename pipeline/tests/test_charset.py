import pytest

from accuscan.charset import BLANK, CHARSET, NUM_CLASSES, decode_ids, encode, normalise_text


def test_blank_is_zero_and_printable_ascii_follows() -> None:
    assert BLANK == 0
    assert NUM_CLASSES == 96
    assert CHARSET[1] == " "
    assert CHARSET[-1] == "~"
    assert "".join(CHARSET[1:]) == "".join(chr(c) for c in range(32, 127))


def test_encode_round_trips() -> None:
    text = "Hello, World! 42? no: 'quotes'"
    clean = normalise_text(text)
    assert clean is not None
    assert decode_ids(encode(clean)) == clean


def test_normalise_maps_typographic_characters() -> None:
    text = "\u201cdon\u2019t\u201d \u2013 ok\u2026"
    assert normalise_text(text) == '"don\'t" - ok...'


def test_normalise_rejects_unknown_characters() -> None:
    assert normalise_text("café ∞") is None


def test_normalise_collapses_whitespace() -> None:
    assert normalise_text("  a \t b\n") == "a b"


def test_encode_rejects_unknown() -> None:
    with pytest.raises(KeyError):
        encode("é")
