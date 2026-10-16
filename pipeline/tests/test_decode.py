import math

import numpy as np
import pytest
from numpy.typing import NDArray

from accuscan.charset import CHARSET, NUM_CLASSES, encode
from accuscan.decode import (
    Lexicon,
    align_confidences,
    beam_search,
    greedy_decode,
)

IDX = {ch: i for i, ch in enumerate(CHARSET)}


def frames(*rows: dict[str, float]) -> NDArray[np.float64]:
    """Log-probs from per-frame {symbol: prob}; the remainder is spread evenly."""
    out = np.zeros((len(rows), NUM_CLASSES))
    for t, row in enumerate(rows):
        rest = max(1e-6, 1.0 - sum(row.values()))
        out[t, :] = rest / (NUM_CLASSES - len(row))
        for sym, p in row.items():
            out[t, IDX[sym] if sym != "_" else 0] = p
        out[t] /= out[t].sum()
    return np.log(out)


def path(text: str, p: float = 0.9) -> NDArray[np.float64]:
    """One peaked frame per symbol; '_' is blank."""
    return frames(*({ch: p} for ch in text))


def test_greedy_collapses_repeats_and_drops_blanks() -> None:
    assert greedy_decode(path("aa_ab_b")).text == "aabb"


def test_greedy_all_blank_is_empty() -> None:
    out = greedy_decode(path("___"))
    assert out.text == ""
    assert out.confidences == []


def test_greedy_confidence_is_max_over_the_run() -> None:
    lp = frames({"h": 0.6}, {"h": 0.8}, {"_": 0.9}, {"i": 0.7})
    out = greedy_decode(lp)
    assert out.text == "hi"
    assert out.confidences == pytest.approx([0.8, 0.7])


def test_greedy_ties_pick_lowest_index() -> None:
    lp = np.log(np.full((1, NUM_CLASSES), 1.0 / NUM_CLASSES))
    assert greedy_decode(lp).text == ""  # blank (index 0) wins the tie


def test_beam_sums_over_paths_where_greedy_does_not() -> None:
    lp = frames({"a": 0.4, "_": 0.6}, {"a": 0.4, "_": 0.6})
    assert greedy_decode(lp).text == ""
    out = beam_search(lp, None, beam_width=8)
    assert out.text == "a"


def test_beam_without_lexicon_matches_greedy_on_peaked_input() -> None:
    lp = path("he_ll_lo_ w0rld!")
    assert beam_search(lp, None).text == greedy_decode(lp).text == "hello w0rld!"


def test_lexicon_corrects_to_a_known_word() -> None:
    lex = Lexicon.from_counts({"cat": 10, "cot": 5, "dog": 3})
    lp = frames({"c": 0.9}, {"s": 0.5, "a": 0.4}, {"t": 0.9})
    assert greedy_decode(lp).text == "cst"
    assert beam_search(lp, lex).text == "cat"


def test_lexicon_is_case_insensitive_and_keeps_case() -> None:
    lex = Lexicon.from_counts({"cat": 1})
    assert beam_search(path("Cat"), lex).text == "Cat"


def test_lexicon_allows_digits_and_punctuation_between_words() -> None:
    lex = Lexicon.from_counts({"cat": 1, "dog": 1})
    assert beam_search(path("cat_ 42, dog!"), lex).text == "cat 42, dog!"


def test_lexicon_prefers_frequent_words_when_visual_evidence_is_equal() -> None:
    lex = Lexicon.from_counts({"cat": 1000, "cot": 1})
    lp = frames({"c": 0.9}, {"a": 0.45, "o": 0.45}, {"t": 0.9})
    assert beam_search(lp, lex, lm_weight=1.0).text == "cat"


def test_contractions_use_the_apostrophe_inside_words() -> None:
    lex = Lexicon.from_counts({"don't": 2, "don": 1, "it": 1})
    assert beam_search(path("don't"), lex).text == "don't"
    assert beam_search(path("'it'"), lex).text == "'it'"


def test_incomplete_word_falls_back_to_greedy() -> None:
    lex = Lexicon.from_counts({"house": 1})
    out = beam_search(path("hou"), lex)
    assert out.fallback
    assert out.text == "hou"


def test_beam_confidences_have_one_value_per_character() -> None:
    lex = Lexicon.from_counts({"cat": 10})
    out = beam_search(frames({"c": 0.9}, {"s": 0.5, "a": 0.4}, {"t": 0.9}), lex)
    assert len(out.confidences) == 3
    assert out.confidences[1] == pytest.approx(0.4, abs=1e-6)
    assert all(0.0 < c <= 1.0 for c in out.confidences)


def test_align_confidences_handles_repeated_letters() -> None:
    lp = frames({"l": 0.7}, {"_": 0.9}, {"l": 0.6})
    assert align_confidences(lp, encode("ll")) == pytest.approx([0.7, 0.6])


def test_align_confidences_too_short_returns_zeros() -> None:
    assert align_confidences(path("l"), encode("ll")) == [0.0, 0.0]


def test_lexicon_logprob_uses_counts() -> None:
    lex = Lexicon.from_counts({"a": 3, "b": 1})
    assert lex.logprob("a") == pytest.approx(math.log(3 / 4))
    assert lex.logprob("zzz") is None
