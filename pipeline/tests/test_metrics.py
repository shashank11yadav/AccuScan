import pytest

from accuscan.metrics import ErrorRates, edit_distance


@pytest.mark.parametrize(
    ("a", "b", "d"),
    [("", "", 0), ("abc", "abc", 0), ("kitten", "sitting", 3), ("", "abc", 3), ("abc", "", 3)],
)
def test_edit_distance(a: str, b: str, d: int) -> None:
    assert edit_distance(a, b) == d
    assert edit_distance(b, a) == d


def test_error_rates_use_reference_length() -> None:
    rates = ErrorRates()
    rates.add("helo world", "hello world")
    rates.add("cat", "cat")
    assert rates.cer == pytest.approx(1 / 14)
    assert rates.wer == pytest.approx(1 / 3)
    assert rates.as_dict()["exact_match"] == 0.5


def test_word_errors_count_insertions() -> None:
    rates = ErrorRates()
    rates.add("a b c", "a c")
    assert rates.word_errors == 1
