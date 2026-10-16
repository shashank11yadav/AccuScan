from pathlib import Path

import numpy as np
import pytest

from accuscan.charset import CHARSET
from accuscan.render import render_line, supported_chars
from accuscan.resources import corpus_texts
from accuscan.shards import read_shards
from accuscan.synth import SynthJob, render_sample, run_synth
from accuscan.text_source import MAX_CHARS, TextSource, sample_text


@pytest.fixture(scope="module")
def train_text() -> TextSource:
    return TextSource.from_texts(list(corpus_texts().values()), "train", 0.1)


def test_bundled_corpus_has_five_books() -> None:
    texts = corpus_texts()
    assert sorted(texts) == ["pg11", "pg1342", "pg1661", "pg55", "pg74"]
    assert all("PROJECT GUTENBERG" not in t.upper()[:2000] for t in texts.values())


def test_heldout_text_is_disjoint_from_training_text(train_text: TextSource) -> None:
    held = TextSource.from_texts(list(corpus_texts().values()), "heldout", 0.1)
    train_paras = set(train_text.paragraphs)
    overlap = [p for p in held.paragraphs if p in train_paras]
    assert len(overlap) <= len(held.paragraphs) * 0.01  # only repeated stock phrases


def test_samples_fit_the_charset_and_length(train_text: TextSource) -> None:
    rng = np.random.default_rng(0)
    allowed = set(CHARSET[1:])
    for _ in range(2000):
        text = sample_text(train_text, rng)
        assert 0 < len(text) <= MAX_CHARS
        assert set(text) <= allowed
        assert text == text.strip()


def test_samples_cover_every_character(train_text: TextSource) -> None:
    rng = np.random.default_rng(1)
    seen: set[str] = set()
    for _ in range(20000):
        seen |= set(sample_text(train_text, rng))
    assert seen == set(CHARSET[1:])


def test_render_is_deterministic_for_a_seed() -> None:
    for style in ("paper", "canvas"):
        a = render_line("Hello 42!", None, style, np.random.default_rng(3))
        b = render_line("Hello 42!", None, style, np.random.default_rng(3))
        np.testing.assert_array_equal(a, b)
        c = render_line("Hello 42!", None, style, np.random.default_rng(4))
        assert a.shape != c.shape or not np.array_equal(a, c)


def test_canvas_style_is_black_on_white() -> None:
    img = render_line("draw", None, "canvas", np.random.default_rng(0))
    assert img.dtype == np.uint8
    assert np.median(img) == 255
    assert img.min() < 60


def test_paper_style_has_a_textured_background() -> None:
    img = render_line("note", None, "paper", np.random.default_rng(0))
    assert 140 < np.median(img) < 255
    assert img.std() > 2


def test_default_font_supports_all_characters() -> None:
    assert supported_chars(None) == frozenset(CHARSET[1:])


def test_render_sample_is_preprocessed(train_text: TextSource) -> None:
    sample = render_sample(train_text, [None], 0.5, np.random.default_rng(0))
    assert sample.pixels.shape[0] == 64
    assert sample.pixels.shape[1] % 4 == 0
    assert sample.source in ("paper", "canvas")


def test_synth_shards_are_reproducible_and_resumable(tmp_path: Path) -> None:
    def job(out: Path) -> SynthJob:
        return SynthJob(out, (None,), "train", count=10, shard_size=4, canvas_fraction=0.5, seed=9)

    assert run_synth(job(tmp_path / "a"), workers=1, log=lambda _: None) == 10
    assert run_synth(job(tmp_path / "a"), workers=1, log=lambda _: None) == 0  # resumed
    run_synth(job(tmp_path / "b"), workers=1, log=lambda _: None)
    a, b = read_shards(tmp_path / "a"), read_shards(tmp_path / "b")
    assert len(a) == 10
    assert [s.text for s in a] == [s.text for s in b]
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x.pixels, y.pixels)
