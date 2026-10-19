import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from accuscan.gnhk import (
    PrepStats,
    iter_pages,
    line_quad,
    order_quad,
    page_samples,
    parse_annotations,
    prepare_gnhk,
    validation_pages,
    warp_quad,
    word_order,
)
from accuscan.gnhk_fixture import write_fixture
from accuscan.shards import read_shard


@pytest.fixture(scope="module")
def gnhk_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_fixture(tmp_path_factory.mktemp("gnhk"))


def test_order_quad_handles_both_windings() -> None:
    tl, tr, br, bl = [10, 10], [50, 12], [49, 30], [9, 28]
    expected = np.array([tl, tr, br, bl], dtype=float)
    np.testing.assert_array_equal(order_quad(np.array([tl, bl, br, tr], float)), expected)
    np.testing.assert_array_equal(order_quad(np.array([tl, tr, br, bl], float)), expected)


def test_parse_reads_the_published_format() -> None:
    raw = [
        {
            "text": "Proof",
            "polygon": {
                "x0": 845,
                "y0": 1592,
                "x1": 859,
                "y1": 1809,
                "x2": 1188,
                "y2": 1888,
                "x3": 1300,
                "y3": 1588,
            },
            "line_idx": 4,
            "type": "H",
        }
    ]
    (ann,) = parse_annotations(raw)
    assert ann.text == "Proof"
    assert ann.line_idx == 4
    assert ann.handwritten
    np.testing.assert_array_equal(ann.quad[0], [845, 1592])  # top-left first


def test_warp_quad_rectifies_a_rotated_box() -> None:
    img = np.full((200, 300), 255, np.uint8)
    quad = np.array([[50, 50], [250, 50], [250, 100], [50, 100]], float)
    crop = warp_quad(img, quad, margin=0.0)
    assert crop.shape == (50, 200)
    angle = np.deg2rad(10)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = order_quad((quad - 150) @ rot.T + 150)
    crop = warp_quad(img, rotated, margin=0.0)
    assert crop.shape == (50, 200)


def test_word_order_follows_the_line_direction() -> None:
    a = np.array([[100, 10], [140, 12], [140, 32], [100, 30]], float)
    b = np.array([[10, 5], [60, 7], [60, 27], [10, 25]], float)
    assert word_order([a, b]) == [1, 0]
    lq = line_quad([a, b])
    assert lq.shape == (4, 2)


def test_fixture_pages_are_read_from_the_zip(gnhk_root: Path) -> None:
    pages = list(iter_pages(gnhk_root, "train"))
    assert [p.page_id for p in pages] == ["eng_AF_003", "eng_AS_004", "eng_EU_001", "eng_NA_002"]
    assert pages[0].image.ndim == 2


def test_extracted_folder_works_too(tmp_path: Path, gnhk_root: Path) -> None:
    with zipfile.ZipFile(gnhk_root / "test_data.zip") as zf:
        zf.extractall(tmp_path)
    assert len(list(iter_pages(tmp_path, "test"))) == 2


def test_missing_split_explains_where_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"goodnotes\.com/gnhk"):
        list(iter_pages(tmp_path, "train"))


def test_filters_special_printed_and_out_of_charset_words(gnhk_root: Path) -> None:
    page = next(iter_pages(gnhk_root, "test"))
    stats = PrepStats()
    samples = list(page_samples(page, stats))
    words = [s.text for s in samples if s.source == "gnhk-word"]
    lines = [s.text for s in samples if s.source == "gnhk-line"]
    assert words == ["Shopping", "list", "milk", "2", "eggs", "call", "mum", "see", "you", "soon!"]
    assert lines == ["Shopping list", "milk 2 eggs", "see you soon!"]
    assert (stats.words_special, stats.words_printed, stats.words_charset) == (1, 1, 1)
    assert stats.lines_dropped == 2
    for s in samples:
        assert s.pixels.shape[0] == 64


def test_validation_pages_are_deterministic() -> None:
    ids = [f"p{i}" for i in range(40)]
    a = validation_pages(ids, 0.1, seed=7)
    assert a == validation_pages(list(reversed(ids)), 0.1, seed=7)
    assert len(a) == 4


def test_prepare_writes_official_splits_plus_validation(tmp_path: Path, gnhk_root: Path) -> None:
    prepare_gnhk(gnhk_root, tmp_path, val_fraction=0.25, seed=1, log=lambda _: None)
    counts = {p.stem: len(list(read_shard(p))) for p in sorted(tmp_path.glob("*.npz"))}
    assert counts == {
        "test-line": 6,
        "test-word": 20,
        "train-line": 9,
        "train-word": 30,
        "val-line": 3,
        "val-word": 10,
    }


def test_fixture_json_uses_the_published_keys(gnhk_root: Path) -> None:
    with zipfile.ZipFile(gnhk_root / "train_data.zip") as zf:
        anns = json.loads(zf.read("train/eng_EU_001.json"))
    assert set(anns[0]) == {"text", "polygon", "line_idx", "type"}
    assert set(anns[0]["polygon"]) == {f"{a}{i}" for i in range(4) for a in "xy"}


def test_archive_members_over_the_size_limit_are_not_read(
    gnhk_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The declared size is checked before decompressing, so a zip bomb is refused.
    monkeypatch.setattr("accuscan.gnhk.MAX_MEMBER_BYTES", 64)
    with pytest.raises(ValueError, match="too large"):
        next(iter_pages(gnhk_root, "test"))
