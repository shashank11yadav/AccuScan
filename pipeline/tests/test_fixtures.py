"""The committed parity fixtures must match what the pipeline computes now."""

import base64
import json
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from accuscan.charset import encode
from accuscan.decode import Lexicon, align_confidences, beam_search, greedy_decode
from accuscan.fixtures import apply_exif_orientation
from accuscan.paths import FIXTURES_DIR
from accuscan.preprocess import EmptyImageError, preprocess, rgba_to_gray

TOLERANCE = 1e-4


def _image(spec: dict[str, Any]) -> NDArray[np.uint8]:
    data = np.frombuffer(base64.b64decode(spec["data"]), dtype=np.uint8)
    shape = (int(spec["height"]), int(spec["width"]))
    channels = int(spec["channels"])
    return data.reshape(shape if channels == 1 else (*shape, channels))


def _load(name: str) -> dict[str, Any]:
    path = FIXTURES_DIR / name
    if not path.exists():
        pytest.fail(f"{path} missing; run `uv run accuscan export-fixtures`")
    data: dict[str, Any] = json.loads(path.read_text())
    return data


PREPROCESS: list[dict[str, Any]] = _load("preprocess.json")["cases"]
DECODE = _load("decode.json")


@pytest.mark.parametrize("case", PREPROCESS, ids=lambda c: c["name"])
def test_preprocess_matches_fixture(case: dict[str, Any]) -> None:
    img = _image(case["input"])
    if img.ndim == 3:
        img = rgba_to_gray(img)
    if "exifOrientation" in case:
        img = apply_exif_orientation(img, int(case["exifOrientation"]))
    expected = case["expected"]
    if "error" in expected:
        with pytest.raises(EmptyImageError):
            preprocess(img)
        return
    out = preprocess(img)
    assert list(out.shape) == expected["shape"]
    np.testing.assert_allclose(out.ravel(), np.array(expected["data"]), atol=TOLERANCE)


@pytest.mark.parametrize("case", DECODE["cases"], ids=lambda c: c["name"])
def test_decode_matches_fixture(case: dict[str, Any]) -> None:
    lexicon = Lexicon.from_counts(DECODE["lexicon"])
    width: int = DECODE["beamWidth"]
    lp = np.array(case["logProbs"])
    assert greedy_decode(lp).text == case["greedy"]["text"]
    assert beam_search(lp, None, beam_width=width).text == case["beam"]["text"]
    dictionary = beam_search(lp, lexicon, beam_width=width, lm_weight=DECODE["lmWeight"])
    assert dictionary.text == case["dictionary"]["text"]
    assert dictionary.fallback == case["dictionary"]["fallback"]
    np.testing.assert_allclose(
        np.array(dictionary.confidences),
        np.array(case["dictionary"]["confidences"]),
        atol=TOLERANCE,
    )
    alignment = case["alignment"]
    np.testing.assert_allclose(
        np.array(align_confidences(lp, encode(alignment["labels"]))),
        np.array(alignment["confidences"]),
        atol=TOLERANCE,
    )


def test_fixture_covers_the_dictionary_correcting_a_letter() -> None:
    cases = {c["name"]: c for c in DECODE["cases"]}
    prior = cases["lexicon-prior"]
    assert prior["greedy"]["text"] != prior["dictionary"]["text"]
    assert prior["dictionary"]["text"] == "cat"
    assert cases["unknown-word-fallback"]["dictionary"]["fallback"] is True


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_matches_pillow(orientation: int) -> None:
    stored = np.random.default_rng(orientation).integers(0, 256, size=(3, 5), dtype=np.uint8)
    reference = np.asarray(_pillow_orient(Image.fromarray(stored), orientation))
    np.testing.assert_array_equal(apply_exif_orientation(stored, orientation), reference)


def _pillow_orient(img: Image.Image, orientation: int) -> Image.Image:
    method = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    return img.transpose(method) if method is not None else img
