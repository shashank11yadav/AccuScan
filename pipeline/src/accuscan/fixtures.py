"""Parity fixtures shared with the TypeScript tests (shared/fixtures/*.json).

preprocess.json: greyscale or RGBA inputs and the expected [1, 64, W] tensors
(or the expected "empty" error), including a photo stored sideways with EXIF
orientation 6, which must match the upright image once oriented.

decode.json: log-prob matrices with the expected greedy, beam and dictionary
beam results for a small lexicon.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from .charset import CHARSET, NUM_CLASSES, encode
from .decode import Lexicon, align_confidences, beam_search, greedy_decode
from .preprocess import EmptyImageError, preprocess, rgba_to_gray

FIXTURE_LEXICON = {"the": 50, "cat": 20, "cot": 2, "sat": 10, "hello": 8, "world": 6, "don't": 3}


def apply_exif_orientation(img: NDArray[np.uint8], orientation: int) -> NDArray[np.uint8]:
    """Turn stored pixels into display orientation (EXIF tag 274, values 1-8)."""
    if orientation == 2:
        out = img[:, ::-1]
    elif orientation == 3:
        out = img[::-1, ::-1]
    elif orientation == 4:
        out = img[::-1, :]
    elif orientation == 5:
        out = np.swapaxes(img, 0, 1)  # transpose
    elif orientation == 6:
        out = np.rot90(img, k=-1)  # 90 degrees clockwise
    elif orientation == 7:
        out = np.swapaxes(img[::-1, ::-1], 0, 1)  # transverse
    elif orientation == 8:
        out = np.rot90(img, k=1)  # 90 degrees anticlockwise
    else:
        out = img
    return np.ascontiguousarray(out)


def _text_image(text: str, size: int, pad: int, ink: int, paper: int) -> NDArray[np.uint8]:
    font = ImageFont.load_default(size)
    box = font.getbbox(text)
    img = Image.new("L", (int(box[2]) + 2 * pad, int(box[3]) + 2 * pad), paper)
    ImageDraw.Draw(img).text((pad, pad), text, font=font, fill=ink)
    return np.asarray(img, dtype=np.uint8).copy()


def _encode_image(arr: NDArray[np.uint8]) -> dict[str, Any]:
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    return {
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "channels": channels,
        "data": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii"),
    }


def _expected(gray: NDArray[np.uint8]) -> dict[str, Any]:
    try:
        out = preprocess(gray)
    except EmptyImageError:
        return {"error": "empty"}
    values = [round(float(v), 6) for v in out.ravel()]
    return {"shape": list(out.shape), "data": values}


def preprocess_cases() -> list[dict[str, Any]]:
    rng = np.random.default_rng(42)
    word = _text_image("Hand 42", 28, 6, 30, 235)
    cases: list[dict[str, Any]] = []

    def add(
        name: str, image: NDArray[np.uint8], expected_from: NDArray[np.uint8], **extra: Any
    ) -> None:
        cases.append(
            {
                "name": name,
                "input": _encode_image(image),
                "expected": _expected(expected_from),
                **extra,
            }
        )

    add("word-upscale", word, word)
    add("inverted", (255 - word).astype(np.uint8), (255 - word).astype(np.uint8))
    grey = np.clip(
        110 + (word.astype(np.int32) - 30) * 60 // 205 + rng.integers(-3, 4, word.shape), 0, 255
    )
    add("grey-paper-noise", grey.astype(np.uint8), grey.astype(np.uint8))
    big = _text_image("List", 110, 20, 10, 245)
    add("downscale", big, big)
    wide = np.full((24, 420), 250, np.uint8)
    wide[6:18, 4:416] = 20
    wide[9:15, 100:140] = 250
    add("very-wide-squeezed", wide, wide)
    edge = np.full((50, 80), 240, np.uint8)
    edge[0:20, 0:30] = 15  # ink touching the border: padding must be paper
    add("ink-at-edge", edge, edge)
    rgba = np.zeros((48, 120, 4), np.uint8)
    rgba[14:34, 10:110, 3] = 255  # opaque black bar on a transparent canvas
    rgba[20:28, 30:60, :3] = 200
    add("rgba-canvas", rgba, rgba_to_gray(rgba))
    sideways = np.ascontiguousarray(np.rot90(word, k=1))
    add("exif-orientation-6", sideways, word, exifOrientation=6)
    add("blank", np.full((40, 40), 255, np.uint8), np.full((40, 40), 255, np.uint8))
    faint = (238 + rng.integers(-6, 7, (30, 60))).astype(np.uint8)
    add("faint-noise", faint, faint)
    return cases


def _peaked(
    text: str, rng: np.random.Generator, ambiguous: dict[int, str] | None = None
) -> NDArray[np.float64]:
    """Log-probs for a path spelling text with 1-2 frames per char and blanks between."""
    path: list[int] = [0]
    for i, ch in enumerate(text):
        path += [encode(ch)[0]] * int(rng.integers(1, 3))
        if i + 1 < len(text) and (text[i + 1] == ch or rng.random() < 0.6):
            path.append(0)
    path.append(0)
    logits = rng.normal(0.0, 1.0, size=(len(path), NUM_CLASSES))
    for t, k in enumerate(path):
        logits[t, k] += 7.0
    for t, alt in (ambiguous or {}).items():
        logits[t, encode(alt)[0]] = logits[t].max() + 0.3
    logits -= logits.max(axis=1, keepdims=True)
    log_probs: NDArray[np.float64] = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return log_probs


def decode_cases() -> list[dict[str, Any]]:
    rng = np.random.default_rng(7)
    lexicon = Lexicon.from_counts(FIXTURE_LEXICON)
    specs: list[tuple[str, NDArray[np.float64]]] = [
        ("plain", _peaked("the cat sat", rng)),
        ("repeats", _peaked("hello", rng)),
        ("digits-punctuation", _peaked("cat 42, don't!", rng)),
    ]
    noisy = _peaked("cat", rng)
    t = int(np.argmax(noisy[:, encode("a")[0]]))
    noisy[t, encode("o")[0]] = noisy[t, encode("a")[0]] + 0.2  # 'o' slightly beats 'a'
    specs.append(("lexicon-prior", noisy))
    specs.append(("unknown-word-fallback", _peaked("xqz", rng)))
    specs.append(("all-blank", _peaked("", rng)))
    cases = []
    for name, lp in specs:
        greedy = greedy_decode(lp)
        plain = beam_search(lp, None, beam_width=8)
        dictionary = beam_search(lp, lexicon, beam_width=8, lm_weight=0.5)
        cases.append(
            {
                "name": name,
                "logProbs": [[float(v) for v in row] for row in lp],
                "greedy": {"text": greedy.text, "confidences": greedy.confidences},
                "beam": {
                    "text": plain.text,
                    "confidences": plain.confidences,
                    "fallback": plain.fallback,
                },
                "dictionary": {
                    "text": dictionary.text,
                    "confidences": dictionary.confidences,
                    "fallback": dictionary.fallback,
                },
                "alignment": {
                    "labels": greedy.text,
                    "confidences": align_confidences(lp, encode(greedy.text)),
                },
            }
        )
    return cases


def write_fixtures(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = out_dir / "preprocess.json"
    pre.write_text(json.dumps({"version": 1, "cases": preprocess_cases()}) + "\n", encoding="utf-8")
    dec = out_dir / "decode.json"
    payload = {
        "version": 1,
        "charset": list(CHARSET),
        "lexicon": FIXTURE_LEXICON,
        "beamWidth": 8,
        "lmWeight": 0.5,
        "cases": decode_cases(),
    }
    dec.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return [pre, dec]
