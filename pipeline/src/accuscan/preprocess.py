"""Image preprocessing shared with the browser (`web/src/ml/preprocess.ts`).

Both implementations must agree to 1e-4 on `shared/fixtures/preprocess.json`.
Every step uses integer order statistics or an explicit resampling kernel so
that numpy and TypeScript produce the same numbers. Bump `FEATURE_VERSION`
whenever the output for a given input changes.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FEATURE_VERSION = 1
INPUT_HEIGHT = 64
MAX_WIDTH = 1024
WIDTH_MULTIPLE = 4
CROP_PADDING = 4
# Below this spread between paper and darkest ink the image is treated as blank.
MIN_CONTRAST = 24
# Pixels darker than this fraction of the paper-to-ink range count as ink.
INK_THRESHOLD = 0.5

GrayImage = NDArray[np.uint8]
Tensor = NDArray[np.float32]


class EmptyImageError(ValueError):
    """The image has no visible ink (blank canvas, flat photo or faint noise)."""


def rgba_to_gray(rgba: NDArray[np.uint8]) -> GrayImage:
    """Composite RGBA on white, then convert to grey with integer BT.601 weights."""
    px = rgba.astype(np.int64)
    alpha = px[..., 3]
    rgb = (px[..., :3] * alpha[..., None] + 255 * (255 - alpha[..., None]) + 127) // 255
    gray = (rgb[..., 0] * 299 + rgb[..., 1] * 587 + rgb[..., 2] * 114 + 500) // 1000
    return np.asarray(gray, dtype=np.uint8)


def rgb_to_gray(rgb: NDArray[np.uint8]) -> GrayImage:
    px = rgb.astype(np.int64)
    gray = (px[..., 0] * 299 + px[..., 1] * 587 + px[..., 2] * 114 + 500) // 1000
    return gray.astype(np.uint8)


def _order_statistic(hist: NDArray[np.int64], k: int) -> int:
    """Smallest grey value v such that more than k pixels are <= v."""
    cumulative = np.cumsum(hist)
    return int(np.searchsorted(cumulative, k + 1))


def normalise_contrast(gray: GrayImage) -> NDArray[np.float64]:
    """Map paper to 1.0 and the darkest ink to 0.0, inverting light-on-dark input.

    The median is the paper level (text never covers half the image) and the
    extreme on the other side is the ink level.
    """
    if gray.ndim != 2 or gray.size == 0:
        raise EmptyImageError("expected a non-empty 2-D greyscale image")
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.int64)
    n = int(gray.size)
    median = _order_statistic(hist, (n - 1) // 2)
    lo = int(np.flatnonzero(hist)[0])
    hi = int(np.flatnonzero(hist)[-1])
    values = gray.astype(np.float64)
    if median - lo < hi - median:
        # Paper is closer to black than to white: light ink on a dark surface.
        values = 255.0 - values
        median, lo, hi = 255 - median, 255 - hi, 255 - lo
    paper, ink = float(median), float(lo)
    if paper - ink < MIN_CONTRAST:
        raise EmptyImageError("no ink found")
    result: NDArray[np.float64] = np.clip((values - ink) / (paper - ink), 0.0, 1.0)
    return result


def ink_bounds(norm: NDArray[np.float64]) -> tuple[int, int, int, int]:
    """Inclusive (top, bottom, left, right) of pixels darker than the threshold."""
    mask = norm < INK_THRESHOLD
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        raise EmptyImageError("no ink found")
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def crop_with_padding(norm: NDArray[np.float64], pad: int = CROP_PADDING) -> NDArray[np.float64]:
    top, bottom, left, right = ink_bounds(norm)
    h = bottom - top + 1 + 2 * pad
    w = right - left + 1 + 2 * pad
    out = np.ones((h, w), dtype=np.float64)
    y0, y1 = max(0, top - pad), min(norm.shape[0], bottom + pad + 1)
    x0, x1 = max(0, left - pad), min(norm.shape[1], right + pad + 1)
    out[y0 - (top - pad) : y1 - (top - pad), x0 - (left - pad) : x1 - (left - pad)] = norm[
        y0:y1, x0:x1
    ]
    return out


def resample_weights(src: int, dst: int) -> NDArray[np.float64]:
    """Dense (dst, src) matrix for an antialiased triangle (bilinear) filter.

    When shrinking, the triangle is widened by the scale factor so every input
    pixel contributes (like PIL's BILINEAR with reducing). Rows sum to 1.
    """
    scale = src / dst
    support = max(scale, 1.0)
    centres = (np.arange(dst, dtype=np.float64) + 0.5) * scale
    positions = np.arange(src, dtype=np.float64) + 0.5
    weights = np.clip(1.0 - np.abs(positions[None, :] - centres[:, None]) / support, 0.0, None)
    sums = weights.sum(axis=1, keepdims=True)
    normalised: NDArray[np.float64] = weights / sums
    return normalised


def target_width(height: int, width: int) -> int:
    """Width after scaling to INPUT_HEIGHT, clamped to MAX_WIDTH (before padding)."""
    scaled = math.floor(width * INPUT_HEIGHT / height + 0.5)
    return min(MAX_WIDTH, max(1, scaled))


def padded_width(width: int) -> int:
    return max(WIDTH_MULTIPLE, -(-width // WIDTH_MULTIPLE) * WIDTH_MULTIPLE)


def preprocess(gray: GrayImage) -> Tensor:
    """Greyscale uint8 [H, W] -> float32 [1, 64, W'] in [-1, 1], ink positive.

    Steps: contrast-normalise (auto-invert), crop to ink bounds plus 4 px, resize
    to height 64 keeping aspect (squeezing to 1024 px if wider), pad right to a
    multiple of 4 with paper, then map paper to -1 and ink to +1.
    """
    norm = normalise_contrast(gray)
    crop = crop_with_padding(norm)
    h, w = crop.shape
    new_w = target_width(h, w)
    resized = resample_weights(h, INPUT_HEIGHT) @ crop @ resample_weights(w, new_w).T
    canvas = np.ones((INPUT_HEIGHT, padded_width(new_w)), dtype=np.float64)
    canvas[:, :new_w] = resized
    out = (1.0 - 2.0 * canvas).astype(np.float32)
    return out[None, :, :]


def to_uint8(tensor: Tensor) -> NDArray[np.uint8]:
    """Quantise a preprocessed [1, 64, W] tensor for compact caching."""
    return np.asarray(np.round((tensor[0] + 1.0) * 127.5), dtype=np.uint8)


def from_uint8(cached: NDArray[np.uint8]) -> Tensor:
    return (cached.astype(np.float32) / np.float32(127.5) - np.float32(1.0))[None, :, :]
