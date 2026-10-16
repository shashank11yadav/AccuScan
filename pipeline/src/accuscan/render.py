"""Render a text line in a handwriting font, in "paper" or "canvas" style.

- paper: ink texture, tinted and unevenly lit background, noise, ruled lines,
  blur, slight perspective and JPEG artefacts, like a phone photo of a note.
- canvas: white background and a uniform round-capped stroke of random width
  with a wobbly path, like a word drawn with a mouse or finger.

All randomness comes from the caller's generator, so a seed fixes the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

Style = Literal["paper", "canvas"]
FontRef = Path | None  # None selects Pillow's built-in font (used by the smoke run)


@dataclass(frozen=True)
class RenderConfig:
    min_size: int = 30
    max_size: int = 64
    max_rotation_deg: float = 3.0
    max_shear: float = 0.35


@lru_cache(maxsize=256)
def _font(ref: FontRef, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if ref is None:
        return ImageFont.load_default(size)
    return ImageFont.truetype(str(ref), size)


@lru_cache(maxsize=128)
def supported_chars(ref: FontRef) -> frozenset[str]:
    """Printable ASCII characters the font has glyphs for (space always counts)."""
    ascii_chars = {chr(c) for c in range(32, 127)}
    if ref is None:
        return frozenset(ascii_chars)
    cmap = TTFont(str(ref), lazy=True).getBestCmap()
    return frozenset({c for c in ascii_chars if ord(c) in cmap} | {" "})


def _draw_text(text: str, ref: FontRef, size: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Ink coverage in [0, 1] (1 = ink) for the text, drawn whole or per character."""
    font = _font(ref, size)
    per_char = rng.random() < 0.5
    margin = size
    width = int(font.getlength(text) * 1.3) + 2 * margin
    height = int(size * 2.2)
    img = Image.new("L", (max(width, 8), height), 0)
    draw = ImageDraw.Draw(img)
    baseline = int(size * 0.4)
    if not per_char:
        draw.text((margin, baseline), text, font=font, fill=255)
    else:
        x = float(margin)
        spacing = rng.uniform(0.92, 1.12)
        for ch in text:
            dy = rng.normal(0.0, 0.035 * size)
            draw.text((x, baseline + dy), ch, font=font, fill=255)
            x += font.getlength(ch) * spacing + rng.normal(0.0, 0.03 * size)
    coverage: NDArray[np.float64] = np.asarray(img, dtype=np.float64) / 255.0
    return coverage


def _affine(
    mask: NDArray[np.float64], cfg: RenderConfig, rng: np.random.Generator
) -> NDArray[np.float64]:
    h, w = mask.shape
    angle = np.deg2rad(rng.uniform(-cfg.max_rotation_deg, cfg.max_rotation_deg))
    shear = rng.uniform(-cfg.max_shear, cfg.max_shear * 0.5)
    sy = rng.uniform(0.9, 1.1)
    c, s = np.cos(angle), np.sin(angle)
    m = np.array([[c, -s], [s, c]]) @ np.array([[1.0, shear], [0.0, sy]])
    centre = np.array([w / 2, h / 2])
    offset = centre - m @ centre
    matrix = np.hstack([m, offset[:, None]])
    out = np.asarray(
        cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0),
        dtype=np.float64,
    )
    return out


def _elastic(
    mask: NDArray[np.float64], amplitude: float, sigma: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    h, w = mask.shape
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)), (0, 0), sigma)
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)), (0, 0), sigma)
    scale = amplitude / max(1e-6, float(np.abs(np.stack([dx, dy])).max()))
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (xs + dx * scale).astype(np.float32)
    map_y = (ys + dy * scale).astype(np.float32)
    out = np.asarray(
        cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderValue=0.0), dtype=np.float64
    )
    return out


def _disk(radius: float) -> NDArray[np.uint8]:
    r = max(1, int(np.ceil(radius)))
    ys, xs = np.mgrid[-r : r + 1, -r : r + 1]
    return np.asarray((xs**2 + ys**2) <= radius**2 + 0.25, dtype=np.uint8)


def _canvas_stroke(
    mask: NDArray[np.float64], size: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Redraw glyphs as a uniform stroke: wobble, skeletonise, dilate, antialias."""
    wobbly = _elastic(mask, amplitude=rng.uniform(0.5, 2.0), sigma=size * 0.25, rng=rng)
    skeleton = np.asarray(skeletonize(wobbly > 0.5), dtype=np.uint8)  # type: ignore[no-untyped-call]
    radius = rng.uniform(0.012, 0.045) * size + 0.5
    thick = cv2.dilate(skeleton, _disk(radius))
    if rng.random() < 0.5:
        # Stroke-width variation: blend in a thicker version under a smooth mask.
        thicker = cv2.dilate(skeleton, _disk(radius * rng.uniform(1.15, 1.4)))
        blend = cv2.GaussianBlur(rng.random(mask.shape), (0, 0), size * 0.5)
        blend = (blend - blend.min()) / max(1e-6, float(np.ptp(blend)))
        thick = np.where(blend > 0.5, thicker, thick)
    out = np.asarray(cv2.GaussianBlur(thick.astype(np.float64), (0, 0), 0.6), dtype=np.float64)
    return np.asarray(np.clip(out, 0.0, 1.0), dtype=np.float64)


def _paper(mask: NDArray[np.float64], size: int, rng: np.random.Generator) -> NDArray[np.uint8]:
    h, w = mask.shape
    paper = rng.uniform(165, 250)
    ink = rng.uniform(0, min(110, paper - 60))
    # Uneven lighting: a random linear gradient across the crop.
    gx, gy = rng.uniform(-25, 25, size=2)
    ys, xs = np.mgrid[0:h, 0:w]
    background = paper + gx * (xs / max(w, 1) - 0.5) + gy * (ys / max(h, 1) - 0.5)
    if rng.random() < 0.3:
        row = int(rng.uniform(0.6, 0.85) * h)
        background[row : row + max(1, size // 30), :] -= rng.uniform(20, 60)
    if rng.random() < 0.5:
        # Pen pressure and ink texture.
        texture = cv2.GaussianBlur(rng.random((h, w)), (0, 0), 1.5)
        mask = mask * np.clip(0.6 + texture, 0.0, 1.0)
    img = background - mask * (background - ink)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.3))
    img = img + rng.normal(0, rng.uniform(1, 9), size=img.shape)
    if rng.random() < 0.4:
        d = 0.04 * h
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = (src + rng.uniform(-d, d, size=src.shape)).astype(np.float32)
        warp = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, warp, (w, h), borderMode=cv2.BORDER_REPLICATE)
    out = np.asarray(np.clip(img, 0, 255), dtype=np.uint8)
    if rng.random() < 0.4:
        quality = int(rng.integers(30, 90))
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out = np.asarray(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
    return out


def render_line(
    text: str,
    font: FontRef,
    style: Style,
    rng: np.random.Generator,
    cfg: RenderConfig | None = None,
) -> NDArray[np.uint8]:
    """Greyscale uint8 image of the text: dark ink on a light background."""
    cfg = cfg or RenderConfig()
    size = int(rng.integers(cfg.min_size, cfg.max_size + 1))
    mask = _affine(_draw_text(text, font, size, rng), cfg, rng)
    if style == "canvas":
        stroke = _canvas_stroke(mask, size, rng)
        return np.clip(255.0 * (1.0 - stroke), 0, 255).astype(np.uint8)
    return _paper(mask, size, rng)


def save_png(img: NDArray[np.uint8], path: Path) -> None:
    Image.fromarray(img).save(path, optimize=True)
