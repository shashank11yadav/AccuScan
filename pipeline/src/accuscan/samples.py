"""Gallery images for the demo: canvas-style lines rendered in held-out fonts."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from .render import FontRef, RenderConfig, render_line, supported_chars

GALLERY_TEXTS = (
    "hello world",
    "Meet at 7:30pm",
    "buy milk and eggs",
    "Thank you!",
    "the quick brown fox",
    "Call me on Friday",
    "42 Baker Street",
    "see you soon",
)
PADDING = 24


def write_samples(
    fonts: Sequence[tuple[str, FontRef]],
    out_dir: Path,
    seed: int,
    log: Callable[[str], None] = print,
) -> Path:
    """Render GALLERY_TEXTS in canvas style, cycling through the given fonts.

    Fonts must be held-out families so the gallery shows writing the model was
    never trained on. Writes sample-N.png plus samples.json (text and font).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cfg = RenderConfig(min_size=56, max_size=64, max_rotation_deg=2.0, max_shear=0.2)
    entries = []
    for i, text in enumerate(GALLERY_TEXTS):
        name, font = fonts[i % len(fonts)]
        if not set(text) <= supported_chars(font):
            continue
        img = render_line(text, font, "canvas", rng, cfg)
        ys, xs = np.nonzero(img < 128)
        crop = img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        padded = np.pad(crop, PADDING, constant_values=255)
        file = f"sample-{i + 1}.png"
        Image.fromarray(padded).save(out_dir / file, optimize=True)
        entries.append({"file": file, "text": text, "font": name})
    manifest = out_dir / "samples.json"
    manifest.write_text(json.dumps({"samples": entries}, indent=2) + "\n", encoding="utf-8")
    log(f"{len(entries)} gallery samples -> {out_dir}")
    return manifest
