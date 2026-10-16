"""A small synthetic dataset in the GNHK download format, for tests and the smoke run.

Pages are photo-like grey images with words drawn in rotated boxes. The JSON
matches the published format, including special tokens, a printed word and a
word outside the charset, so the filters are exercised.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from .gnhk import ARCHIVES, Split

LINES: tuple[tuple[tuple[str, str], ...], ...] = (
    (("Shopping", "H"), ("list", "H")),
    (("milk", "H"), ("2", "H"), ("eggs", "H")),
    (("call", "H"), ("%SC%", "H"), ("mum", "H")),
    (("Total", "P"), ("£12", "H")),
    (("see", "H"), ("you", "H"), ("soon!", "H")),
)


def _rotate(
    points: NDArray[np.float64], angle: float, centre: NDArray[np.float64]
) -> NDArray[np.float64]:
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    rotated: NDArray[np.float64] = (points - centre) @ rotation.T + centre
    return rotated


def make_page(
    seed: int, angle_deg: float = 4.0
) -> tuple[NDArray[np.uint8], list[dict[str, object]]]:
    """(RGB page, GNHK-style annotation list) with words on rotated lines."""
    rng = np.random.default_rng(seed)
    width, height = 900, 600
    page = Image.new("L", (width, height), 225)
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(40)
    annotations: list[dict[str, object]] = []
    centre = np.array([width / 2, height / 2])
    angle = np.deg2rad(angle_deg)
    for line_idx, line in enumerate(LINES):
        x = 60.0
        y = 50.0 + line_idx * 105.0
        for text, kind in line:
            label = "~~~" if text == "%SC%" else text
            box = draw.textbbox((0, 0), label, font=font)
            w, h = box[2] - box[0] + 16, box[3] - box[1] + 20
            draw.text((x + 8 - box[0], y + 10 - box[1]), label, font=font, fill=30)
            corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
            tl, tr, br, bl = _rotate(corners, angle, centre)
            # GNHK lists the top-left corner first and then goes down the left edge.
            ordered = (tl, bl, br, tr)
            polygon = {
                f"{axis}{i}": round(float(p[j]))
                for i, p in enumerate(ordered)
                for j, axis in enumerate("xy")
            }
            annotations.append(
                {"text": text, "polygon": polygon, "line_idx": line_idx, "type": kind}
            )
            x += w + 30
    arr = np.asarray(page, dtype=np.float64)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -angle_deg, 1.0)
    arr = np.asarray(
        cv2.warpAffine(arr, matrix, (width, height), borderValue=225.0), dtype=np.float64
    )
    arr = arr + rng.normal(0, 4, size=arr.shape)
    gray = np.clip(arr, 0, 255).astype(np.uint8)
    rgb: NDArray[np.uint8] = np.repeat(gray[:, :, None], 3, axis=2)
    return rgb, annotations


def write_fixture(root: Path, pages_per_split: dict[Split, int] | None = None) -> Path:
    """Write train_data.zip and test_data.zip under root; returns root."""
    counts = pages_per_split or {"train": 4, "test": 2}
    root.mkdir(parents=True, exist_ok=True)
    regions = ("EU", "NA", "AF", "AS")
    for split, n in counts.items():
        with zipfile.ZipFile(root / ARCHIVES[split], "w") as zf:
            for i in range(n):
                seed = (0 if split == "train" else 1000) + i
                rgb, anns = make_page(seed, angle_deg=float((-1) ** i * (2 + i)))
                stem = f"{split}/eng_{regions[i % 4]}_{i + 1:03d}"
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
                zf.writestr(stem + ".jpg", buf.getvalue())
                zf.writestr(stem + ".json", json.dumps(anns, indent=1))
    return root
