"""GNHK (GoodNotes Handwriting Kollection) reader and word/line cropper.

Download layout (from goodnotes.com/gnhk, CC BY 4.0), placed in _datasets/gnhk/:

    train_data.zip   -> train/eng_<REGION>_<NNN>.jpg + .json
    test_data.zip    -> test/eng_<REGION>_<NNN>.jpg + .json

The archives may also be extracted in place (train/ and test/ folders). Each
JSON file is a list of objects:

    {"text": "Proof",
     "polygon": {"x0": 845, "y0": 1592, "x1": 859, "y1": 1809,
                 "x2": 1188, "y2": 1888, "x3": 1300, "y3": 1588},
     "line_idx": 4,
     "type": "H"}

`text` has no whitespace and uses printable ASCII plus the pound sign, or one of
the special tokens %math%, %SC% (illegible scribble) and %NA% (not text).
`type` is H (handwritten) or P (printed). The quadrilateral starts at the
top-left corner; this module orders the four points by geometry, so either
winding direction works.

Kept: handwritten words whose text fits the charset. A line is kept only if
every word on it is kept, so line labels are never missing a word.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .charset import normalise_text
from .preprocess import EmptyImageError, preprocess, rgb_to_gray, to_uint8
from .shards import Sample, ShardWriter

Split = Literal["train", "test"]
SPECIAL_TOKENS = frozenset({"%math%", "%SC%", "%NA%"})
ARCHIVES: dict[Split, str] = {"train": "train_data.zip", "test": "test_data.zip"}
MAX_IMAGE_BYTES = 40 * 1024 * 1024
CROP_MARGIN = 0.08  # of the word height, so ascenders cut by a tight polygon survive

Quad = NDArray[np.float64]  # [4, 2] ordered TL, TR, BR, BL


@dataclass(frozen=True)
class Annotation:
    text: str
    quad: Quad
    line_idx: int
    handwritten: bool


@dataclass(frozen=True)
class Page:
    split: Split
    page_id: str
    image: NDArray[np.uint8]  # greyscale
    annotations: tuple[Annotation, ...]


@dataclass
class PrepStats:
    pages: int = 0
    words_kept: int = 0
    words_special: int = 0
    words_printed: int = 0
    words_charset: int = 0
    lines_kept: int = 0
    lines_dropped: int = 0
    crops_empty: int = 0


def order_quad(points: NDArray[np.float64]) -> Quad:
    """Order four points as TL, TR, BR, BL (y down) for rotations under 45 degrees."""
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    by_y = pts[np.argsort(pts[:, 1], kind="stable")]
    top = by_y[:2][np.argsort(by_y[:2, 0], kind="stable")]
    bottom = by_y[2:][np.argsort(by_y[2:, 0], kind="stable")]
    return np.array([top[0], top[1], bottom[1], bottom[0]])


def parse_annotations(raw: object) -> tuple[Annotation, ...]:
    if not isinstance(raw, list):
        raise ValueError("GNHK annotation must be a JSON list")
    out = []
    for obj in raw:
        poly = obj["polygon"]
        pts = np.array([[poly[f"x{i}"], poly[f"y{i}"]] for i in range(4)], dtype=np.float64)
        out.append(
            Annotation(
                text=str(obj["text"]),
                quad=order_quad(pts),
                line_idx=int(obj["line_idx"]),
                handwritten=obj.get("type", "H") == "H",
            )
        )
    return tuple(out)


def _decode_image(data: bytes) -> NDArray[np.uint8]:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image too large")
    with Image.open(io.BytesIO(data)) as img:
        rgb = np.asarray(img.convert("RGB"))
    return rgb_to_gray(rgb)


def _fit_orientation(
    image: NDArray[np.uint8], anns: tuple[Annotation, ...], data: bytes
) -> NDArray[np.uint8]:
    """Apply EXIF orientation only if the polygons do not fit the raw pixel frame."""
    if not anns:
        return image
    max_x = max(float(a.quad[:, 0].max()) for a in anns)
    max_y = max(float(a.quad[:, 1].max()) for a in anns)
    h, w = image.shape
    if max_x <= w + 2 and max_y <= h + 2:
        return image
    with Image.open(io.BytesIO(data)) as img:
        from PIL import ImageOps

        oriented = ImageOps.exif_transpose(img).convert("RGB")
    return rgb_to_gray(np.asarray(oriented))


def _members(root: Path, split: Split) -> Iterator[tuple[str, Callable[[str], bytes]]]:
    """(page_id, reader) for every annotated page, from the zip or an extracted folder."""
    archive = root / ARCHIVES[split]
    folder = root / split
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            for name in sorted(names):
                path = PurePosixPath(name)
                if path.suffix != ".json" or ".." in path.parts or path.name.startswith("."):
                    continue
                image_name = str(path.with_suffix(".jpg"))
                if image_name not in names:
                    continue
                stem = str(path.with_suffix(""))

                def read(suffix: str, stem: str = stem) -> bytes:
                    return zf.read(stem + suffix)

                yield path.stem, read
    elif folder.is_dir():
        for json_path in sorted(folder.glob("*.json")):
            if not json_path.with_suffix(".jpg").exists():
                continue

            def read_file(suffix: str, base: Path = json_path) -> bytes:
                return base.with_suffix(suffix).read_bytes()

            yield json_path.stem, read_file
    else:
        raise FileNotFoundError(
            f"GNHK {split} split not found: expected {archive} or {folder}/ "
            "(download from https://goodnotes.com/gnhk)"
        )


def iter_pages(root: Path, split: Split) -> Iterator[Page]:
    for page_id, read in _members(root, split):
        anns = parse_annotations(json.loads(read(".json").decode("utf-8")))
        data = read(".jpg")
        image = _fit_orientation(_decode_image(data), anns, data)
        yield Page(split, page_id, image, anns)


def warp_quad(
    image: NDArray[np.uint8], quad: Quad, margin: float = CROP_MARGIN
) -> NDArray[np.uint8]:
    """Perspective-rectify a quadrilateral (plus a margin) into an upright crop."""
    tl, tr, br, bl = quad
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    width, height = max(2.0, float(width)), max(2.0, float(height))
    m = margin * height
    dst = np.array(
        [[m, m], [m + width, m], [m + width, m + height], [m, m + height]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    size = (round(width + 2 * m), round(height + 2 * m))
    out = np.asarray(
        cv2.warpPerspective(
            image, matrix, size, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE
        ),
        dtype=np.uint8,
    )
    return out


def line_quad(quads: list[Quad]) -> Quad:
    """Rotated rectangle around all word quads, aligned with their mean baseline."""
    direction = np.zeros(2)
    for q in quads:
        direction += (q[1] - q[0]) + (q[2] - q[3])
    norm = float(np.linalg.norm(direction))
    u = direction / norm if norm > 0 else np.array([1.0, 0.0])
    v = np.array([-u[1], u[0]])
    pts = np.concatenate(quads)
    pu, pv = pts @ u, pts @ v
    u0, u1, v0, v1 = pu.min(), pu.max(), pv.min(), pv.max()
    corners = [u0 * u + v0 * v, u1 * u + v0 * v, u1 * u + v1 * v, u0 * u + v1 * v]
    return np.array(corners)


def word_order(quads: list[Quad]) -> list[int]:
    """Indices of word quads sorted along the line direction."""
    lq = line_quad(quads)
    u = lq[1] - lq[0]
    centres = [float(q.mean(axis=0) @ u) for q in quads]
    return sorted(range(len(quads)), key=lambda i: centres[i])


def _keep_word(ann: Annotation, stats: PrepStats) -> str | None:
    if ann.text in SPECIAL_TOKENS:
        stats.words_special += 1
        return None
    if not ann.handwritten:
        stats.words_printed += 1
        return None
    text = normalise_text(ann.text)
    if not text:
        stats.words_charset += 1
        return None
    return text


def page_samples(page: Page, stats: PrepStats) -> Iterator[Sample]:
    stats.pages += 1
    lines: dict[int, list[tuple[Annotation, str | None]]] = {}
    for ann in page.annotations:
        text = _keep_word(ann, stats)
        lines.setdefault(ann.line_idx, []).append((ann, text))
        if text is None:
            continue
        crop = warp_quad(page.image, ann.quad)
        try:
            pixels = to_uint8(preprocess(crop))
        except EmptyImageError:
            stats.crops_empty += 1
            continue
        stats.words_kept += 1
        yield Sample(pixels, text, "gnhk-word", raw=crop)
    for _, words in sorted(lines.items()):
        if any(text is None for _, text in words) or len(words) < 2:
            stats.lines_dropped += 1
            continue
        quads = [ann.quad for ann, _ in words]
        label = " ".join(str(words[i][1]) for i in word_order(quads))
        crop = warp_quad(page.image, line_quad(quads), margin=0.04)
        try:
            pixels = to_uint8(preprocess(crop))
        except EmptyImageError:
            stats.crops_empty += 1
            continue
        stats.lines_kept += 1
        yield Sample(pixels, label, "gnhk-line", raw=crop)


def validation_pages(page_ids: list[str], fraction: float, seed: int) -> set[str]:
    """A fixed subset of training pages (by id) held out for early stopping."""
    ordered = sorted(page_ids)
    rng = np.random.default_rng(seed)
    count = max(1, round(len(ordered) * fraction)) if ordered else 0
    return {ordered[int(i)] for i in rng.permutation(len(ordered))[:count]}


def prepare_gnhk(
    root: Path, out_dir: Path, val_fraction: float, seed: int, log: Callable[[str], None] = print
) -> dict[str, PrepStats]:
    """Write {train,val,test}-{word,line}.npz shards under out_dir."""
    train_ids = [page_id for page_id, _ in _members(root, "train")]
    val_ids = validation_pages(train_ids, val_fraction, seed)
    all_stats: dict[str, PrepStats] = {}
    for split in ("train", "test"):
        writers = {
            (part, kind): ShardWriter(out_dir / f"{part}-{kind}.npz")
            for part in (("train", "val") if split == "train" else ("test",))
            for kind in ("word", "line")
        }
        stats = PrepStats()
        for page in iter_pages(root, split):
            part = "val" if page.page_id in val_ids else split
            for sample in page_samples(page, stats):
                kind = "word" if sample.source == "gnhk-word" else "line"
                writers[(part, kind)].add(sample)
        for (part, kind), writer in writers.items():
            writer.close()
            log(f"{part}-{kind}: {len(writer)} crops")
        all_stats[split] = stats
        log(f"{split}: {stats}")
    return all_stats
