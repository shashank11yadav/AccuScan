"""Batching for CTC training: width-bucketed batches, padding and light augmentation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from .charset import encode
from .model import DOWNSAMPLE
from .preprocess import INPUT_HEIGHT, MAX_WIDTH, padded_width
from .shards import Sample

WIDTH_BUCKET = 64


@dataclass(frozen=True)
class Example:
    pixels: NDArray[np.uint8]  # [64, W], 0 = paper, 255 = ink
    labels: tuple[int, ...]
    text: str
    source: str


@dataclass
class Batch:
    images: torch.Tensor  # [B, 1, 64, W] in [-1, 1]
    targets: torch.Tensor  # concatenated label ids
    input_lengths: torch.Tensor  # frames per sample
    target_lengths: torch.Tensor
    texts: list[str]


def required_frames(labels: Sequence[int]) -> int:
    """CTC needs one frame per label plus a blank between repeated labels."""
    repeats = sum(1 for a, b in pairwise(labels) if a == b)
    return len(labels) + repeats


def to_examples(samples: Sequence[Sample]) -> list[Example]:
    """Drop samples whose image has too few frames for their label."""
    out = []
    for s in samples:
        labels = tuple(encode(s.text))
        if labels and s.pixels.shape[1] // DOWNSAMPLE >= required_frames(labels):
            out.append(Example(s.pixels, labels, s.text, s.source))
    return out


def augment(pixels: NDArray[np.uint8], rng: np.random.Generator) -> NDArray[np.uint8]:
    """Random width stretch and stroke thickness change, then re-pad to a multiple of 4."""
    out = pixels
    if rng.random() < 0.7:
        scale = rng.uniform(0.8, 1.2)
        width = int(np.clip(round(out.shape[1] * scale), 4, MAX_WIDTH))
        out = np.asarray(
            cv2.resize(out, (width, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR), dtype=np.uint8
        )
    r = rng.random()
    if r < 0.15:
        out = np.asarray(cv2.dilate(out, np.ones((2, 2), np.uint8)), dtype=np.uint8)
    elif r < 0.3:
        out = np.asarray(cv2.erode(out, np.ones((2, 2), np.uint8)), dtype=np.uint8)
    width = padded_width(out.shape[1])
    if width != out.shape[1]:
        out = np.pad(out, ((0, 0), (0, width - out.shape[1])))
    return np.ascontiguousarray(out)


def collate(
    examples: Sequence[Example],
    rng: np.random.Generator | None = None,
    noise: float = 0.0,
    width_bucket: int = WIDTH_BUCKET,
) -> Batch:
    """Pad to a multiple of `width_bucket` columns.

    Few distinct tensor shapes matter on GPUs that compile kernels per shape
    (MPS especially); the padding is paper (-1) and lies beyond input_lengths.
    """
    images = [augment(e.pixels, rng) if rng is not None else e.pixels for e in examples]
    width = bucket_width(max(img.shape[1] for img in images), width_bucket)
    batch = np.zeros((len(images), 1, INPUT_HEIGHT, width), dtype=np.float32)
    lengths = []
    for i, img in enumerate(images):
        batch[i, 0, :, : img.shape[1]] = img
        lengths.append(img.shape[1] // DOWNSAMPLE)
    tensor = torch.from_numpy(batch / np.float32(127.5) - np.float32(1.0))
    if rng is not None and noise > 0:
        tensor = tensor + noise * float(rng.random()) * torch.randn_like(tensor)
    # Augmentation can shrink an image below what its label needs; CTC's
    # zero_infinity drops those rare cases instead of producing inf losses.
    return Batch(
        images=tensor,
        targets=torch.tensor([t for e in examples for t in e.labels], dtype=torch.long),
        input_lengths=torch.tensor(lengths, dtype=torch.long),
        target_lengths=torch.tensor([len(e.labels) for e in examples], dtype=torch.long),
        texts=[e.text for e in examples],
    )


def bucket_width(width: int, bucket: int = WIDTH_BUCKET) -> int:
    return -(-width // bucket) * bucket


def bucketed_batches(
    examples: Sequence[Example],
    batch_size: int,
    rng: np.random.Generator | None,
    max_columns: int | None = None,
    drop_last: bool = False,
) -> Iterator[list[Example]]:
    """Batches of examples from the same width bucket. Shuffled when rng is given.

    The batch size of a bucket is fixed (`batch_size`, reduced so batch size x
    padded width stays under `max_columns`), so training sees only a few
    distinct tensor shapes and memory use stays flat for long lines. With
    `drop_last`, each bucket's incomplete final batch is skipped (training
    reshuffles every epoch, so nothing is lost for good); a new shape costs a
    kernel compile on MPS.
    """
    groups: dict[int, list[int]] = {}
    order = np.arange(len(examples)) if rng is None else rng.permutation(len(examples))
    for i in order.tolist():
        groups.setdefault(bucket_width(examples[i].pixels.shape[1]), []).append(i)
    batches: list[list[int]] = []
    for width, members in sorted(groups.items()):
        size = batch_size if max_columns is None else max(1, min(batch_size, max_columns // width))
        for j in range(0, len(members), size):
            if drop_last and j + size > len(members) and j > 0:
                break
            batches.append(members[j : j + size])
    if rng is not None:
        rng.shuffle(batches)
    for idx in batches:
        yield [examples[i] for i in idx]
