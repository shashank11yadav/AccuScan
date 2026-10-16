"""Compressed shards of preprocessed line images and their labels.

A shard is one `.npz` (no pickled objects) holding:
- pixels: uint8 [64, sum(widths)]   images laid side by side (see preprocess.to_uint8)
- widths: int32 [N]
- text:   uint8 [bytes]             UTF-8 labels concatenated
- text_offsets: int64 [N + 1]
- source: uint8 [N]                 index into SOURCES
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .preprocess import INPUT_HEIGHT

SOURCES = ("paper", "canvas", "gnhk-word", "gnhk-line")


@dataclass
class Sample:
    pixels: NDArray[np.uint8]  # [64, W]
    text: str
    source: str
    # The crop before preprocessing, when the producer has it (not stored in shards).
    raw: NDArray[np.uint8] | None = None


@dataclass
class ShardWriter:
    path: Path
    _pixels: list[NDArray[np.uint8]] = field(default_factory=list)
    _texts: list[str] = field(default_factory=list)
    _sources: list[int] = field(default_factory=list)

    def add(self, sample: Sample) -> None:
        if sample.pixels.ndim != 2 or sample.pixels.shape[0] != INPUT_HEIGHT:
            raise ValueError(f"expected [64, W] pixels, got {sample.pixels.shape}")
        self._pixels.append(sample.pixels)
        self._texts.append(sample.text)
        self._sources.append(SOURCES.index(sample.source))

    def __len__(self) -> int:
        return len(self._texts)

    def close(self) -> None:
        encoded = [t.encode("utf-8") for t in self._texts]
        offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(e) for e in encoded])
        empty = np.zeros((INPUT_HEIGHT, 0), np.uint8)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".part.npz")
        np.savez_compressed(
            tmp,
            pixels=np.concatenate(self._pixels, axis=1) if self._pixels else empty,
            widths=np.array([p.shape[1] for p in self._pixels], dtype=np.int32),
            text=np.frombuffer(b"".join(encoded), dtype=np.uint8),
            text_offsets=offsets,
            source=np.array(self._sources, dtype=np.uint8),
        )
        tmp.replace(self.path)


def read_shard(path: Path) -> Iterator[Sample]:
    with np.load(path, allow_pickle=False) as data:
        pixels = data["pixels"]
        widths = data["widths"]
        text = data["text"].tobytes()
        offsets = data["text_offsets"]
        sources = data["source"]
    starts = np.concatenate([[0], np.cumsum(widths)])
    for i in range(len(widths)):
        yield Sample(
            pixels=pixels[:, starts[i] : starts[i + 1]],
            text=text[offsets[i] : offsets[i + 1]].decode("utf-8"),
            source=SOURCES[int(sources[i])],
        )


def read_shards(directory: Path, pattern: str = "*.npz") -> list[Sample]:
    paths = [p for p in sorted(directory.glob(pattern)) if not p.name.endswith(".part.npz")]
    return [s for path in paths for s in read_shard(path)]
