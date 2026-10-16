"""Generate sharded synthetic line images (paper and canvas styles)."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .preprocess import EmptyImageError, preprocess, to_uint8
from .render import FontRef, RenderConfig, Style, render_line, supported_chars
from .resources import corpus_texts
from .shards import Sample, ShardWriter
from .text_source import TextSource, sample_text

TextPart = Literal["train", "heldout"]


@dataclass(frozen=True)
class SynthJob:
    out_dir: Path
    fonts: tuple[FontRef, ...]
    text_part: TextPart
    count: int
    shard_size: int
    canvas_fraction: float
    seed: int
    heldout_text_fraction: float = 0.1

    def shard_counts(self) -> list[int]:
        full, rest = divmod(self.count, self.shard_size)
        return [self.shard_size] * full + ([rest] if rest else [])


_SOURCES: dict[tuple[TextPart, float], TextSource] = {}


def _text_source(part: TextPart, fraction: float) -> TextSource:
    """Parsed corpus, cached per process (each pool worker parses it once)."""
    key = (part, fraction)
    if key not in _SOURCES:
        _SOURCES[key] = TextSource.from_texts(list(corpus_texts().values()), part, fraction)
    return _SOURCES[key]


def render_sample(
    source: TextSource,
    fonts: Sequence[FontRef],
    canvas_fraction: float,
    rng: np.random.Generator,
    cfg: RenderConfig | None = None,
) -> Sample:
    """One preprocessed sample; retries text or font until both are usable."""
    for _ in range(20):
        font = fonts[int(rng.integers(0, len(fonts)))]
        text = sample_text(source, rng)
        if not set(text) <= supported_chars(font):
            continue
        style: Style = "canvas" if rng.random() < canvas_fraction else "paper"
        try:
            pixels = to_uint8(preprocess(render_line(text, font, style, rng, cfg)))
        except EmptyImageError:
            continue
        return Sample(pixels, text, style)
    raise RuntimeError("could not render a usable sample in 20 attempts")


def _write_shard(args: tuple[SynthJob, int, int]) -> int:
    job, index, n = args
    path = job.out_dir / f"shard-{index:05d}.npz"
    if path.exists():
        return 0
    rng = np.random.default_rng([job.seed, index])
    source = _text_source(job.text_part, job.heldout_text_fraction)
    writer = ShardWriter(path)
    for _ in range(n):
        writer.add(render_sample(source, job.fonts, job.canvas_fraction, rng))
    writer.close()
    return n


def _init_worker() -> None:
    # One thread per process: OpenCV and torch thread pools in every worker
    # otherwise oversubscribe the CPU and slow rendering down by orders of magnitude.
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def run_synth(job: SynthJob, workers: int, log: Callable[[str], None] = print) -> int:
    """Render all shards (skipping existing ones). Output depends only on the seed."""
    job.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(job, i, n) for i, n in enumerate(job.shard_counts())]
    written = 0
    if workers <= 1:
        for task in tqdm(tasks, desc=job.out_dir.name, unit="shard"):
            written += _write_shard(task)
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
            for n in tqdm(
                pool.imap_unordered(_write_shard, tasks), total=len(tasks), desc=job.out_dir.name
            ):
                written += n
    log(f"{job.out_dir}: {written} new samples ({job.count} requested)")
    return written
