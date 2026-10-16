"""Baseline: microsoft/trocr-small-handwritten (MIT) on GNHK test, CPU only.

Needs the optional dependencies: `uv sync --extra baseline`. TrOCR gets the
same perspective-rectified word and line crops the CRNN is scored on, before
the CRNN's own 64 px preprocessing, with the same filtering of special tokens,
printed text and out-of-charset words.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .gnhk import PrepStats, iter_pages, page_samples
from .metrics import ErrorRates
from .paths import REPORTS_DIR
from .report import write_report

MODEL_ID = "microsoft/trocr-small-handwritten"
# Pinned commit of the Hugging Face repository. Its weights are a PyTorch
# checkpoint, which transformers loads with torch's weights_only unpickler.
MODEL_REVISION = "b4648cfa171985a6745f37ddd637e98c0da958ac"
SETS = {"gnhk-word": "gnhk_test_words", "gnhk-line": "gnhk_test_lines"}


def run_baseline(
    gnhk_root: Path,
    reports_dir: Path = REPORTS_DIR,
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Score TrOCR on GNHK test (at most `limit` crops in total); writes reports/baseline.json."""
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    torch.set_num_threads(1)
    processor = TrOCRProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).eval()
    n_params = sum(p.numel() for p in model.parameters())
    rates = {key: ErrorRates() for key in SETS.values()}
    times: dict[str, list[float]] = {key: [] for key in SETS.values()}
    stats = PrepStats()
    for page in iter_pages(gnhk_root, "test"):
        for sample in page_samples(page, stats):
            key = SETS[sample.source]
            if limit is not None and sum(r.samples for r in rates.values()) >= limit:
                break
            if sample.raw is None:
                continue
            image = Image.fromarray(sample.raw).convert("RGB")
            started = time.perf_counter()
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            with torch.no_grad():
                ids = model.generate(pixel_values, max_new_tokens=64)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            times[key].append((time.perf_counter() - started) * 1000)
            rates[key].add(text, sample.text)
        if limit is not None and sum(r.samples for r in rates.values()) >= limit:
            break
    results = {
        key: {**r.as_dict(), "ms_per_crop": round(float(np.median(times[key])), 1)}
        for key, r in rates.items()
        if r.samples
    }
    for key, value in results.items():
        log(f"{key}: {value}")
    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "parameters": n_params,
        # float32 weights, which is what a browser would have to download
        "model_mb": round(n_params * 4 / 1e6, 1),
        "limit": limit,
        "sets": results,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "baseline.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_report(reports_dir)
    return report
