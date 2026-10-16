"""Held-out evaluation: GNHK test, the held-out-font canvas set, latency and the gate."""

from __future__ import annotations

import json
import platform
import statistics
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from .config import Config
from .dataset import Example, to_examples
from .decode import Lexicon, beam_search, greedy_decode
from .export import INPUT_NAME, export_onnx
from .metrics import ErrorRates
from .model import CRNN
from .paths import LEXICON_PATH, REPORTS_DIR
from .preprocess import INPUT_HEIGHT
from .report import write_report
from .resources import read_lexicon
from .shards import read_shard, read_shards
from .train import DataPaths, load_checkpoint, pick_device, predict


def score(
    log_probs: Sequence[np.ndarray],
    examples: Sequence[Example],
    lexicon: Lexicon | None,
    cfg: Config,
) -> dict[str, Any]:
    greedy = ErrorRates()
    dictionary = ErrorRates()
    fallbacks = 0
    for lp, example in zip(log_probs, examples, strict=True):
        greedy.add(greedy_decode(lp).text, example.text)
        if lexicon is not None:
            out = beam_search(
                lp, lexicon, beam_width=cfg.evaluate.beam_width, lm_weight=cfg.evaluate.lm_weight
            )
            fallbacks += out.fallback
            dictionary.add(out.text, example.text)
    result: dict[str, Any] = {"greedy": greedy.as_dict()}
    if lexicon is not None:
        result["dictionary"] = {
            **dictionary.as_dict(),
            "fallback_rate": round(fallbacks / max(1, len(examples)), 4),
        }
    return result


def measure_latency(model: CRNN, cfg: Config, repeats: int = 20) -> dict[str, float]:
    """Median ms per line for the exported (quantised) model, one CPU thread."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        export_onnx(model, path, cfg.export.opset, cfg.export.quantize, cfg.export.float_convs)
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        size_mb = path.stat().st_size / 1e6
        result: dict[str, float] = {"model_mb": round(size_mb, 2)}
        for width in cfg.evaluate.latency_widths:
            x = np.zeros((1, 1, INPUT_HEIGHT, width), dtype=np.float32)
            session.run(None, {INPUT_NAME: x})
            times = []
            for _ in range(repeats):
                started = time.perf_counter()
                session.run(None, {INPUT_NAME: x})
                times.append((time.perf_counter() - started) * 1000)
            result[f"ms_width_{width}"] = round(statistics.median(times), 2)
    return result


def _gnhk_test(paths: DataPaths, kind: str) -> list[Example]:
    path = paths.gnhk / f"test-{kind}.npz"
    return to_examples(list(read_shard(path))) if path.exists() else []


def evaluate(
    cfg: Config,
    cache: Path,
    checkpoint: Path,
    reports_dir: Path = REPORTS_DIR,
    device_name: str = "auto",
    limit: int | None = None,
    lexicon_path: Path = LEXICON_PATH,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    device = pick_device(device_name)
    model, payload = load_checkpoint(checkpoint, device)
    model = model.to(device)
    paths = DataPaths.under(cache)
    lexicon = Lexicon.from_counts(read_lexicon(lexicon_path))
    sets: dict[str, list[Example]] = {
        "gnhk_test_words": _gnhk_test(paths, "word"),
        "gnhk_test_lines": _gnhk_test(paths, "line"),
        "canvas_heldout_fonts": to_examples(read_shards(paths.canvas_test)),
    }
    if not sets["canvas_heldout_fonts"]:
        raise FileNotFoundError(
            f"no canvas test set under {paths.canvas_test}; run `accuscan synth`"
        )
    results: dict[str, Any] = {}
    for name, examples in sets.items():
        if limit is not None:
            examples = examples[:limit]
        if not examples:
            log(f"{name}: missing, skipped")
            continue
        started = time.time()
        log_probs = predict(model, examples, device)
        use_lexicon = name != "gnhk_test_lines"  # beam search on long lines is slow in Python
        results[name] = score(log_probs, examples, lexicon if use_lexicon else None, cfg)
        log(f"{name}: {json.dumps(results[name])} ({time.time() - started:.0f}s)")

    canvas_cer = results["canvas_heldout_fonts"]["greedy"]["cer"]
    gate = {
        "canvas_cer": canvas_cer,
        "threshold": cfg.evaluate.gate_cer,
        "defaultMode": "free" if canvas_cer <= cfg.evaluate.gate_cer else "dictionary",
    }
    metrics: dict[str, Any] = {
        "checkpoint": {
            "epoch": payload.get("epoch"),
            "val_set": payload.get("val_set"),
            "val_cer": payload.get("val_cer"),
        },
        "sets": results,
        "latency_cpu": measure_latency(model.cpu(), cfg),
        "gate": gate,
        "environment": {
            "machine": platform.machine(),
            "system": platform.system(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    write_report(reports_dir)
    log(f"gate: {gate}")
    return metrics
