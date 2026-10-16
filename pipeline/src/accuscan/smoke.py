"""End-to-end smoke run on tiny data: synth, GNHK fixture, train, evaluate, export.

Uses the downloaded fonts when they are present and Pillow's built-in font
otherwise, so it also runs in CI without network access to the font files.
Everything is written to a temporary directory.
"""

from __future__ import annotations

import dataclasses
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from .config import Config
from .evaluate import evaluate
from .export import check_onnx, write_model_files
from .fonts import FontSpec, load_font_set, verify_font
from .gnhk import prepare_gnhk
from .gnhk_fixture import write_fixture
from .render import FontRef
from .synth import SynthJob, run_synth
from .train import DataPaths, load_checkpoint, train


def _fonts(root: Path, specs: tuple[FontSpec, ...], limit: int) -> tuple[FontRef, ...]:
    found: list[FontRef] = []
    for spec in specs[:limit]:
        try:
            verify_font(spec, root)
        except (FileNotFoundError, ValueError):
            return (None,)
        found.append(spec.font_path(root))
    return tuple(found) or (None,)


def smoke(cfg: Config, fonts_root: Path, log: Callable[[str], None] = print) -> None:
    started = time.time()
    font_set = load_font_set()
    train_fonts = _fonts(fonts_root, font_set.train, 4)
    heldout_fonts = _fonts(fonts_root, font_set.heldout, 2)
    log(f"fonts: {'downloaded set' if train_fonts[0] is not None else 'built-in fallback'}")
    small = dataclasses.replace(
        cfg,
        train=dataclasses.replace(
            cfg.train, epochs=1, batch_size=16, synth_per_epoch=64, gnhk_upsample=1, patience=1
        ),
    )
    with tempfile.TemporaryDirectory(prefix="accuscan-smoke-") as tmp:
        root = Path(tmp)
        paths = DataPaths.under(root / "cache")
        for out, fonts, part, count, canvas in (
            (paths.synth_train, train_fonts, "train", 64, cfg.synth.canvas_fraction),
            (paths.synth_val, train_fonts, "train", 16, cfg.synth.canvas_fraction),
            (paths.canvas_test, heldout_fonts, "heldout", 16, 1.0),
        ):
            job = SynthJob(out, fonts, part, count, 32, canvas, cfg.seed)  # type: ignore[arg-type]
            run_synth(job, workers=1, log=log)
        prepare_gnhk(write_fixture(root / "gnhk"), paths.gnhk, cfg.gnhk_val_fraction, cfg.seed, log)
        best = train(small, root / "cache", root / "run", device_name="cpu", max_steps=3, log=log)
        evaluate(
            small,
            root / "cache",
            best,
            reports_dir=root / "reports",
            device_name="cpu",
            limit=16,
            log=log,
        )
        model, _ = load_checkpoint(best)
        onnx_path = write_model_files(small, model, root / "models", stub=False, log=log)
        check_onnx(onnx_path)
    log(f"smoke run passed in {time.time() - started:.0f}s")
