"""Command line: `uv run accuscan <command>`. Every command reads configs/default.toml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

import typer

from .config import DEFAULT_CONFIG, Config, load_config
from .paths import FIXTURES_DIR, MODELS_DIR, REPORTS_DIR, RUNS_DIR, SAMPLES_DIR, default_data_dir

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

ConfigOpt = Annotated[Path, typer.Option("--config", help="TOML config file.")]
DataDirOpt = Annotated[
    Path | None,
    typer.Option("--data-dir", help="Datasets root (default: $DATA_DIR, else ../_datasets)."),
]


def _setup(config: Path, data_dir: Path | None) -> tuple[Config, Path, Path, Path]:
    cfg = load_config(config)
    fonts, gnhk, cache = cfg.resolve(data_dir or default_data_dir())
    return cfg, fonts, gnhk, cache


@app.command()
def fonts(config: ConfigOpt = DEFAULT_CONFIG, data_dir: DataDirOpt = None) -> None:
    """Download the pinned handwriting fonts (OFL) and verify every file."""
    from .fonts import download_fonts, load_font_set

    _, fonts_dir, _, _ = _setup(config, data_dir)
    font_set = load_font_set()
    download_fonts(font_set, fonts_dir)
    typer.echo(f"train fonts: {len(font_set.train)}, held-out fonts: {len(font_set.heldout)}")


@app.command()
def resources() -> None:
    """Rebuild the bundled text corpus and the dictionary-mode lexicon (needs network)."""
    from .resources import build_resources

    build_resources()


@app.command()
def synth(
    config: ConfigOpt = DEFAULT_CONFIG,
    data_dir: DataDirOpt = None,
    count: Annotated[int | None, typer.Option(help="Override the training line count.")] = None,
    workers: Annotated[int, typer.Option(help="Worker processes.")] = max(
        1, (os.cpu_count() or 2) - 2
    ),
    only: Annotated[
        Literal["train", "val", "canvas-test"] | None, typer.Option(help="Render one split.")
    ] = None,
) -> None:
    """Render synthetic lines: training, validation and the held-out-font canvas set."""
    from .fonts import available_fonts, load_font_set
    from .synth import SynthJob, run_synth
    from .train import DataPaths

    cfg, fonts_dir, _, cache = _setup(config, data_dir)
    font_set = load_font_set()
    train_fonts = tuple(available_fonts(font_set.train, fonts_dir))
    heldout_fonts = tuple(available_fonts(font_set.heldout, fonts_dir))
    paths = DataPaths.under(cache)
    s = cfg.synth
    jobs = {
        "train": SynthJob(paths.synth_train, train_fonts, "train", count or s.train_count,
                          s.shard_size, s.canvas_fraction, cfg.seed, s.heldout_text_fraction),
        "val": SynthJob(paths.synth_val, train_fonts, "train", s.val_count, s.shard_size,
                        s.canvas_fraction, cfg.seed + 1, s.heldout_text_fraction),
        "canvas-test": SynthJob(paths.canvas_test, heldout_fonts, "heldout", s.canvas_test_count,
                                s.shard_size, 1.0, cfg.seed + 2, s.heldout_text_fraction),
    }  # fmt: skip
    for name, job in jobs.items():
        if only is None or only == name:
            run_synth(job, workers)


@app.command()
def gnhk(config: ConfigOpt = DEFAULT_CONFIG, data_dir: DataDirOpt = None) -> None:
    """Crop GNHK words and lines (official train/test split plus a validation subset)."""
    from .gnhk import prepare_gnhk

    cfg, _, gnhk_dir, cache = _setup(config, data_dir)
    prepare_gnhk(gnhk_dir, cache / "gnhk", cfg.gnhk_val_fraction, cfg.seed)


@app.command("train")
def train_cmd(
    config: ConfigOpt = DEFAULT_CONFIG,
    data_dir: DataDirOpt = None,
    run: Annotated[str, typer.Option(help="Run name under pipeline/runs/.")] = "main",
    device: Annotated[str, typer.Option(help="auto, mps, cuda or cpu.")] = "auto",
    max_steps: Annotated[int | None, typer.Option(help="Stop after this many steps.")] = None,
    resume: Annotated[
        bool, typer.Option(help="Continue from runs/<run>/last.pt if it exists.")
    ] = False,
) -> None:
    """Train the CRNN; the best checkpoint is runs/<run>/best.pt."""
    from .train import train

    cfg, _, _, cache = _setup(config, data_dir)
    train(cfg, cache, RUNS_DIR / run, device, max_steps, resume=resume)


@app.command("evaluate")
def evaluate_cmd(
    config: ConfigOpt = DEFAULT_CONFIG,
    data_dir: DataDirOpt = None,
    run: Annotated[str, typer.Option(help="Run name under pipeline/runs/.")] = "main",
    device: Annotated[str, typer.Option(help="auto, mps, cuda or cpu.")] = "auto",
    limit: Annotated[int | None, typer.Option(help="Score at most N samples per set.")] = None,
    reports_dir: Annotated[Path, typer.Option(help="Where metrics.json goes.")] = REPORTS_DIR,
) -> None:
    """Score held-out sets, measure latency, apply the canvas gate -> reports/metrics.json."""
    from .evaluate import evaluate

    cfg, _, _, cache = _setup(config, data_dir)
    evaluate(cfg, cache, RUNS_DIR / run / "best.pt", reports_dir, device, limit)


@app.command()
def baseline(
    config: ConfigOpt = DEFAULT_CONFIG,
    data_dir: DataDirOpt = None,
    limit: Annotated[int | None, typer.Option(help="Score at most N crops in total.")] = None,
    reports_dir: Annotated[Path, typer.Option(help="Where baseline.json goes.")] = REPORTS_DIR,
) -> None:
    """TrOCR-small baseline on GNHK test (needs `uv sync --extra baseline`)."""
    from .baseline import run_baseline

    _, _, gnhk_dir, _ = _setup(config, data_dir)
    run_baseline(gnhk_dir, reports_dir, limit)


@app.command("export")
def export_cmd(
    config: ConfigOpt = DEFAULT_CONFIG,
    run: Annotated[str, typer.Option(help="Run name under pipeline/runs/.")] = "main",
    stub: Annotated[
        bool, typer.Option("--stub", help="Random weights, same I/O contract.")
    ] = False,
    out: Annotated[Path, typer.Option(help="Output directory.")] = MODELS_DIR,
) -> None:
    """Write model.onnx, model-meta.json and charset.json for the web app."""
    import json

    from .export import stub_model, write_model_files
    from .train import load_checkpoint

    cfg = load_config(config)
    if stub:
        write_model_files(cfg, stub_model(cfg), out, stub=True)
        return
    model, payload = load_checkpoint(RUNS_DIR / run / "best.pt")
    metrics_path = REPORTS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise typer.BadParameter(
            "run `accuscan evaluate` first so the model ships with its metrics"
        )
    metrics = json.loads(metrics_path.read_text())
    trained_on = {
        "data": ["GNHK train (CC BY 4.0)", "synthetic lines in 54 OFL handwriting fonts"],
        "epoch": payload.get("epoch"),
        "seed": payload.get("seed"),
    }
    write_model_files(cfg, model, out, stub=False, metrics=metrics, trained_on=trained_on)


@app.command("export-fixtures")
def export_fixtures() -> None:
    """Write the preprocessing and decoding parity fixtures to shared/fixtures/."""
    from .fixtures import write_fixtures

    for path in write_fixtures(FIXTURES_DIR):
        typer.echo(path)


@app.command("export-samples")
def export_samples(config: ConfigOpt = DEFAULT_CONFIG, data_dir: DataDirOpt = None) -> None:
    """Render the demo's sample gallery in held-out fonts."""
    from .fonts import available_fonts, load_font_set
    from .samples import write_samples

    cfg, fonts_dir, _, _ = _setup(config, data_dir)
    held = load_font_set().heldout
    paths = available_fonts(held, fonts_dir)
    write_samples(
        [(s.family_dir, p) for s, p in zip(held, paths, strict=True)], SAMPLES_DIR, cfg.seed
    )


@app.command("smoke")
def smoke_cmd(config: ConfigOpt = DEFAULT_CONFIG, data_dir: DataDirOpt = None) -> None:
    """Tiny end-to-end run (synth, GNHK fixture, train, evaluate, export) in a temp dir."""
    from .smoke import smoke

    cfg, fonts_dir, _, _ = _setup(config, data_dir)
    smoke(cfg, fonts_dir)


def main() -> None:
    app()
