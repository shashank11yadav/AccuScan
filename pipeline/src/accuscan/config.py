"""Typed view of configs/default.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .model import ModelConfig
from .paths import CONFIG_DIR

DEFAULT_CONFIG = CONFIG_DIR / "default.toml"


@dataclass(frozen=True)
class SynthConfig:
    train_count: int
    val_count: int
    canvas_test_count: int
    shard_size: int
    canvas_fraction: float
    heldout_text_fraction: float


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    max_batch_columns: int
    lr: float
    weight_decay: float
    grad_clip: float
    synth_per_epoch: int
    gnhk_upsample: int
    patience: int
    num_workers: int
    curriculum: tuple[tuple[int, int], ...]

    def max_label_length(self, epoch: int) -> int:
        limit = self.curriculum[0][1]
        for start, length in self.curriculum:
            if epoch >= start:
                limit = length
        return limit


@dataclass(frozen=True)
class EvalConfig:
    gate_cer: float
    beam_width: int
    lm_weight: float
    latency_widths: tuple[int, ...]


@dataclass(frozen=True)
class ExportConfig:
    opset: int
    quantize: bool
    float_convs: int


@dataclass(frozen=True)
class Config:
    seed: int
    fonts_dir: str
    gnhk_dir: str
    cache_dir: str
    gnhk_val_fraction: float
    synth: SynthConfig
    model: ModelConfig
    train: TrainConfig
    evaluate: EvalConfig
    export: ExportConfig

    def resolve(self, data_dir: Path) -> tuple[Path, Path, Path]:
        """(fonts, gnhk, cache) directories under data_dir."""
        return data_dir / self.fonts_dir, data_dir / self.gnhk_dir, data_dir / self.cache_dir


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    model = raw["model"]
    train = dict(raw["train"])
    train["curriculum"] = tuple((int(a), int(b)) for a, b in train["curriculum"])
    evaluate = dict(raw["evaluate"])
    evaluate["latency_widths"] = tuple(evaluate["latency_widths"])
    return Config(
        seed=int(raw["seed"]),
        fonts_dir=raw["paths"]["fonts"],
        gnhk_dir=raw["paths"]["gnhk"],
        cache_dir=raw["paths"]["cache"],
        gnhk_val_fraction=float(raw["gnhk"]["val_fraction"]),
        synth=SynthConfig(**raw["synth"]),
        model=ModelConfig(
            channels=tuple(model["channels"]),
            hidden=model["hidden"],
            layers=model["layers"],
            dropout=model["dropout"],
        ),
        train=TrainConfig(**train),
        evaluate=EvalConfig(**evaluate),
        export=ExportConfig(**raw["export"]),
    )
