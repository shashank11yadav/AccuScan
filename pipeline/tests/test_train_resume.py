import dataclasses
import json
from pathlib import Path

import pytest
import torch

from accuscan.config import Config, load_config
from accuscan.gnhk import prepare_gnhk
from accuscan.gnhk_fixture import write_fixture
from accuscan.synth import SynthJob, run_synth
from accuscan.train import DataPaths, train


class SimulatedCrashError(Exception):
    pass


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[Config, Path]:
    cfg = load_config()
    cfg = dataclasses.replace(
        cfg,
        model=dataclasses.replace(cfg.model, channels=(8, 8, 16, 16, 16), hidden=16),
        train=dataclasses.replace(
            cfg.train, epochs=2, batch_size=8, synth_per_epoch=24, gnhk_upsample=1, patience=10
        ),
    )
    root = tmp_path_factory.mktemp("resume")
    paths = DataPaths.under(root / "cache")
    for out, count in ((paths.synth_train, 24), (paths.synth_val, 8)):
        run_synth(
            SynthJob(out, (None,), "train", count, 32, 0.5, cfg.seed), workers=1, log=lambda _: None
        )
    prepare_gnhk(
        write_fixture(root / "gnhk"), paths.gnhk, cfg.gnhk_val_fraction, cfg.seed, lambda _: None
    )
    return cfg, root / "cache"


def _records(run: Path) -> list[dict[str, float]]:
    lines = (run / "log.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_resumed_run_matches_an_uninterrupted_one(
    tiny: tuple[Config, Path], tmp_path: Path
) -> None:
    cfg, cache = tiny
    train(cfg, cache, tmp_path / "full", device_name="cpu", log=lambda _: None)

    def crash_after_first_epoch(message: str) -> None:
        if message.startswith('{"epoch": 0'):
            raise SimulatedCrashError

    with pytest.raises(SimulatedCrashError):
        train(cfg, cache, tmp_path / "resumed", device_name="cpu", log=crash_after_first_epoch)
    messages: list[str] = []
    train(cfg, cache, tmp_path / "resumed", device_name="cpu", log=messages.append, resume=True)

    assert any(m.startswith("resumed after epoch 0") for m in messages)
    full, resumed = _records(tmp_path / "full"), _records(tmp_path / "resumed")
    assert [r["epoch"] for r in resumed] == [0, 1]
    assert [r["loss"] for r in resumed] == [r["loss"] for r in full]
    a = torch.load(tmp_path / "full" / "last.pt", weights_only=True)["model"]
    b = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)["model"]
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), name


def test_resume_without_a_checkpoint_starts_fresh(
    tiny: tuple[Config, Path], tmp_path: Path
) -> None:
    cfg, cache = tiny
    one_epoch = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, epochs=1))
    messages: list[str] = []
    train(one_epoch, cache, tmp_path / "run", device_name="cpu", log=messages.append, resume=True)
    assert any("no checkpoint to resume" in m for m in messages)
    assert [r["epoch"] for r in _records(tmp_path / "run")] == [0]


def test_a_fresh_run_replaces_an_old_log(tiny: tuple[Config, Path], tmp_path: Path) -> None:
    cfg, cache = tiny
    one_epoch = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, epochs=1))
    for _ in range(2):
        train(one_epoch, cache, tmp_path / "run", device_name="cpu", log=lambda _: None)
    assert [r["epoch"] for r in _records(tmp_path / "run")] == [0]
