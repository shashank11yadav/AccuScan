import json
from pathlib import Path

import numpy as np
import pytest
import torch

from accuscan.config import load_config
from accuscan.dataset import (
    Example,
    bucket_width,
    bucketed_batches,
    collate,
    required_frames,
    to_examples,
)
from accuscan.export import INPUT_NAME, OUTPUT_NAME, check_onnx, stub_model, write_model_files
from accuscan.model import CRNN, count_parameters
from accuscan.paths import MODELS_DIR
from accuscan.shards import Sample
from accuscan.train import lr_schedule


def test_parameter_count_is_within_the_4_to_8_million_budget() -> None:
    n = count_parameters(CRNN(load_config().model))
    assert 4_000_000 <= n <= 8_000_000


@pytest.mark.parametrize("width", [4, 64, 260, 1024])
def test_output_is_time_major_log_softmax(width: int) -> None:
    model = CRNN(load_config().model).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 1, 64, width))
    assert out.shape == (width // 4, 2, 96)
    assert torch.allclose(out.exp().sum(-1), torch.ones(width // 4, 2), atol=1e-5)


def test_required_frames_counts_blanks_between_repeats() -> None:
    assert required_frames([1, 2, 3]) == 3
    assert required_frames([5, 5, 5]) == 5


def test_to_examples_drops_images_too_narrow_for_their_label() -> None:
    narrow = Sample(np.zeros((64, 8), np.uint8), "hello", "paper")  # 2 frames, needs 6
    wide = Sample(np.zeros((64, 64), np.uint8), "hello", "paper")
    assert [e.text for e in to_examples([narrow, wide])] == ["hello"]


def _examples(widths: list[int]) -> list[Example]:
    return [Example(np.zeros((64, w), np.uint8), (1,), "a", "paper") for w in widths]


def test_batches_share_a_width_bucket_and_respect_the_column_cap() -> None:
    examples = _examples([60, 64, 100, 128, 900, 1000, 1024, 200] * 10)
    batches = list(bucketed_batches(examples, 8, np.random.default_rng(0), max_columns=2048))
    for batch in batches:
        buckets = {bucket_width(e.pixels.shape[1]) for e in batch}
        assert len(buckets) == 1
        assert len(batch) * buckets.pop() <= 2048
    assert sum(len(b) for b in batches) == len(examples)


def test_drop_last_keeps_batch_shapes_fixed() -> None:
    examples = _examples([64] * 20)
    batches = list(bucketed_batches(examples, 8, np.random.default_rng(0), drop_last=True))
    assert [len(b) for b in batches] == [8, 8]


def test_collate_pads_with_paper_and_reports_true_lengths() -> None:
    examples = _examples([64, 120])
    batch = collate(examples)
    assert batch.images.shape == (2, 1, 64, 128)
    assert float(batch.images.min()) == -1.0
    assert batch.input_lengths.tolist() == [16, 30]


def test_lr_schedule_warms_up_then_decays_and_never_errors_past_the_end() -> None:
    f = lr_schedule(100)
    assert f(0) < f(4) <= 1.0
    assert f(99) < f(50) < f(10)
    assert f(500) == pytest.approx(f(100))


def test_stub_export_matches_the_io_contract(tmp_path: Path) -> None:
    cfg = load_config()
    write_model_files(cfg, stub_model(cfg), tmp_path, stub=True, log=lambda _: None)
    onnx_path = tmp_path / "model.onnx"
    assert onnx_path.stat().st_size < 10 * 1024 * 1024
    check_onnx(onnx_path)
    meta = json.loads((tmp_path / "model-meta.json").read_text())
    assert meta["stub"] is True
    assert meta["metrics"] is None
    assert meta["input"]["name"] == INPUT_NAME
    assert meta["output"]["name"] == OUTPUT_NAME
    for key in ("version", "featureVersion", "fps", "featureDim", "classes", "trainedOn"):
        assert key in meta
    charset = json.loads((tmp_path / "charset.json").read_text())
    assert len(charset["symbols"]) == meta["classes"] == 96


def test_committed_web_model_has_the_same_contract() -> None:
    meta = json.loads((MODELS_DIR / "model-meta.json").read_text())
    assert meta["classes"] == 96
    assert meta["input"] == {"name": "image", "height": 64, "maxWidth": 1024, "widthMultiple": 4}
    check_onnx(MODELS_DIR / "model.onnx")
