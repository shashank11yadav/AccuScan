"""ONNX export (int8 dynamic quantisation) and the browser's model files.

I/O contract, identical for the stub and the trained model:
    input  image     float32 [1, 1, 64, W]   W dynamic, a multiple of 4, <= 1024
    output log_probs float32 [W/4, 1, 96]    log-softmax over blank + printable ASCII

Files written to web/public/models/: model.onnx, model-meta.json, charset.json.
"""

from __future__ import annotations

import json
import logging
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from .charset import BLANK, NUM_CLASSES, write_charset
from .config import Config
from .decode import DEFAULT_MAX_DROP_PER_WORD, DEFAULT_PRUNE
from .model import CRNN, DOWNSAMPLE, count_parameters
from .preprocess import FEATURE_VERSION, INPUT_HEIGHT, MAX_WIDTH, WIDTH_MULTIPLE

INPUT_NAME = "image"
OUTPUT_NAME = "log_probs"
MODEL_VERSION = "2.0.0"


def export_onnx(model: CRNN, path: Path, opset: int, quantize: bool, float_convs: int) -> None:
    model = model.cpu().eval()
    example = torch.zeros(1, 1, INPUT_HEIGHT, 128)
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        # The TorchScript exporter is deprecated but still the one that emits a
        # standard LSTM op; its warnings (and the quantiser's hints) are noise here.
        warnings.simplefilter("ignore")
        logging.getLogger().setLevel(logging.ERROR)
        fp32 = Path(tmp) / "model-fp32.onnx"
        torch.onnx.export(
            model,
            (example,),
            str(fp32),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {3: "width"}, OUTPUT_NAME: {0: "frames"}},
            opset_version=opset,
            dynamo=False,  # the TorchScript exporter emits a plain ONNX LSTM
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not quantize:
            path.write_bytes(fp32.read_bytes())
            return
        graph = onnx.load(str(fp32)).graph
        convs = [n.name for n in graph.node if n.op_type == "Conv"]
        quantize_dynamic(
            str(fp32),
            str(path),
            weight_type=QuantType.QInt8,
            nodes_to_exclude=convs[:float_convs],
        )


def check_onnx(path: Path, model: CRNN | None = None, tolerance: float = 0.05) -> dict[str, Any]:
    """Run the exported file at several widths; compare with PyTorch when given."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (inp,) = session.get_inputs()
    (out,) = session.get_outputs()
    if (inp.name, out.name) != (INPUT_NAME, OUTPUT_NAME):
        raise ValueError(f"unexpected I/O names {inp.name} -> {out.name}")
    rng = np.random.default_rng(0)
    worst = 0.0
    for width in (4, 64, 256, MAX_WIDTH):
        x = rng.uniform(-1, 1, size=(1, 1, INPUT_HEIGHT, width)).astype(np.float32)
        (y,) = session.run(None, {INPUT_NAME: x})
        if y.shape != (width // DOWNSAMPLE, 1, NUM_CLASSES):
            raise ValueError(f"width {width}: got output shape {y.shape}")
        if not np.allclose(np.exp(y).sum(-1), 1.0, atol=1e-3):
            raise ValueError("output is not a log-softmax")
        if model is not None:
            with torch.no_grad():
                ref = model.cpu().eval()(torch.from_numpy(x)).numpy()
            worst = max(worst, float(np.abs(np.exp(y) - np.exp(ref)).max()))
    if worst > tolerance:
        raise ValueError(f"ONNX output differs from PyTorch by {worst:.4f} (probability)")
    return {"max_prob_diff": round(worst, 5)}


def model_meta(
    cfg: Config,
    model: CRNN,
    onnx_bytes: int,
    stub: bool,
    metrics: dict[str, Any] | None,
    trained_on: dict[str, Any] | None,
) -> dict[str, Any]:
    default_mode = "free"
    if metrics is not None:
        default_mode = str(metrics.get("gate", {}).get("defaultMode", "free"))
    return {
        "version": MODEL_VERSION + ("-stub" if stub else ""),
        "stub": stub,
        "featureVersion": FEATURE_VERSION,
        "fps": None,
        "featureDim": INPUT_HEIGHT,
        "classes": NUM_CLASSES,
        "blank": BLANK,
        "input": {"name": INPUT_NAME, "height": INPUT_HEIGHT, "maxWidth": MAX_WIDTH,
                  "widthMultiple": WIDTH_MULTIPLE},
        "output": {"name": OUTPUT_NAME, "downsample": DOWNSAMPLE},
        "parameters": count_parameters(model),
        "bytes": onnx_bytes,
        "quantization": "int8-dynamic" if cfg.export.quantize else "none",
        "decoder": {
            "beamWidth": cfg.evaluate.beam_width,
            "lmWeight": cfg.evaluate.lm_weight,
            "prune": DEFAULT_PRUNE,
            "maxDropPerWord": DEFAULT_MAX_DROP_PER_WORD,
        },
        "defaultMode": default_mode,
        "trainedOn": trained_on,
        "metrics": metrics,
    }  # fmt: skip


def write_model_files(
    cfg: Config,
    model: CRNN,
    out_dir: Path,
    stub: bool,
    metrics: dict[str, Any] | None = None,
    trained_on: dict[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    export_onnx(model, onnx_path, cfg.export.opset, cfg.export.quantize, cfg.export.float_convs)
    check = check_onnx(onnx_path, model)
    meta = model_meta(cfg, model, onnx_path.stat().st_size, stub, metrics, trained_on)
    (out_dir / "model-meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    write_charset(out_dir / "charset.json")
    log(
        f"{onnx_path}: {onnx_path.stat().st_size / 1e6:.2f} MB, "
        f"{meta['parameters']:,} parameters, quantisation check {check}"
    )
    return onnx_path


def stub_model(cfg: Config) -> CRNN:
    """Random weights with a fixed seed: same I/O contract, no training."""
    torch.manual_seed(cfg.seed)
    return CRNN(cfg.model).eval()
