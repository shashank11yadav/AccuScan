"""Train the CRNN with CTC on synthetic lines plus GNHK (upsampled)."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import Config
from .dataset import Example, bucketed_batches, collate, to_examples
from .decode import greedy_decode
from .metrics import ErrorRates
from .model import CRNN, ModelConfig, count_parameters
from .preprocess import FEATURE_VERSION
from .shards import read_shard, read_shards


@dataclass(frozen=True)
class DataPaths:
    synth_train: Path
    synth_val: Path
    canvas_test: Path
    gnhk: Path

    @classmethod
    def under(cls, cache: Path) -> DataPaths:
        return cls(
            cache / "synth" / "train",
            cache / "synth" / "val",
            cache / "synth" / "canvas-test",
            cache / "gnhk",
        )

    def gnhk_split(self, part: str) -> list[Example]:
        samples = []
        for kind in ("word", "line"):
            path = self.gnhk / f"{part}-{kind}.npz"
            if path.exists():
                samples += list(read_shard(path))
        return to_examples(samples)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - also seeds any library using the global state


@torch.no_grad()
def predict(
    model: nn.Module, examples: Sequence[Example], device: torch.device, batch_size: int = 64
) -> list[np.ndarray]:
    """Per-example log-probs [T, C] trimmed to the example's own width."""
    model.eval()
    outputs: list[np.ndarray] = [np.empty(0)] * len(examples)
    index = {id(e): i for i, e in enumerate(examples)}
    for chunk in bucketed_batches(examples, batch_size, rng=None, max_columns=16384):
        batch = collate(chunk)
        log_probs = model(batch.images.to(device)).float().cpu().numpy()  # [T, B, C]
        for j, example in enumerate(chunk):
            outputs[index[id(example)]] = log_probs[: int(batch.input_lengths[j]), j, :]
    return outputs


def greedy_cer(model: nn.Module, examples: Sequence[Example], device: torch.device) -> ErrorRates:
    rates = ErrorRates()
    for lp, example in zip(predict(model, examples, device), examples, strict=True):
        rates.add(greedy_decode(lp).text, example.text)
    return rates


def _epoch_examples(
    synth: Sequence[Example],
    gnhk: Sequence[Example],
    cfg: Config,
    epoch: int,
    rng: np.random.Generator,
) -> list[Example]:
    limit = cfg.train.max_label_length(epoch)
    pool = [e for e in synth if len(e.labels) <= limit]
    take = min(len(pool), cfg.train.synth_per_epoch)
    chosen = [pool[int(i)] for i in rng.choice(len(pool), size=take, replace=False)] if take else []
    real = [e for e in gnhk if len(e.labels) <= limit] * cfg.train.gnhk_upsample
    return chosen + real


def lr_schedule(
    total_steps: int, warmup: float = 0.05, floor: float = 0.02
) -> Callable[[int], float]:
    """Linear warm-up, then cosine decay to `floor`; constant after total_steps."""
    warm = max(1, int(total_steps * warmup))

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, total_steps - warm))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def build_model(model_cfg: ModelConfig) -> CRNN:
    return CRNN(model_cfg)


def save_checkpoint(
    path: Path, model: CRNN, model_cfg: ModelConfig, extra: dict[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "model_config": asdict(model_cfg),
        "feature_version": FEATURE_VERSION,
        **extra,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[CRNN, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    raw = payload["model_config"]
    model_cfg = ModelConfig(
        channels=tuple(raw["channels"]),
        hidden=raw["hidden"],
        layers=raw["layers"],
        dropout=raw["dropout"],
        num_classes=raw["num_classes"],
    )
    model = CRNN(model_cfg)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def train(
    cfg: Config,
    cache: Path,
    run_dir: Path,
    device_name: str = "auto",
    max_steps: int | None = None,
    log: Callable[[str], None] = print,
    resume: bool = False,
) -> Path:
    """Train and return the path of the best checkpoint.

    Every epoch also writes runs/<run>/last.pt with the optimiser, scheduler
    and random-number state, so `resume=True` continues an interrupted run
    exactly where it stopped.
    """
    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = pick_device(device_name)
    paths = DataPaths.under(cache)
    synth = to_examples(read_shards(paths.synth_train))
    gnhk_train = paths.gnhk_split("train")
    gnhk_val = paths.gnhk_split("val")
    val = gnhk_val or to_examples(read_shards(paths.synth_val))
    val_name = "gnhk-val" if gnhk_val else "synth-val"
    if not synth and not gnhk_train:
        raise FileNotFoundError(f"no training data under {cache}; run `accuscan synth` first")
    log(f"device={device} synth={len(synth)} gnhk-train={len(gnhk_train)} {val_name}={len(val)}")

    model = build_model(cfg.model).to(device)
    log(f"parameters: {count_parameters(model):,}")
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    per_epoch = (
        min(cfg.train.synth_per_epoch, len(synth)) + len(gnhk_train) * cfg.train.gnhk_upsample
    )
    total_steps = (
        max_steps or max(1, math.ceil(per_epoch / cfg.train.batch_size)) * cfg.train.epochs
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_schedule(total_steps))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    log_path = run_dir / "log.jsonl"
    best_cer = math.inf
    stale = 0
    step = 0
    start_epoch = 0
    if resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        start_epoch, step = state["epoch"] + 1, state["step"]
        best_cer, stale = state["best_cer"], state["stale"]
        log(f"resumed after epoch {state['epoch']} (step {step}, best CER {best_cer:.4f})")
    else:
        if resume:
            log(f"no checkpoint to resume in {run_dir}; starting fresh")
        log_path.write_text("", encoding="utf-8")
    for epoch in range(start_epoch, cfg.train.epochs):
        if stale >= cfg.train.patience:
            break
        model.train()
        started = time.time()
        losses = []
        examples = _epoch_examples(synth, gnhk_train, cfg, epoch, rng)
        for chunk in bucketed_batches(
            examples, cfg.train.batch_size, rng, cfg.train.max_batch_columns, drop_last=True
        ):
            batch = collate(chunk, rng, noise=0.1)
            log_probs = model(batch.images.to(device))
            # CTC is not implemented on MPS; the loss runs on CPU and autograd
            # carries the gradient back to the device.
            loss = ctc(
                log_probs.float().cpu(), batch.targets, batch.input_lengths, batch.target_lengths
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimiser.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        rates = greedy_cer(model, val, device)
        record = {
            "epoch": epoch,
            "step": step,
            "loss": round(float(np.mean(losses)) if losses else math.nan, 4),
            f"{val_name}_cer": round(rates.cer, 4),
            f"{val_name}_wer": round(rates.wer, 4),
            "seconds": round(time.time() - started, 1),
        }
        if rates.cer < best_cer:
            best_cer, stale = rates.cer, 0
            save_checkpoint(
                best_path,
                model,
                cfg.model,
                {"epoch": epoch, "val_cer": rates.cer, "val_set": val_name, "seed": cfg.seed},
            )
        else:
            stale += 1
        torch.save(
            {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "epoch": epoch,
                "step": step,
                "best_cer": best_cer,
                "stale": stale,
            },
            last_path,
        )
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        log(json.dumps(record))
        if stale >= cfg.train.patience or (max_steps is not None and step >= max_steps):
            break
    log(f"best {val_name} CER {best_cer:.4f} -> {best_path}")
    return best_path
