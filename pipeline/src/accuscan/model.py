"""CRNN: a VGG-style CNN that collapses height, a 2-layer BiLSTM and a CTC head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .charset import NUM_CLASSES
from .preprocess import INPUT_HEIGHT

DOWNSAMPLE = 4  # width reduction: one output frame per 4 input columns


@dataclass(frozen=True)
class ModelConfig:
    channels: tuple[int, ...] = (32, 64, 128, 224, 320)
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.2
    num_classes: int = NUM_CLASSES


def _block(c_in: int, c_out: int) -> list[nn.Module]:
    return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True)]


class CRNN(nn.Module):
    """[B, 1, 64, W] -> log-probs [W/4, B, C] (time-major, as CTC expects)."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        c1, c2, c3, c4, c5 = cfg.channels
        self.features = nn.Sequential(
            *_block(1, c1),
            nn.MaxPool2d(2, 2),  # 32 x W/2
            *_block(c1, c2),
            nn.MaxPool2d(2, 2),  # 16 x W/4
            *_block(c2, c3),
            *_block(c3, c3),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8 x W/4
            *_block(c3, c4),
            *_block(c4, c4),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4 x W/4
            nn.Conv2d(c4, c5, (4, 1), bias=False),  # collapse height -> 1 x W/4
            nn.BatchNorm2d(c5),
            nn.ReLU(True),
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.rnn = nn.LSTM(
            c5,
            cfg.hidden,
            num_layers=cfg.layers,
            bidirectional=True,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )
        self.head = nn.Linear(2 * cfg.hidden, cfg.num_classes)

    def forward(self, image: Tensor) -> Tensor:
        feats = self.features(image)  # [B, C, 1, T]
        seq = feats.squeeze(2).permute(2, 0, 1)  # [T, B, C]
        seq, _ = self.rnn(self.dropout(seq))
        logits = self.head(self.dropout(seq))
        return torch.log_softmax(logits, dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def output_frames(width: int) -> int:
    return width // DOWNSAMPLE


if INPUT_HEIGHT != 64:  # pragma: no cover
    raise RuntimeError("the CNN collapses exactly 64 rows")
