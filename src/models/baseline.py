from __future__ import annotations

import torch
from torch import nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width * 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(width * 2), nn.GELU(),
            nn.Conv1d(width * 2, width * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(width * 4), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.output_dim = width * 4

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.net(signal).flatten(1)


class BaselineCNN(nn.Module):
    def __init__(self, mode: str, width: int = 32, num_classes: int = 3) -> None:
        super().__init__()
        if mode not in {"vibration", "current", "fusion"}:
            raise ValueError(f"unknown baseline mode: {mode}")
        self.mode = mode
        self.vibration_encoder = ConvEncoder(1, width) if mode in {"vibration", "fusion"} else None
        self.current_encoder = ConvEncoder(2, width) if mode in {"current", "fusion"} else None
        feature_dim = 0
        if self.vibration_encoder is not None:
            feature_dim += self.vibration_encoder.output_dim
        if self.current_encoder is not None:
            feature_dim += self.current_encoder.output_dim
        self.classifier = nn.Sequential(nn.Linear(feature_dim, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, num_classes))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        if self.vibration_encoder is not None:
            features.append(self.vibration_encoder(batch["vibration"]))
        if self.current_encoder is not None:
            features.append(self.current_encoder(batch["current"]))
        return self.classifier(torch.cat(features, dim=1))
