"""Arquitetura CNN customizada e constantes de otimização."""

from dataclasses import dataclass
import torch.nn as nn


@dataclass(frozen=True)
class OptimizationConfig:
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout_rate: float = 0.4


HYPERPARAMETER_GRID = {
    "learning_rate": [1e-3, 5e-4],
    "weight_decay": [1e-4, 5e-4],
    "dropout_rate": [0.3, 0.4, 0.5],
    "batch_size": [64, 128],
}


class CustomCNN(nn.Module):
    """CNN simples com BatchNorm + Dropout para regularização."""

    def __init__(self, in_channels: int, num_classes: int = 10, dropout_rate: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
