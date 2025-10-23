# model.py — compact CNN for MNIST (1×28×28).
# Clear layer-by-layer explanation.

from __future__ import annotations
import torch
import torch.nn as nn

class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Block 1: Conv -> ReLU -> MaxPool (28x28 -> 14x14)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # keep HW with padding
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Block 2: Conv -> ReLU -> MaxPool (14x14 -> 7x7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Head: Flatten -> Linear to 10 digits
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))  # (N, 1, 28, 28) -> (N, 32, 14, 14)
        x = self.pool2(self.relu2(self.conv2(x)))  # (N, 32, 14, 14) -> (N, 64, 7, 7)
        x = self.flatten(x)                        # (N, 64*7*7)
        x = self.fc(x)                             # logits (no softmax; CE expects logits)
        return x

def build_model() -> nn.Module:
    """Factory for main.py, easy to swap architectures later."""
    return MnistCNN()
