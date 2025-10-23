# strategies.py — Strategy pattern for pluggable optimizer & augmentation choices.
# The factories use if/elif to pick a strategy, so main training code stays clean.

from __future__ import annotations
from typing import Iterable, Tuple
import torch.optim as optim
from torchvision import transforms

# -------- Optimizer Strategy --------
def make_optimizer(name: str, params: Iterable, lr: float) -> optim.Optimizer:
    """
    Return optimizer chosen by 'name'.
    Strategy: centralize the if/elif here rather than spreading across the codebase.
    """
    name = (name or "adam").lower().strip()
    if name == "sgd":
        # Momentum (and Nesterov) helps SGD converge faster/smoother.
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif name == "adam":
        return optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer strategy: {name}")

# -------- Augmentation Strategy --------
def make_augmentations(level: str) -> Tuple["transforms.Compose", "transforms.Compose"]:
    """
    Return (train_transforms, test_transforms) for a given augmentation 'level'.
    """
    level = (level or "light").lower().strip()

    # MNIST mean/std (widely used reference values).
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    if level == "light":
        # Minor geometric variance; improves generalization slightly.
        train_tf = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            normalize,
        ])
    elif level == "none":
        # Deterministic baseline: just tensor + normalize.
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(f"Unknown augmentation level: {level}")

    test_tf = transforms.Compose([transforms.ToTensor(), normalize])
    return train_tf, test_tf
