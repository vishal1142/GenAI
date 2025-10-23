# data.py — dataset + DataLoader creation using torchvision MNIST and transforms.

from __future__ import annotations
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def build_dataloaders(
    batch_size: int,
    num_workers: int,
    train_transforms: "transforms.Compose",
    test_transforms: "transforms.Compose",
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/test DataLoaders.
    - train uses augmentation/normalization
    - test uses deterministic normalization
    """
    # Fetch MNIST on first run, store under ./data
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=train_transforms)
    test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=test_transforms)

    # DataLoaders handle batching, shuffling, multiprocessing prefetch.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
