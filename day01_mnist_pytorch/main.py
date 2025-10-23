

# 🚀 `main.py`

# main.py — Entrypoint orchestrating config → data → model → train.
# Each step is explained so you can follow how functions call each other.

from __future__ import annotations             # allow forward-refs in type hints
import argparse                                # parse CLI flags to override config
from src.config import AppConfig               # validated config (Pydantic)
from src.data import build_dataloaders         # train/test DataLoaders
from src.model import build_model              # CNN model factory
from src.strategies import make_optimizer, make_augmentations  # Strategy pattern
from src.train import Trainer                  # training loop wrapper
from src.utils import set_seed, ensure_dirs, log_banner        # utilities

def parse_args() -> argparse.Namespace:
    """Define CLI options so you can tweak batch size/epochs/optimizer at runtime."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default=None, choices=["sgd", "adam"],
                        help="Optimizer strategy")
    parser.add_argument("--augment", type=str, default=None, choices=["light", "none"],
                        help="Augmentation strategy")
    return parser.parse_args()

def main() -> None:
    """Top-level orchestration (called when this file is run)."""
    # 1) Load defaults and merge CLI overrides (config-first design).
    cfg = AppConfig()                   # defaults live here
    args = parse_args()                 # parse CLI
    cfg = cfg.merge_cli(args)           # produce a new config with overrides

    # 2) Prepare output folders and reproducibility seed.
    ensure_dirs([cfg.artifacts_dir, cfg.models_dir, cfg.plots_dir, cfg.logs_dir])
    set_seed(cfg.seed)
    log_banner("MNIST TRAINING START")

    # 3) Choose augmentation strategy and build DataLoaders.
    tf_train, tf_test = make_augmentations(cfg.augment)  # Strategy via if/elif inside factory
    train_loader, test_loader = build_dataloaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train_transforms=tf_train,
        test_transforms=tf_test,
    )

    # 4) Build model and optimizer (strategy again).
    model = build_model()                                   # small CNN
    optimizer = make_optimizer(cfg.optimizer, model.parameters(), lr=cfg.lr)

    # 5) Train/evaluate via Trainer abstraction.
    trainer = Trainer(cfg=cfg, model=model, optimizer=optimizer)
    history = trainer.fit(train_loader=train_loader, test_loader=test_loader)
    trainer.plot_history(history)
    trainer.save_best()
    log_banner("MNIST TRAINING DONE")

if __name__ == "__main__":
    main()
