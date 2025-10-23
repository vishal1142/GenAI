# config.py — typed, validated config object (Pydantic v2).
# Centralizes hyperparameters, paths, and seeds.

from __future__ import annotations
from pydantic import BaseModel, Field
from pathlib import Path
import datetime as dt

class AppConfig(BaseModel):
    # ---- training hyperparameters ----
    batch_size: int = Field(default=64, ge=1)      # mini-batch size
    epochs: int = Field(default=3, ge=1)           # keep quick by default
    lr: float = Field(default=1e-3, gt=0)          # learning rate
    optimizer: str = Field(default="adam")         # "adam" or "sgd"
    augment: str = Field(default="light")          # "light" or "none"
    num_workers: int = Field(default=2, ge=0)      # DataLoader workers (0 if Windows)
    seed: int = Field(default=42)                  # reproducibility

    # ---- artifacts/logging ----
    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("artifacts/models")
    plots_dir: Path = Path("artifacts/plots")
    logs_dir: Path = Path("artifacts/logs")
    run_tag: str = Field(default_factory=lambda: dt.datetime.now().strftime("%Y%m%d_%H%M%S"))

    def merge_cli(self, args) -> "AppConfig":
        """
        Create a NEW config with CLI overrides applied.
        Immutable-style update keeps provenance clean.
        """
        data = self.model_dump()
        for key in ["batch_size", "epochs", "lr", "optimizer", "augment"]:
            val = getattr(args, key, None)
            if val is not None:
                data[key] = val
        return AppConfig(**data)
