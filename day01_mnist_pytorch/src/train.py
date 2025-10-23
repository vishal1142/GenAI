# train.py — Training & evaluation loop with decorators, plots, and checkpointing.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# --------------------------------------------------------------------
# ✅ Dynamically add the "src" folder to sys.path (so utils can import)
# --------------------------------------------------------------------
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

# --------------------------------------------------------------------
# ✅ Local imports
# --------------------------------------------------------------------
from utils import timed, log_calls, simple_logger


# --------------------------------------------------------------------
# ✅ Trainer Class
# --------------------------------------------------------------------
@dataclass
class Trainer:
    """Bundle everything needed for training in one place."""
    cfg: Any
    model: nn.Module
    optimizer: torch.optim.Optimizer
    best_acc: float = 0.0
    device: str = "cpu"
    criterion: nn.Module = nn.CrossEntropyLoss()
    model_path: Path = Path("artifacts/models/best_model.pt")
    plot_path: Path = Path("artifacts/plots/loss_accuracy.png")
    log_path: Path = Path("artifacts/logs/train.log")

    def __post_init__(self) -> None:
        """Prepare device, folders, and logging."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        for path in [self.model_path, self.plot_path, self.log_path]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        simple_logger(f"Using device: {self.device}", self.log_path)

    # ----------------------------------------------------------------
    # Core Step
    # ----------------------------------------------------------------
    @timed
    @log_calls
    def _step(self, batch, train: bool = True) -> Dict[str, float]:
        """Process a single mini-batch and compute metrics."""
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        logits = self.model(inputs)
        loss = self.criterion(logits, targets)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean().item()
        return {"loss": loss.item(), "acc": acc}

    # ----------------------------------------------------------------
    # Epoch Loop
    # ----------------------------------------------------------------
    def _epoch(self, loader: DataLoader, train: bool = True) -> Dict[str, float]:
        """Iterate over all batches, compute mean loss/acc."""
        mode = "TRAIN" if train else "EVAL"
        self.model.train(mode == "TRAIN")
        epoch_loss, epoch_acc, n = 0.0, 0.0, 0

        for batch in tqdm(loader, desc=f"{mode}", leave=False):
            metrics = self._step(batch, train=train)
            n += 1
            epoch_loss += (metrics["loss"] - epoch_loss) / n
            epoch_acc += (metrics["acc"] - epoch_acc) / n

        return {"loss": epoch_loss, "acc": epoch_acc}

    # ----------------------------------------------------------------
    # Fit Loop
    # ----------------------------------------------------------------
    def fit(self, train_loader: DataLoader, test_loader: DataLoader) -> Dict[str, list]:
        """Full multi-epoch training with validation, checkpointing, and logging."""
        history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

        for epoch in range(1, self.cfg.epochs + 1):
            simple_logger(f"Epoch {epoch}/{self.cfg.epochs}", self.log_path)

            # Train
            train_m = self._epoch(train_loader, train=True)
            history["train_loss"].append(train_m["loss"])
            history["train_acc"].append(train_m["acc"])

            # Evaluate
            test_m = self._epoch(test_loader, train=False)
            history["test_loss"].append(test_m["loss"])
            history["test_acc"].append(test_m["acc"])

            # Checkpoint
            if test_m["acc"] > self.best_acc:
                self.best_acc = test_m["acc"]
                torch.save(self.model.state_dict(), self.model_path)
                simple_logger(
                    f"New best acc={self.best_acc:.4f} -> saved {self.model_path}",
                    self.log_path
                )

            # Epoch summary
            simple_logger(
                f"epoch={epoch} "
                f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']:.4f} "
                f"test_loss={test_m['loss']:.4f} test_acc={test_m['acc']:.4f}",
                self.log_path
            )

        return history

    # ----------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------
    def plot_history(self, history: Dict[str, list]) -> None:
        """Plot and save loss/accuracy curves for inspection."""
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="train_loss")
        plt.plot(history["test_loss"], label="test_loss")
        plt.plot(history["train_acc"], label="train_acc")
        plt.plot(history["test_acc"], label="test_acc")
        plt.legend()
        plt.title("Training History (Loss & Accuracy)")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.savefig(self.plot_path, bbox_inches="tight")
        plt.close()

    # ----------------------------------------------------------------
    # API symmetry
    # ----------------------------------------------------------------
    def save_best(self) -> None:
        """Checkpointing already occurs during fit; kept for API symmetry."""
        pass
# --------------------------------------------------------------------
# ✅ End of trainer.py