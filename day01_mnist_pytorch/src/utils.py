# utils.py — common helpers: seeding, directory creation, simple logging, and decorators.
# Decorators add cross-cutting concerns (timing/logging) without polluting core logic.

from __future__ import annotations
from pathlib import Path
import os, random, time, functools
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # reproducible convs
    torch.backends.cudnn.benchmark = False     # no autotune variance

def ensure_dirs(paths: list[Path]) -> None:
    """Create directories if missing (idempotent)."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def simple_logger(msg: str, log_file: Path | None = None) -> None:
    """Print to stdout and optionally append to a log file."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + os.linesep)

def log_banner(title: str, log_file: Path | None = None) -> None:
    """Pretty section banners for readability."""
    sep = "=" * max(10, len(title) + 4)
    simple_logger(sep, log_file)
    simple_logger(f"  {title}", log_file)
    simple_logger(sep, log_file)

def timed(func):
    """Decorator: measure wall time of the wrapped function and log it."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        simple_logger(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

def log_calls(func):
    """Decorator: log function name and argument info before calling it."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        simple_logger(f"CALL {func.__name__}(args={len(args)}, kwargs={list(kwargs.keys())})")
        return func(*args, **kwargs)
    return wrapper
