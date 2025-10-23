## 🛠️ `src/utils.py`

# utils.py — logging, seeding, decorators

from __future__ import annotations
from pathlib import Path
import time, os, random, functools
import numpy as np

def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def timed(func):
    """Decorator: measure wall time and log it."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = func(*args, **kwargs)
        dt = time.time() - t0
        log(f"{func.__name__} took {dt:.3f}s")
        return out
    return wrapper

def log_calls(func):
    """Decorator: log function name and arg summary."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log(f"CALL {func.__name__}(args={len(args)}, kwargs={list(kwargs.keys())})")
        return func(*args, **kwargs)
    return wrapper
