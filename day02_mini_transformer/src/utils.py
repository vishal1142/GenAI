# utils.py — simple logging, seeding, and decorators used across the project.

from __future__ import annotations
from pathlib import Path
import os, time, random, functools
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def log(msg: str, log_file: Path | None = None) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + os.linesep)

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = func(*args, **kwargs)
        dt = time.time() - t0
        log(f"{func.__name__} took {dt:.3f}s")
        return out
    return wrapper

def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log(f"CALL {func.__name__}(args={len(args)}, kwargs={list(kwargs.keys())})")
        return func(*args, **kwargs)
    return wrapper
