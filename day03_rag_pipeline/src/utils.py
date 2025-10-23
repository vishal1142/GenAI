
## 🧱 `src/utils.py`

# utils.py — basic logging and setup
from __future__ import annotations
import os, time, random, functools
import numpy as np
from pathlib import Path

def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed)

def ensure_dir(p:Path): p.mkdir(parents=True, exist_ok=True)

def log(msg:str, file:Path|None=None):
    ts=time.strftime("%H:%M:%S"); line=f"[{ts}] {msg}"
    print(line)
    if file: open(file,"a",encoding="utf-8").write(line+"\n")

def timed(func):
    @functools.wraps(func)
    def wrap(*a,**kw):
        t=time.time(); r=func(*a,**kw); log(f"{func.__name__} took {time.time()-t:.2f}s"); return r
    return wrap
