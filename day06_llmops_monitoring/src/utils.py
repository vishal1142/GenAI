from __future__ import annotations
import os, time, functools, random, numpy as np, pandas as pd
from pathlib import Path

def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed)

def timed(func):
    @functools.wraps(func)
    def wrap(*a,**kw):
        t=time.time(); r=func(*a,**kw)
        print(f"{func.__name__} took {time.time()-t:.2f}s")
        return r
    return wrap

def ensure_dir(p:Path): p.mkdir(parents=True, exist_ok=True)
