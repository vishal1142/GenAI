from __future__ import annotations
import random, os, numpy as np, torch, time, functools
import yaml
from pathlib import Path

def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def timed(func):
    @functools.wraps(func)
    def wrap(*a,**kw):
        t=time.time(); r=func(*a,**kw)
        print(f"{func.__name__} took {time.time()-t:.2f}s")
        return r
    return wrap

def load_yaml(path:str|Path)->dict:
    return yaml.safe_load(open(path,encoding="utf-8"))
