# decorators.py — extra logging decorators
from .utils import log
import functools

def log_calls(func):
    @functools.wraps(func)
    def wrap(*a,**kw):
        log(f"CALL {func.__name__}(args={len(a)}, kwargs={list(kw.keys())})")
        return func(*a,**kw)
    return wrap
