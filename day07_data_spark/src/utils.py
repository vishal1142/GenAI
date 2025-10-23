# utils.py — logging, timing decorators, SparkSession builder
from __future__ import annotations
import time, functools, yaml
from pathlib import Path
from pyspark.sql import SparkSession

def log(msg: str) -> None:
    print(f"[LOG] {msg}")

def timed(fn):
    @functools.wraps(fn)
    def _wrap(*a, **kw):
        t0 = time.time()
        out = fn(*a, **kw)
        dt = time.time() - t0
        log(f"{fn.__name__} took {dt:.2f}s")
        return out
    return _wrap

def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(open(path, encoding="utf-8"))

def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def build_spark(app_name: str, master: str, shuffle_partitions: int, sink: str):
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
    )
    # Optional Delta Lake
    if sink == "delta":
        builder = (
            builder
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
