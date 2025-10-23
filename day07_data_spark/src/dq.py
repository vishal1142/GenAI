# dq.py — Data Quality checks (nulls, ranges, uniqueness, optional FK)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, countDistinct, isnan, when, lit
from pyspark.sql import functions as F
from .utils import log, timed

@dataclass
class DQConfig:
    id_column: str
    required_columns: List[str]
    numeric_ranges: Dict[str, Tuple[float, float]]
    unique_keys: List[str]
    fk_check: Dict

def _null_ratio(df: DataFrame, colname: str) -> float:
    total = df.count()
    nulls = df.filter(col(colname).isNull()).count()
    return (nulls / total) if total > 0 else 0.0

@timed
def run_dq(df: DataFrame, cfg: DQConfig, policy: str = "strict") -> Dict[str, float | bool]:
    results = {}

    # Required columns present?
    for c in cfg.required_columns:
        exists = c in df.columns
        results[f"exists_{c}"] = bool(exists)

    # Null ratios
    for c in cfg.required_columns:
        if c in df.columns:
            results[f"null_ratio_{c}"] = _null_ratio(df, c)

    # Numeric ranges
    for c, (lo, hi) in cfg.numeric_ranges.items():
        if c in df.columns:
            bad = df.filter( (col(c) < lo) | (col(c) > hi) ).count()
            results[f"out_of_range_{c}"] = bad

    # Uniqueness
    for k in cfg.unique_keys:
        if k in df.columns:
            total = df.count()
            distinct = df.select(k).distinct().count()
            dupes = total - distinct
            results[f"dupes_{k}"] = dupes

    # FK check (optional)
    if cfg.fk_check.get("enabled", False):
        from pyspark.sql import SparkSession
        spark = df.sparkSession
        dim = spark.read.parquet(cfg.fk_check["dim_path"])
        joined = df.join(dim.select(F.col(cfg.fk_check["dim_key"]).alias("fk_key")),
                         df[cfg.fk_check["fk_col"]] == F.col("fk_key"),
                         "left")
        missing = joined.filter(F.col("fk_key").isNull()).count()
        results["fk_missing"] = missing

    # Policy evaluation (very simple gates)
    strict_fail = []
    for c in cfg.required_columns:
        if not results.get(f"exists_{c}", False): strict_fail.append(f"missing_col:{c}")
        if results.get(f"null_ratio_{c}", 0) > 0.05: strict_fail.append(f"nulls:{c}")
    for c in cfg.numeric_ranges.keys():
        if results.get(f"out_of_range_{c}", 0) > 0: strict_fail.append(f"range:{c}")
    for k in cfg.unique_keys:
        if results.get(f"dupes_{k}", 0) > 0: strict_fail.append(f"dupes:{k}")

    results["passed"] = len(strict_fail) == 0 if policy == "strict" else True
    results["violations"] = ",".join(strict_fail)
    log(f"DQ results: {results}")
    return results
