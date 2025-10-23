# orchestrator.py — simple linear DAG (Bronze → Silver → DQ → Gold + Metrics)
from __future__ import annotations
from dataclasses import dataclass
from .utils import log, timed
from .etl import extract_csv, transform_clean, load_table
from .dq import run_dq, DQConfig
from .metrics import profile_and_compare
from pyspark.sql import DataFrame

@dataclass
class Paths:
    source: str
    bronze: str
    silver: str
    gold: str
    ref_profile: str
    out_profile: str
    sink: str
    write_mode: str

@timed
def run_pipeline(spark, cfg: dict) -> None:
    # --- Extract ---
    df_raw = extract_csv(spark, cfg["io"]["source_path"])

    # --- Bronze (raw persisted) ---
    load_table(df_raw, cfg["io"]["bronze_path"], cfg["io"]["sink"], cfg["run"]["write_mode"])

    # --- Transform to Silver (clean) ---
    df_clean = transform_clean(df_raw)
    load_table(df_clean, cfg["io"]["silver_path"], cfg["io"]["sink"], cfg["run"]["write_mode"])

    # --- Data Quality (fail fast if strict) ---
    dqres = run_dq(
        df_clean,
        DQConfig(
            id_column = cfg["dq"]["id_column"],
            required_columns = cfg["dq"]["required_columns"],
            numeric_ranges = cfg["dq"]["numeric_ranges"],
            unique_keys = cfg["dq"]["unique_keys"],
            fk_check = cfg["dq"]["fk_check"],
        ),
        policy = cfg["dq"]["policy"]
    )
    if not dqres["passed"]:
        log(f"[DQ FAIL] Violations: {dqres['violations']}")
        # In strict mode, you might raise an exception; here we proceed but log.

    # --- Simple Gold metrics (per-country aggregates as an example) ---
    df_gold = (
        df_clean.groupBy("country")
        .agg({"customer_id":"count","income":"avg","age":"avg"})
        .withColumnRenamed("count(customer_id)", "customers")
        .withColumnRenamed("avg(income)", "avg_income")
        .withColumnRenamed("avg(age)", "avg_age")
    )
    load_table(df_gold, cfg["io"]["gold_path"], cfg["io"]["sink"], cfg["run"]["write_mode"])

    # --- Profile & drift check against reference ---
    metrics = profile_and_compare(
        df_clean.select("age","income"),              # focus numeric for demo
        cfg["io"]["ref_path"],
        cfg["io"]["ref_path"].replace("reference", "current")  # write current alongside ref
    )
    log(f"Pipeline finished with drift_score={metrics['drift']:.4f}")
