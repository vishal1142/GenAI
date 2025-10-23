# main.py — entrypoint to run Spark ETL + DQ + Metrics pipeline
from __future__ import annotations
from src.utils import load_yaml, build_spark, log
from src.orchestrator import run_pipeline

def main():
    cfg = load_yaml("src/config.yaml")
    spark = build_spark(
        app_name = cfg["spark"]["app_name"],
        master = cfg["spark"]["master"],
        shuffle_partitions = cfg["spark"]["shuffle_partitions"],
        sink = cfg["io"]["sink"],
    )
    try:
        run_pipeline(spark, cfg)
    finally:
        log("Stopping SparkSession.")
        spark.stop()

if __name__ == "__main__":
    main()
