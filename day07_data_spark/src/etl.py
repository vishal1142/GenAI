# etl.py — Extract → Transform → Load with comments on every step
from __future__ import annotations
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, trim, when, to_timestamp, lower
from .utils import log, timed, ensure_parent
from .schemas import customers_schema

@timed
def extract_csv(spark, path: str) -> DataFrame:
    # Read CSV with a declared schema for stability
    log(f"Reading source CSV: {path}")
    return (
        spark.read.format("csv")
        .option("header", True)
        .schema(customers_schema())
        .load(path)
    )

@timed
def transform_clean(df: DataFrame) -> DataFrame:
    # 1) Trim whitespace, normalize text, handle null-like strings
    df1 = (
        df
        .withColumn("customer_id", trim(col("customer_id")))
        .withColumn("country", lower(trim(col("country"))))
        .withColumn("signup_ts", trim(col("signup_ts")))
    )
    # 2) Cast timestamp reliably (assuming ISO-like format)
    df2 = df1.withColumn("signup_ts", to_timestamp(col("signup_ts")))
    # 3) Replace negative/invalid values with null for later DQ handling
    df3 = (
        df2
        .withColumn("age", when((col("age") < 0) | (col("age") > 150), None).otherwise(col("age")))
        .withColumn("income", when(col("income") < 0, None).otherwise(col("income")))
    )
    return df3

@timed
def load_table(df: DataFrame, path: str, sink: str, mode: str = "overwrite") -> None:
    # Write to Parquet or Delta; partitioning optional (e.g., by country)
    ensure_parent(path)
    log(f"Writing {sink} table to: {path} (mode={mode})")
    if sink == "delta":
        (
            df.write.format("delta")
            .mode(mode)
            .save(path)
        )
    else:
        (
            df.write.format("parquet")
            .mode(mode)
            .save(path)
        )
