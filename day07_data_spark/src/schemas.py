# schemas.py — Spark schema + simple cast utilities
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType

def customers_schema() -> StructType:
    return StructType([
        StructField("customer_id", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("income", DoubleType(), True),
        StructField("country", StringType(), True),
        StructField("signup_ts", StringType(), True),  # read as string; cast later to timestamp
    ])
