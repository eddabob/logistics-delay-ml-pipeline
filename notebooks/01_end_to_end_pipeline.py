import sys
sys.path.append("../src")
import os
import pandas as pd

from pyspark.sql import SparkSession
from generate_data import generate_logistics_data
from features import clean_bronze_to_silver, build_gold_features
from train_model import train_and_log

spark = SparkSession.builder.getOrCreate()

# 1. Generate CSV in driver /tmp and load into Spark via Pandas
bronze_csv_path = "/tmp/bronze_shipments.csv"
generate_logistics_data(num_rows=5000, output_path=bronze_csv_path)

bronze_df = spark.createDataFrame(pd.read_csv(bronze_csv_path))

# 2. Silver & Gold transformations
silver_df = clean_bronze_to_silver(bronze_df)
gold_df = build_gold_features(silver_df)

# 3. Train & Log to Databricks MLflow
gold_pdf = gold_df.toPandas()
# Note: On Databricks, use an experiment path in your user folder: e.g. "/Users/<your-email>/logistics-delay-ml"
train_and_log(gold_pdf, experiment_name="/Users/anna20240312@gmail.com/logistics-delay-ml")