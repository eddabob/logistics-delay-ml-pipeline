# Databricks notebook source
import sys
sys.path.append("../src")

from pyspark.sql import SparkSession
from generate_data import generate_logistics_data
from features import clean_bronze_to_silver, build_gold_features
from train_model import train_and_log

spark = SparkSession.builder.getOrCreate()

# 1. Generate local temp CSV & load into Spark
# 1. Save CSV inside your Workspace user folder so Spark Connect can access it
current_user = spark.sql("SELECT current_user()").collect()
bronze_csv_path = f"/Workspace/Users/{current_user}/bronze_shipments.csv"
generate_logistics_data(num_rows=5000, output_path=bronze_csv_path)

bronze_df = spark.read.csv(f"file:{bronze_csv_path}", header=True, inferSchema=True)

# 2. Silver & Gold transformations
silver_df = clean_bronze_to_silver(bronze_df)
gold_df = build_gold_features(silver_df)

# 3. Train & Log to Databricks MLflow
gold_pdf = gold_df.toPandas()
# Note: On Databricks, use an experiment path in your user folder: e.g. "/Users/<your-email>/logistics-delay-ml"
train_and_log(gold_pdf, experiment_name="/Users/anna20240312@gmail.com/logistics-delay-ml")
