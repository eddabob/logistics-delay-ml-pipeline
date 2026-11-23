# This test file uses pytest to automatically verify that your PySpark data cleaning
# and feature engineering functions behave properly on known edge cases
# (duplicates, missing values, and invalid negative distances).
import pytest

from pyspark.sql import SparkSession
from src.features import clean_bronze_to_silver, build_gold_features

@pytest.fixture(scope = "session")
def spark():
    return (
        SparkSession.builder.master("local[1]").appName("pytest-pyspark-testing").getOrCreate()
    )
# Creates a lightweight, single-threaded (local[1]) Spark session once per test run 
# and injects it into any test function that asks for spark. Starts the JVM once and caches 
# the SparkSession object across all tests so pytest finishes in seconds instead of restarting 
# Spark for every test. 


def test_feature_engineering_pipeline(spark):
    sample_data = [
        #Row 1 (SHP-1): Valid shipment, missing weather (None), dispatched on Saturday 2026-05-02, moving North → South, 1000 kg over 500 km.
        ("SHP-1", "2026-05-02 14:00:00", "North", "South", "SwiftFreight", None, 500.0, 1000.0, 19.7, 20.0, 0),
        #Row 2 (SHP-1): Exact duplicate of Row 1 to test dropDuplicates(["shipment_id"]). # need to drop this row
        ("SHP-1", "2026-05-02 14:00:00", "North", "South", "SwiftFreight", None, 500.0, 1000.0, 19.7, 20.0, 0), # Duplicate
        #Row 3 (SHP-2): Invalid negative distance (-10.0) to test .filter((F.col("distance_km") > 0) & ...): need to filter out this row
        ("SHP-2", "2026-05-04 08:00:00", "East", "East", "EcoExpress", "Snow", -10.0, 500.0, 10.0, 12.0, 1),    # Invalid distance
    ] # so finally, should left only one row and eplaces weather=None with "Unknown"

    columns = [
        "shipment_id", "dispatch_timestamp", "origin_region", "dest_region",
        "carrier", "weather", "distance_km", "weight_kg", "expected_hours", "actual_hours", "is_delayed"
    ]

    raw_df = spark.createDataFrame(sample_data, columns)
    silver_df = clean_bronze_to_silver(raw_df)
    gold_df = build_gold_features(silver_df)
    results = gold_df.collect() #Pulls the Spark DataFrame back into Python as a list of Row objects: 
    #[Row(shipment_id='SHP-1', ..., weather='Unknown', is_weekend=1, is_cross_region=1, weight_per_km=2.0, ...)].

    # should deduplicate SHP-1 and filter out SHP-2
    assert len(results) == 1

    row = results[0]# Unpacks the single remaining Row object so individual column values can be inspected.

    assert row['weather'] == "Unknown" # Verifies fillna({"weather": "Unknown"}) worked.
    assert row["is_weekend"] == 1 # 2026-05-02 is Saturday, so is_weekend must evaluate to 1
    assert row["is_cross_region"] == 1 # origin_region ("North") != dest_region ("South"), so is_cross_region must be 1
    assert row["weight_per_km"] == 2.0 #1000.0 kg / 500.0 km = 2.0, verifying your ratio math.