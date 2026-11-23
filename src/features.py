from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# this python file will do data cleaning (bronze->silver) and feature creation (silver->bronze)
# using PySpark, as it will run on Databricks

"""
Feature Name	Source	        Type	    What It Represents
carrier	        Raw (Bronze)	Categorical	Which logistics company is handling the shipment (e.g., GlobalHaul).
weather	        Raw (Cleaned)	Categorical	Forecasted weather at dispatch (Clear, Rain, Snow, Fog, Unknown).
distance_km	    Raw (Bronze)	Numeric	Total route distance in kilometers.
weight_kg	    Raw (Bronze)	Numeric	Total cargo weight in kilograms.
expected_hours	Raw (Bronze)	Numeric	Baseline estimated transit duration based on distance and handling time.
"""

# clean df
def clean_bronze_to_silver(df:DataFrame) -> DataFrame:
    """
    Cleans the raw shipment data:
    - handles null values on "weather" column
    - cast timestamps on "dispatch_timestamp" column
    - filter invalid records on "distance_km" and "weight_kg" columns
    """
    cleaned_df = df.dropDuplicates(["shipment_id"])\
                   .fillna({"weather":"Unknown"})\
                   .filter((F.col("distance_km")>0) & (F.col("weight_kg")>0))\
                   .withColumn("dispatch_timestamp", F.to_timestamp("dispatch_timestamp"))
    return cleaned_df


# create new features
"""
dispatch_hour	Engineered (Gold)	Numeric	Hour of the day (0–23) extracted from dispatch_timestamp.
is_weekend	    Engineered (Gold)	Binary (0/1)	Whether the shipment was dispatched on a Saturday or Sunday.
is_cross_region	Engineered (Gold)	Binary (0/1)	1 if origin_region differs from dest_region, 0 if within the same region.
weight_per_km	Engineered (Gold)	Numeric	Load density ratio (weight_kg / distance_km).
is_severe_weather	Engineered (Gold)	Binary (0/1)	1 if weather is Snow or Fog, otherwise 0.
"""
def build_gold_features(df:DataFrame) -> DataFrame:
    feature_df = df.withColumn("dispatch_hour", F.hour("dispatch_timestamp"))\
                   .withColumn("dispatch_dayofweek", F.dayofweek("dispatch_timestamp"))\
                   .withColumn("is_weekend", F.when(F.col("dispatch_dayofweek").isin([1,7]), 1).otherwise(0))\
                   .withColumn("is_cross_region", F.when(F.col("origin_region")!=F.col("dest_region"),1).otherwise(0))\
                   .withColumn("weight_per_km", F.round(F.col("weight_kg")/F.col("distance_km"),4))\
                   .withColumn("is_severe_weather", F.when(F.col("weather").isin(["Snow", "Fog"]),1).otherwise(0))
    
    return feature_df

# verified in notebooks/tests.ipynb

