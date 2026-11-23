import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def generate_logistics_data(num_rows: int = 500, output_path: str = "data/bronze_shipments.csv"):
    """
    Generates realistic UK logistics shipment records.
    The goal of src/generate_data.py is to create a realistic, reproducible synthetic dataset 
    that mimics messy real-world logistics operations while embedding hidden statistical patterns 
    so your Random Forest model actually has something meaningful to learn

    What It Predicts (The Target Variable)
    • Target Column: is_delayed
    • Values: 1 (Delayed) or 0 (On Time)
    • Business Goal: Flag high-risk shipments early at dispatch 
    so operations teams can proactively warn customers or reroute cargo.
    """

    """
    The model uses 10 input features (feature_cols in src/train_model.py), 
    split into raw attributes available at dispatch and engineered features created by your PySpark pipeline

Feature Name	Source	        Type	    What It Represents
carrier	        Raw (Bronze)	Categorical	Which logistics company is handling the shipment (e.g., GlobalHaul).
weather	        Raw (Cleaned)	Categorical	Forecasted weather at dispatch (Clear, Rain, Snow, Fog, Unknown).
distance_km	    Raw (Bronze)	Numeric	Total route distance in kilometers.
weight_kg	    Raw (Bronze)	Numeric	Total cargo weight in kilograms.
expected_hours	Raw (Bronze)	Numeric	Baseline estimated transit duration based on distance and handling time.
dispatch_hour	Engineered (Gold)	Numeric	Hour of the day (0–23) extracted from dispatch_timestamp.
is_weekend	    Engineered (Gold)	Binary (0/1)	Whether the shipment was dispatched on a Saturday or Sunday.
is_cross_region	Engineered (Gold)	Binary (0/1)	1 if origin_region differs from dest_region, 0 if within the same region.
weight_per_km	Engineered (Gold)	Numeric	Load density ratio (weight_kg / distance_km).
is_severe_weather	Engineered (Gold)	Binary (0/1)	1 if weather is Snow or Fog, otherwise 0.

Columns Explicitly Excluded (And Why)
Not every column in bronze_shipments.csv should be fed into the model:
• shipment_id: A unique ID with zero predictive pattern; including it would cause overfitting.
• dispatch_timestamp: Raw timestamps aren't directly usable by Random Forests, so we extract dispatch_hour and is_weekend instead.
• actual_hours: Excluded to prevent data leakage. You only know how many hours a trip actually took after it finishes—using it to predict delays at dispatch would be cheating.
    """
    np.random.seed(42)
    carriers = ["SwiftFreight", "GlobalHaul", "EcoExpress", "PrimeCargo"]
    weather_conditions = ["Clear", "Rain", "Snow", "Fog"]
    regions = ["North", "South", "East", "West"]

    base_date = datetime(2026, 1, 1) # 2026-01-01 00:00:00
    dispatch_dates = [base_date + timedelta(hours=int(x)) for x in np.random.randint(0,24*180, num_rows)]
    #np.random.randint(0,24*180, num_rows) # returns a list of length num_rows # 180 days
    
    distance_km = np.random.uniform(50, 2500, num_rows).round(1) 
    #num_rows containing random floating-point numbers Uniformly distributed between 50 (inclusive) and 2500 (exclusive).
    weight_kg = np.random.uniform(10, 5000, num_rows).round(1)
    carrier = np.random.choice(carriers, num_rows, p=[0.35, 0.3, 0.2, 0.15])
    weather = np.random.choice(weather_conditions, num_rows, p=[0.6, 0.25, 0.1, 0.05])
    origin_region = np.random.choice(regions, num_rows)
    dest_region = np.random.choice(regions, num_rows)

    # calculate expected transit hours (avg speed is about 65km/h+handling buffer)
    expected_hours = (distance_km/65.0)+12.0

    # Add in delay estimation (weather, heavy loads, specific carriers)
    delay_risk = (
        (weather == "Snow") * 0.45 +
        (weather == "Rain") * 0.2 +
        (weather == "Fog") * 0.15 + 
        (carrier == "GlobalHaul") * 0.15 + 
        (distance_km > 1500) * 0.15 + 
        (weight_kg > 3500) * 0.1 +
        np.random.normal(0,0.1,num_rows)
    ) 
    #print(delay_risk) # an np.array of length row_nums, range from 1 to -0.3
    #print(np.mean(np.random.normal(0,0.1,num_rows)))
    #print((weather == "Snow")) # True or False
    #print((weather == "Snow") * 0.45) # 0.45 or 0
    is_delayed = (delay_risk > 0.35).astype(int)
    
    # Actual transit hours
    #delay_hours = [np.random.uniform(4,36) if x == 1 else np.random.uniform(-2, 1.5) for x in is_delayed]
    #np.where is still better practice for data science workflows because vectorized operations avoid slow Python loops.
    #np.where(condition, x, y) is a vectorized if-else statement that evaluates a condition across an entire array (is_delayed)
    #  at once, picking values from x where the condition is True and from y where it is False.
    #np.random.uniform(4, 36, num_rows) generates a 1D NumPy array of num_rows random decimal numbers evenly distributed between 4.0 and 36.0.
    #uniform distribution gives every decimal value in the [4, 36) range an equal probability of being chosen (e.g., 4.12, 19.85, and 35.91 are all equally likely).
    delay_hours = np.where(is_delayed == 1, np.random.uniform(4,36, num_rows), np.random.uniform(-2,1.5))
    actual_hours = np.maximum(1.0, expected_hours+delay_hours)
    #print(max(actual_hours)) # max is 83, min is 1
    data_dict = {
        "shipment_id": [f"SHP-{100000+i}" for i in range(num_rows)],
        "dispatch_timestamp": dispatch_dates,
        "origin_region": origin_region,
        "dest_region": dest_region,
        "carrier": carrier,
        "weather": weather,
        "distance_km": distance_km,
        "weight_kg": weight_kg,
        "expected_hours": expected_hours.round(2),
        "actual_hours": actual_hours.round(2),
        "is_delayed": is_delayed
    }
    #print(type(expected_hours)) # np.ndarray
    #print(type(dispatch_dates)) # list

    df = pd.DataFrame(data_dict)

    # add in 2% mssing values to test the silver layer cleaning
    mask = np.random.random(num_rows) < 0.02
    df.loc[mask, "weather"] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_rows} rows at {output_path}. Delay rate: {is_delayed.mean():,.2%}")

if __name__ == "__main__":
    generate_logistics_data()