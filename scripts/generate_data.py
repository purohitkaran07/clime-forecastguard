"""
ForecastGuard AI - Synthetic Demonstration Dataset Generator
--------------------------------------------------------------
Generates a realistic (but SYNTHETIC) dataset simulating medium-range
(Day 1-10) NWP forecast behaviour across Indian states, with engineered
features and a "forecast_bust" label used to train the reliability model.

IMPORTANT (data leakage): the model is trained only on variables that
would be KNOWN AT FORECAST TIME (forecast values, lead time, spatial /
historical-error statistics). It never sees the future observed value
directly as an input feature - the observed value is only used to
DERIVE the label (bust / no-bust) during offline training, exactly as
real forecast verification works.
"""
import numpy as np
import pandas as pd
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(42)

REGIONS = [
    {"name": "Rajasthan", "lat": 27.0, "lon": 74.2, "base_precip": 15, "volatility": 1.4},
    {"name": "Gujarat", "lat": 22.3, "lon": 71.2, "base_precip": 20, "volatility": 1.2},
    {"name": "Maharashtra", "lat": 19.7, "lon": 75.7, "base_precip": 35, "volatility": 1.3},
    {"name": "Madhya Pradesh", "lat": 23.5, "lon": 78.6, "base_precip": 28, "volatility": 1.1},
    {"name": "Uttar Pradesh", "lat": 26.8, "lon": 80.9, "base_precip": 25, "volatility": 1.0},
    {"name": "Odisha", "lat": 20.9, "lon": 85.1, "base_precip": 45, "volatility": 1.5},
    {"name": "West Bengal", "lat": 22.9, "lon": 87.8, "base_precip": 48, "volatility": 1.4},
    {"name": "Assam", "lat": 26.2, "lon": 92.9, "base_precip": 55, "volatility": 1.3},
    {"name": "Kerala", "lat": 10.5, "lon": 76.2, "base_precip": 50, "volatility": 1.2},
    {"name": "Tamil Nadu", "lat": 11.1, "lon": 78.7, "base_precip": 30, "volatility": 1.2},
    {"name": "Karnataka", "lat": 15.3, "lon": 75.7, "base_precip": 32, "volatility": 1.1},
    {"name": "Delhi", "lat": 28.7, "lon": 77.1, "base_precip": 18, "volatility": 0.9},
    {"name": "Bihar", "lat": 25.9, "lon": 85.1, "base_precip": 33, "volatility": 1.2},
    {"name": "Punjab", "lat": 31.1, "lon": 75.3, "base_precip": 20, "volatility": 0.9},
    {"name": "Andhra Pradesh", "lat": 15.9, "lon": 79.7, "base_precip": 30, "volatility": 1.2},
    {"name": "Telangana", "lat": 18.1, "lon": 79.0, "base_precip": 28, "volatility": 1.1},
    {"name": "Jharkhand", "lat": 23.6, "lon": 85.3, "base_precip": 34, "volatility": 1.2},
    {"name": "Chhattisgarh", "lat": 21.3, "lon": 81.9, "base_precip": 36, "volatility": 1.2},
]

DAYS = list(range(1, 11))
SAMPLES_PER_CELL = 40  # synthetic "historical dates" simulated per region/day


def gen_row(region, day, rng):
    lead_time = day
    # reliability decays with lead time, plus regional volatility
    lead_factor = 1 + 0.18 * (lead_time - 1)

    precipitation = max(0, rng.normal(region["base_precip"] * lead_factor * 0.35, 12 * region["volatility"]))
    temperature = rng.normal(29, 4) - 0.15 * lead_time
    pressure = rng.normal(1008, 5)
    humidity = np.clip(rng.normal(65 + 0.5 * lead_time, 10), 10, 100)
    wind_speed = max(0, rng.normal(12 + 0.4 * lead_time, 5))
    wind_direction = rng.uniform(0, 360)
    latitude = region["lat"]
    longitude = region["lon"]

    spatial_gradient = max(0, rng.normal(0.4 * region["volatility"] * lead_factor, 0.2))
    historical_error = max(0, rng.normal(8 * region["volatility"] * lead_factor, 4))
    pressure_variation = max(0, rng.normal(2.5 * lead_factor, 1.2))
    precipitation_variability = max(0, rng.normal(6 * region["volatility"] * lead_factor, 3))

    # "true" observed value used ONLY to derive the label (never fed to the model as a feature)
    error_std = 6 * region["volatility"] * lead_factor
    observed_precip = max(0, precipitation + rng.normal(0, error_std) + 0.6 * spatial_gradient * 10)
    forecast_error = abs(observed_precip - precipitation)

    return {
        "region": region["name"],
        "lead_time": lead_time,
        "precipitation": round(precipitation, 1),
        "temperature": round(temperature, 1),
        "pressure": round(pressure, 1),
        "humidity": round(humidity, 1),
        "wind_speed": round(wind_speed, 1),
        "wind_direction": round(wind_direction, 1),
        "latitude": latitude,
        "longitude": longitude,
        "spatial_gradient": round(spatial_gradient, 3),
        "historical_error": round(historical_error, 2),
        "pressure_variation": round(pressure_variation, 2),
        "precipitation_variability": round(precipitation_variability, 2),
        "_observed_precip": round(observed_precip, 1),   # NOT a model feature - label source only
        "_forecast_error": round(forecast_error, 1),      # NOT a model feature - label source only
    }


def main():
    rows = []
    for region in REGIONS:
        for day in DAYS:
            for _ in range(SAMPLES_PER_CELL):
                rows.append(gen_row(region, day, RNG))

    df = pd.DataFrame(rows)

    # Bust label: forecast error exceeds the 80th percentile of historical error
    # FOR THAT REGION'S OWN error distribution (region-relative threshold),
    # which is standard practice in forecast verification.
    threshold = df.groupby("region")["_forecast_error"].transform(lambda x: x.quantile(0.80))
    df["forecast_bust"] = (df["_forecast_error"] > threshold).astype(int)

    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    df.to_csv(os.path.join(BASE, "data", "training_dataset.csv"), index=False)

    with open(os.path.join(BASE, "data", "regions.json"), "w") as f:
        json.dump(REGIONS, f, indent=2)

    print(f"Generated {len(df)} rows | bust rate: {df['forecast_bust'].mean():.3f}")
    print(df.head())


if __name__ == "__main__":
    main()
