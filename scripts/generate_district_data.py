"""
ForecastGuard AI - District-Level Synthetic Dataset Generator
------------------------------------------------------------------
Extends the state-level demo dataset to district level (554 real Indian
districts across the 18 states this prototype covers), using REAL district
centroids (from public 2011-census district boundaries) so the spatial
structure is genuine even though the weather values are synthetic.

data_type: "synthetic_prototype" everywhere. This is NOT real historical
observation data and must never be presented as such.

DETERMINISM / SPATIAL CORRELATION (per requirements):
  - Every value is a deterministic function of (latitude, longitude, lead_time,
    sample_index) - re-running this script produces byte-identical output.
  - Nearby districts share similar conditions because the underlying "weather
    field" is a smooth function of lat/lon (a small sum of sinusoids), not
    independent random draws per district. A per-district+sample noise term
    is layered on top (also seeded deterministically) to create enough
    within-district variation for the classifier to learn from.
  - Each of the 40 samples per district/day represents a distinct simulated
    forecast ISSUE DATE, so a genuine time-based train/test split is possible
    (see train_district_model.py).

DATA LEAKAGE SAFEGUARD (unchanged from the state-level version):
  The observed/reference value and the resulting forecast error are used
  ONLY to derive the training label (forecast_bust). They are never passed
  to the model as input features.
"""
import json
import math
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_PER_CELL = 40   # = 40 simulated forecast-issue dates per district/day
BASE_ISSUE_DATE = datetime(2024, 1, 1)

# Per-state baseline climate parameters (reused from the original state-level
# config) - districts inherit their state's general climate character, then
# get real-coordinate-driven spatial variation on top.
STATE_PARAMS = {
    "Rajasthan": {"base_precip": 15, "volatility": 1.4},
    "Gujarat": {"base_precip": 20, "volatility": 1.2},
    "Maharashtra": {"base_precip": 35, "volatility": 1.3},
    "Madhya Pradesh": {"base_precip": 28, "volatility": 1.1},
    "Uttar Pradesh": {"base_precip": 25, "volatility": 1.0},
    "Odisha": {"base_precip": 45, "volatility": 1.5},
    "West Bengal": {"base_precip": 48, "volatility": 1.4},
    "Assam": {"base_precip": 55, "volatility": 1.3},
    "Kerala": {"base_precip": 50, "volatility": 1.2},
    "Tamil Nadu": {"base_precip": 30, "volatility": 1.2},
    "Karnataka": {"base_precip": 32, "volatility": 1.1},
    "Delhi": {"base_precip": 18, "volatility": 0.9},
    "Bihar": {"base_precip": 33, "volatility": 1.2},
    "Punjab": {"base_precip": 20, "volatility": 0.9},
    "Andhra Pradesh": {"base_precip": 30, "volatility": 1.2},
    "Telangana": {"base_precip": 28, "volatility": 1.1},
    "Jharkhand": {"base_precip": 34, "volatility": 1.2},
    "Chhattisgarh": {"base_precip": 36, "volatility": 1.2},
}


def smooth_field(lat, lon, phase=0.0, freq=0.35):
    """Deterministic smooth pseudo-weather field over lat/lon so nearby
    districts (close lat/lon) get correlated values. Range approx [-1, 1]."""
    v = (
        math.sin(lat * freq + phase)
        + math.cos(lon * freq * 1.3 + phase * 0.7)
        + math.sin((lat + lon) * freq * 0.55 + phase * 1.9)
    ) / 3.0
    return v


def district_rng(district_code, day, sample_idx):
    """Deterministic per-(district, day, sample) RNG seed -> reproducible."""
    seed = abs(hash((district_code, day, sample_idx))) % (2**32)
    return np.random.default_rng(seed)


def gen_row(district, day, sample_idx, state_params):
    lat, lon = district["latitude"], district["longitude"]
    dcode = district["district_code"]
    rng = district_rng(dcode, day, sample_idx)

    lead_factor = 1 + 0.18 * (day - 1)
    base_precip = state_params["base_precip"]
    volatility = state_params["volatility"]

    precip_field = smooth_field(lat, lon, phase=0.0)       # spatial precip pattern
    temp_field = smooth_field(lat, lon, phase=2.1)          # spatial temp pattern
    error_field = smooth_field(lat, lon, phase=4.6)         # spatial error-proneness

    precipitation = max(0, base_precip * lead_factor * 0.35
                         + precip_field * 10 * volatility
                         + rng.normal(0, 8 * volatility))
    temperature = (30 - 0.25 * abs(lat - 15)) + temp_field * 3 - 0.15 * day + rng.normal(0, 2.5)
    pressure = 1008 + temp_field * 2 + rng.normal(0, 4)
    humidity = np.clip(60 + 0.5 * day + precip_field * 15 + rng.normal(0, 8), 10, 100)
    wind_speed = max(0, 11 + 0.4 * day + abs(temp_field) * 4 + rng.normal(0, 4))
    wind_direction = (lon * 4 + day * 7) % 360

    spatial_gradient = max(0, 0.35 * volatility * lead_factor + abs(precip_field) * 0.4 + rng.normal(0, 0.15))
    historical_error = max(0, 7 * volatility * lead_factor + abs(error_field) * 6 + rng.normal(0, 3))
    pressure_variation = max(0, 2.3 * lead_factor + abs(temp_field) * 1.2 + rng.normal(0, 1))
    precipitation_variability = max(0, 5.5 * volatility * lead_factor + abs(precip_field) * 5 + rng.normal(0, 2.5))

    error_std = 5.5 * volatility * lead_factor + abs(error_field) * 4
    observed_precip = max(0, precipitation + rng.normal(0, error_std) + spatial_gradient * 9)
    forecast_error = abs(observed_precip - precipitation)

    issue_date = BASE_ISSUE_DATE + timedelta(days=sample_idx)
    valid_date = issue_date + timedelta(days=day)

    return {
        "state": district["state"],
        "district": district["district"],
        "district_code": dcode,
        "lead_time": day,
        "precipitation": round(precipitation, 1),
        "temperature": round(temperature, 1),
        "pressure": round(pressure, 1),
        "humidity": round(humidity, 1),
        "wind_speed": round(wind_speed, 1),
        "wind_direction": round(wind_direction, 1),
        "latitude": lat,
        "longitude": lon,
        "spatial_gradient": round(spatial_gradient, 3),
        "historical_error": round(historical_error, 2),
        "pressure_variation": round(pressure_variation, 2),
        "precipitation_variability": round(precipitation_variability, 2),
        "forecast_issued_date": issue_date.strftime("%Y-%m-%d"),
        "valid_date": valid_date.strftime("%Y-%m-%d"),
        "_observed_precip": round(observed_precip, 1),
        "_forecast_error": round(forecast_error, 1),
        "data_type": "synthetic_prototype",
    }


def main():
    with open(f"{BASE}/data/districts.json") as f:
        districts = json.load(f)

    rows = []
    for district in districts:
        params = STATE_PARAMS.get(district["state"], {"base_precip": 28, "volatility": 1.2})
        for day in range(1, 11):
            for s in range(SAMPLES_PER_CELL):
                rows.append(gen_row(district, day, s, params))

    df = pd.DataFrame(rows)

    # Bust label: forecast error exceeds the 80th percentile of that
    # DISTRICT's own historical error distribution (spatially-local threshold).
    threshold = df.groupby("district_code")["_forecast_error"].transform(lambda x: x.quantile(0.80))
    df["forecast_bust"] = (df["_forecast_error"] > threshold).astype(int)

    df.to_csv(f"{BASE}/data/district_training_dataset.csv", index=False)
    print(f"Generated {len(df)} rows across {len(districts)} districts | bust rate: {df['forecast_bust'].mean():.3f}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
