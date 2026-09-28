"""
Precomputes district-level predictions (bust probability, confidence, top
contributors) for every district x Day 1-10 cell, plus a state-level
aggregation (mean over that state's districts) so state rollups are derived
from real district predictions rather than hardcoded independently.
"""
import json
import os
import numpy as np
import pandas as pd
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = [
    "precipitation", "temperature", "pressure", "humidity", "wind_speed",
    "wind_direction", "lead_time", "latitude", "longitude",
    "spatial_gradient", "historical_error", "pressure_variation",
    "precipitation_variability",
]
FEATURE_LABELS = {
    "precipitation": "Forecast Precipitation",
    "temperature": "Forecast Temperature",
    "pressure": "Surface Pressure",
    "humidity": "Humidity",
    "wind_speed": "Wind Speed",
    "wind_direction": "Wind Direction",
    "lead_time": "Increasing Lead Time",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "spatial_gradient": "Strong Spatial Gradient",
    "historical_error": "Historical Error Pattern",
    "pressure_variation": "Pressure Variation",
    "precipitation_variability": "Precipitation Variability",
}

model = joblib.load(f"{BASE}/model/district_forecast_model.joblib")
df = pd.read_csv(f"{BASE}/data/district_training_dataset.csv")
with open(f"{BASE}/data/districts.json") as f:
    districts = json.load(f)
with open(f"{BASE}/model/district_feature_importance.json") as f:
    feat_imp = json.load(f)
feat_imp_map = {d["feature"]: d["importance"] for d in feat_imp}
global_means = df[FEATURES].mean()
global_stds = df[FEATURES].std().replace(0, 1)


def confidence_from_bust(p):
    conf = 1 - p
    if p <= 0.35:
        level = "HIGH"
    elif p <= 0.65:
        level = "MODERATE"
    else:
        level = "LOW"
    return round(conf * 100, 1), level


def top_contributors(row, n=5):
    scores = {}
    for f in FEATURES:
        z = abs((row[f] - global_means[f]) / global_stds[f])
        scores[f] = feat_imp_map.get(f, 0) * (1 + z)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:n]
    return [FEATURE_LABELS.get(f, f) for f, _ in ranked]


cells = []
for district in districts:
    dcode = district["district_code"]
    sub_all = df[df["district_code"] == dcode]
    for day in range(1, 11):
        sub = sub_all[sub_all["lead_time"] == day]
        if sub.empty:
            continue
        rep = sub[FEATURES].mean()
        proba = float(model.predict_proba(pd.DataFrame([rep]))[0, 1])
        confidence, level = confidence_from_bust(proba)
        cells.append({
            "state": district["state"],
            "district": district["district"],
            "district_code": dcode,
            "lat": district["latitude"],
            "lon": district["longitude"],
            "day": day,
            "bust_probability": round(proba * 100, 1),
            "confidence": confidence,
            "risk_level": level,
            "error_prone": level == "LOW",
            "forecast_precipitation": round(float(rep["precipitation"]), 1),
            "forecast_temperature": round(float(rep["temperature"]), 1),
            "forecast_pressure": round(float(rep["pressure"]), 1),
            "forecast_humidity": round(float(rep["humidity"]), 1),
            "forecast_wind_speed": round(float(rep["wind_speed"]), 1),
            "historical_error": round(float(rep["historical_error"]), 2),
            "spatial_gradient": round(float(rep["spatial_gradient"]), 3),
            "top_contributors": top_contributors(rep),
            "data_type": "synthetic_prototype",
        })

with open(f"{BASE}/data/district_snapshot.json", "w") as f:
    json.dump({"cells": cells, "data_type": "synthetic_prototype"}, f, separators=(",", ":"))

# ---- State-level aggregation, DERIVED from district cells (not independent) ----
cells_df = pd.DataFrame(cells)
state_cells = []
for (state, day), grp in cells_df.groupby(["state", "day"]):
    state_cells.append({
        "state": state,
        "day": int(day),
        "bust_probability": round(float(grp["bust_probability"].mean()), 1),
        "confidence": round(float(grp["confidence"].mean()), 1),
        "risk_level": "LOW" if grp["confidence"].mean() <= 35 else ("MODERATE" if grp["confidence"].mean() <= 65 else "HIGH"),
        "district_count": int(len(grp)),
        "highest_risk_district": grp.loc[grp["bust_probability"].idxmax(), "district"],
        "highest_risk_district_bust": round(float(grp["bust_probability"].max()), 1),
        "lat": round(float(grp["lat"].mean()), 4),
        "lon": round(float(grp["lon"].mean()), 4),
        "data_type": "synthetic_prototype",
    })

with open(f"{BASE}/data/state_aggregation.json", "w") as f:
    json.dump({"cells": state_cells, "data_type": "synthetic_prototype",
               "note": "State values are the mean of that state's district-level predictions."}, f, indent=2)

print(f"District snapshot: {len(cells)} cells across {len(districts)} districts")
print(f"State aggregation: {len(state_cells)} state-day cells across {cells_df['state'].nunique()} states")
print("Sample district cell:", cells[0])
print("Sample state aggregation:", state_cells[0])
