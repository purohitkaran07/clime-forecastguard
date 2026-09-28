"""
Precomputes a demo snapshot so the deployed API can respond instantly
without retraining or heavy computation per request (Vercel cold-start safe).
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

model = joblib.load(f"{BASE}/model/forecast_model.joblib")
df = pd.read_csv(f"{BASE}/data/training_dataset.csv")
with open(f"{BASE}/data/regions.json") as f:
    regions = json.load(f)
with open(f"{BASE}/model/feature_importance.json") as f:
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
    """Lightweight, honest approximation of feature contribution when SHAP
    is not used: combines each feature's global importance with how far this
    row's value deviates from the dataset average (z-score). This is NOT a
    causal explanation - labelled as such in the UI."""
    scores = {}
    for f in FEATURES:
        z = abs((row[f] - global_means[f]) / global_stds[f])
        scores[f] = feat_imp_map.get(f, 0) * (1 + z)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:n]
    return [FEATURE_LABELS.get(f, f) for f, _ in ranked]


# Representative row per region/day = mean of that cell's samples
snapshot = {"regions": [], "generated_from": "synthetic demonstration dataset"}
cells = {}
for region in regions:
    rname = region["name"]
    for day in range(1, 11):
        sub = df[(df["region"] == rname) & (df["lead_time"] == day)]
        rep = sub[FEATURES].mean()
        proba = float(model.predict_proba(pd.DataFrame([rep]))[0, 1])
        confidence, level = confidence_from_bust(proba)
        contributors = top_contributors(rep)
        cells[(rname, day)] = {
            "region": rname,
            "lat": region["lat"],
            "lon": region["lon"],
            "day": day,
            "bust_probability": round(proba * 100, 1),
            "confidence": confidence,
            "risk_level": level,
            "forecast_precipitation": round(float(rep["precipitation"]), 1),
            "forecast_temperature": round(float(rep["temperature"]), 1),
            "forecast_pressure": round(float(rep["pressure"]), 1),
            "forecast_humidity": round(float(rep["humidity"]), 1),
            "forecast_wind_speed": round(float(rep["wind_speed"]), 1),
            "historical_error": round(float(rep["historical_error"]), 2),
            "spatial_gradient": round(float(rep["spatial_gradient"]), 3),
            "top_contributors": contributors,
        }

snapshot["cells"] = [cells[k] for k in cells]

with open(f"{BASE}/data/demo_snapshot.json", "w") as f:
    json.dump(snapshot, f, indent=2)

print(f"Snapshot built: {len(snapshot['cells'])} region-day cells")
print("Sample:", snapshot["cells"][3])
