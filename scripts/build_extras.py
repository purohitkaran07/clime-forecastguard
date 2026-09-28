import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(f"{BASE}/data/district_snapshot.json") as f:
    snap = json.load(f)
cells = snap["cells"]

# ---- KPIs (derived from real district-level snapshot data) ----
avg_confidence = round(sum(c["confidence"] for c in cells) / len(cells), 1)
high_risk = [c for c in cells if c["risk_level"] == "LOW"]
worst = max(cells, key=lambda c: c["bust_probability"])
by_day_avg_bust = {}
for c in cells:
    by_day_avg_bust.setdefault(c["day"], []).append(c["bust_probability"])
worst_day = max(by_day_avg_bust, key=lambda d: sum(by_day_avg_bust[d]) / len(by_day_avg_bust[d]))

n_districts = len(set(c["district_code"] for c in cells))
n_states = len(set(c["state"] for c in cells))

kpis = {
    "average_confidence": avg_confidence,
    "high_risk_regions_count": len(set(c["district_code"] for c in high_risk)),
    "highest_bust_probability": worst["bust_probability"],
    "highest_bust_region": f"{worst['district']}, {worst['state']}",
    "highest_bust_day": worst["day"],
    "most_uncertain_lead_time": f"Day {worst_day}",
    "regions_monitored": n_districts,
    "regions_monitored_label": f"{n_districts} districts across {n_states} states",
    "data_type": "synthetic_prototype",
}
with open(f"{BASE}/data/kpis.json", "w") as f:
    json.dump(kpis, f, indent=2)

# ---- Alerts: top risk cells, deduped by district (keep worst day per district), top 8 ----
best_per_district = {}
for c in cells:
    k = c["district_code"]
    if k not in best_per_district or c["bust_probability"] > best_per_district[k]["bust_probability"]:
        best_per_district[k] = c
alerts = sorted(best_per_district.values(), key=lambda c: -c["bust_probability"])[:8]
alerts_out = []
for a in alerts:
    level = "HIGH RISK" if a["bust_probability"] >= 65 else ("MODERATE RISK" if a["bust_probability"] >= 40 else "LOW RISK")
    alerts_out.append({
        "state": a["state"],
        "district": a["district"],
        "region": f"{a['district']}, {a['state']}",
        "day": a["day"],
        "bust_probability": a["bust_probability"],
        "level": level,
    })
with open(f"{BASE}/data/alerts.json", "w") as f:
    json.dump(alerts_out, f, indent=2)

# ---- Historical Event Replay (clearly labelled synthetic demonstration events) ----
events = [
    {
        "id": "heavy_rainfall", "name": "Heavy Rainfall Event", "region": "Jodhpur, Rajasthan",
        "lead_time": 4, "forecast_value": 48, "reference_value": 91, "unit": "mm",
        "forecast_error": 43, "bust_probability": 81, "result": "HIGH-RISK REGION IDENTIFIED",
        "note": "Demonstration Event — Synthetic Data",
    },
    {
        "id": "monsoon_depression", "name": "Monsoon Depression", "region": "Puri, Odisha",
        "lead_time": 5, "forecast_value": 62, "reference_value": 118, "unit": "mm",
        "forecast_error": 56, "bust_probability": 84, "result": "HIGH-RISK REGION IDENTIFIED",
        "note": "Demonstration Event — Synthetic Data",
    },
    {
        "id": "western_disturbance", "name": "Western Disturbance", "region": "Amritsar, Punjab",
        "lead_time": 3, "forecast_value": 14, "reference_value": 27, "unit": "mm",
        "forecast_error": 13, "bust_probability": 58, "result": "MODERATE-RISK REGION IDENTIFIED",
        "note": "Demonstration Event — Synthetic Data",
    },
    {
        "id": "heat_wave", "name": "Heat Wave", "region": "New Delhi, Delhi",
        "lead_time": 6, "forecast_value": 39.5, "reference_value": 44.8, "unit": "°C",
        "forecast_error": 5.3, "bust_probability": 66, "result": "HIGH-RISK REGION IDENTIFIED",
        "note": "Demonstration Event — Synthetic Data",
    },
    {
        "id": "cyclonic_system", "name": "Cyclonic System", "region": "South 24 Parganas, West Bengal",
        "lead_time": 7, "forecast_value": 70, "reference_value": 142, "unit": "mm",
        "forecast_error": 72, "bust_probability": 89, "result": "HIGH-RISK REGION IDENTIFIED",
        "note": "Demonstration Event — Synthetic Data",
    },
]
with open(f"{BASE}/data/events.json", "w") as f:
    json.dump(events, f, indent=2)

print("KPIs:", kpis)
print("Alerts:", alerts_out)
print("Events:", len(events))
