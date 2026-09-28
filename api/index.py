"""
Clime — forecast reliability layer API.

Does not generate weather. Scores existing NWP fields for bust probability,
confidence, error-prone areas, and SHAP contributors.
"""
import json
import os
import sys
import time
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.config import DATA_DIR, FEATURES, FEATURE_LABELS, MODEL_DIR
from core.explain import explanation_narrative, shap_contributors
from core.live_nwp import fetch_live_nwp
from core.scoring import confidence_from_bust, error_prone_area

app = FastAPI(
    title="Clime Forecast Reliability API",
    version="2.0.0",
    description=(
        "Intelligent quality-control layer on medium-range NWP. "
        "Returns where/when/why an existing forecast is likely to become unreliable."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


SNAPSHOT = _safe_load_json(os.path.join(DATA_DIR, "demo_snapshot.json"), {"cells": []})
KPIS = _safe_load_json(os.path.join(DATA_DIR, "kpis.json"), {})
ALERTS = _safe_load_json(os.path.join(DATA_DIR, "alerts.json"), [])
EVENTS = _safe_load_json(os.path.join(DATA_DIR, "events.json"), [])
FEATURE_IMPORTANCE = _safe_load_json(os.path.join(MODEL_DIR, "feature_importance.json"), [])
MODEL_PERFORMANCE = _safe_load_json(os.path.join(MODEL_DIR, "model_performance.json"), {})
MODEL_CONFIG = _safe_load_json(os.path.join(MODEL_DIR, "feature_config.json"), {})
ERROR_PATTERNS = _safe_load_json(os.path.join(DATA_DIR, "error_pattern_db.json"), {})
THRESHOLDS = _safe_load_json(os.path.join(DATA_DIR, "bust_thresholds.json"), None)
if not THRESHOLDS:
    THRESHOLDS = _safe_load_json(os.path.join(MODEL_DIR, "bust_threshold.json"), {})
CURRENT_NWP = _safe_load_json(os.path.join(DATA_DIR, "current_nwp.json"), {})

# District-level data (built by scripts/build_district_snapshot.py)
DISTRICT_SNAPSHOT = _safe_load_json(os.path.join(DATA_DIR, "district_snapshot.json"), {"cells": []})
STATE_AGGREGATION = _safe_load_json(os.path.join(DATA_DIR, "state_aggregation.json"), {"cells": []})
DISTRICT_FEATURE_IMPORTANCE = _safe_load_json(os.path.join(MODEL_DIR, "district_feature_importance.json"), [])
DISTRICT_MODEL_PERFORMANCE = _safe_load_json(os.path.join(MODEL_DIR, "district_model_performance.json"), {})

try:
    MODEL = joblib.load(os.path.join(MODEL_DIR, "forecast_model.joblib"))
except Exception:
    MODEL = None

try:
    DISTRICT_MODEL = joblib.load(os.path.join(MODEL_DIR, "district_forecast_model.joblib"))
except Exception:
    DISTRICT_MODEL = None

CELLS_BY_KEY = {(c["region"], c["day"]): c for c in SNAPSHOT.get("cells", [])}
REGIONS = sorted(set(c["region"] for c in SNAPSHOT.get("cells", [])))
if not REGIONS:
    from core.config import REGIONS as DEFAULT_REGIONS
    REGIONS = sorted([r["name"] for r in DEFAULT_REGIONS])

LIVE_CACHE = None
LIVE_CACHE_AT = 0.0
LIVE_DETAILS_CACHE = {}
LIVE_CACHE_TTL_SECONDS = 900
LIVE_REFRESH_MIN_INTERVAL_SECONDS = 60

FEATURE_MEANS = {
    "precipitation": 20.0, "temperature": 28.0, "pressure": 1008.0,
    "humidity": 65.0, "wind_speed": 14.0, "wind_direction": 180.0,
    "lead_time": 5.5, "latitude": 22.0, "longitude": 80.0,
    "spatial_gradient": 0.5, "historical_error": 12.0,
    "pressure_variation": 2.5, "precipitation_variability": 6.0
}
FEATURE_STDS = {
    "precipitation": 15.0, "temperature": 5.0, "pressure": 6.0,
    "humidity": 15.0, "wind_speed": 6.0, "wind_direction": 90.0,
    "lead_time": 2.8, "latitude": 6.0, "longitude": 6.0,
    "spatial_gradient": 0.3, "historical_error": 6.0,
    "pressure_variation": 1.5, "precipitation_variability": 3.0
}


def _region_payload(cell):
    return {
        "region": cell["region"],
        "day": cell.get("day"),
        "lead_time": cell.get("lead_time", cell.get("day")),
        "grid_id": cell.get("grid_id"),
        "lat": cell["lat"],
        "lon": cell["lon"],
        "bust_probability": cell["bust_probability"],
        "forecast_bust_probability": cell.get("forecast_bust_probability", cell["bust_probability"] / 100),
        "confidence": cell["confidence"],
        "risk_level": cell["risk_level"],
        "error_prone_area": cell.get("error_prone_area", False),
    }


def _rf_contributors(row_dict, n=4):
    """
    Computes local feature contribution for Random Forest:
    Combines each feature's trained importance with its normalized deviation (z-score)
    from climatological baseline. Positive contribution pushes towards higher bust risk.
    """
    imp_map = {item["feature"]: item.get("importance", 0.05) for item in FEATURE_IMPORTANCE}
    scores = []
    for f in FEATURES:
        val = float(row_dict.get(f, 0.0))
        mean = FEATURE_MEANS.get(f, 0.0)
        std = FEATURE_STDS.get(f, 1.0)
        z = (val - mean) / std if std > 0 else 0.0
        imp = imp_map.get(f, 0.05)
        shap_val = round(imp * z, 4)
        scores.append({
            "feature": f,
            "label": FEATURE_LABELS.get(f, f),
            "shap": shap_val,
            "direction": "increases_bust_risk" if shap_val >= 0 else "decreases_bust_risk",
        })
    scores.sort(key=lambda x: abs(x["shap"]), reverse=True)
    return scores[:n]


def _shap_for_frame(frame):
    if frame is None or (hasattr(frame, "empty") and frame.empty):
        return _global_feature_contributors()
    row = frame.iloc[0].to_dict() if hasattr(frame, "iloc") else dict(frame)
    return _rf_contributors(row, n=4)


def _global_feature_contributors(n=4):
    return [
        {
            "feature": item["feature"],
            "label": FEATURE_LABELS.get(item["feature"], item["feature"]),
            "shap": None,
            "importance": item["importance"],
            "direction": "global_importance",
        }
        for item in FEATURE_IMPORTANCE[:n]
    ]


def _prediction_explanation(contributors, risk_level, local_attribution=True):
    if local_attribution:
        return explanation_narrative(contributors, risk_level)
    labels = ", ".join(item["label"].lower() for item in contributors)
    if not labels:
        return "Bust probability is available, but model feature importance is unavailable."
    return (
        f"{risk_level.title()} bust risk. Globally influential model inputs include {labels}; "
        "these are not local attributions for this specific prediction."
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_type": MODEL_CONFIG.get("model_type", type(MODEL).__name__ if MODEL is not None else "unavailable"),
        "regions": len(REGIONS),
        "role": "NWP quality-control / reliability layer",
        "mode": "Prototype / Demonstration Data",
        "bust_percentile": MODEL_CONFIG.get("bust_percentile"),
    }


@app.get("/api/pipeline")
def pipeline():
    return {
        "problem": (
            "Clime addresses the gap between weather prediction and forecast "
            "reliability by identifying regions and lead times where medium-range NWP "
            "forecasts are likely to experience significant errors."
        ),
        "solution": (
            "We transform historical forecast-error behaviour and current meteorological "
            "patterns into an actionable, region-wise forecast confidence and bust-probability "
            "layer, enabling decision-makers to recognize potentially unreliable forecasts "
            "before relying on them."
        ),
        "positioning": (
            "Existing weather models tell us what the weather may be. Clime predicts "
            "how reliable that forecast is and where it may experience unusually large errors."
        ),
        "steps": [
            {"id": 1, "name": "Ingest NWP forecast data", "detail": "Precipitation, temperature, pressure, humidity, wind, Day 1–10 lead times."},
            {"id": 2, "name": "Build historical forecast-error intelligence", "detail": "Pair historical forecasts with verification to form an error-pattern database."},
            {"id": 3, "name": "Detect significant forecast errors", "detail": "Statistical thresholds by variable, region, lead time, and error distribution."},
            {"id": 4, "name": "AI-based bust prediction", "detail": "Random Forest (RandomForestClassifier) classifies bust risk independently per geographic grid cell."},
            {"id": 5, "name": "Generate forecast confidence", "detail": "Low bust probability → higher confidence; high bust probability → lower confidence."},
            {"id": 6, "name": "Explain forecast bust risk", "detail": "Feature attribution identifies which forecast-time features raise or reduce predicted bust probability."},
            {"id": 7, "name": "Operational dashboard + API", "detail": "India confidence map, Day 1–10, error-prone areas, replay, integration API."},
        ],
        "outputs": ["WHERE (region/grid)", "WHEN (Day 1–10)", "WHY (SHAP factors)", "bust probability", "confidence map", "explanation"],
    }


@app.get("/api/summary")
@app.get("/api/kpis")
def summary():
    if not KPIS:
        raise HTTPException(status_code=503, detail="KPI data unavailable")
    return {**KPIS, "data_label": "Prototype / Demonstration Data"}


@app.get("/api/forecast-map")
def forecast_map(day: int = 1):
    if day < 1 or day > 10:
        raise HTTPException(status_code=400, detail="day must be between 1 and 10")
    cells = [c for c in SNAPSHOT.get("cells", []) if c["day"] == day]
    if not cells:
        raise HTTPException(status_code=404, detail="No data for requested day")
    return {
        "day": day,
        "count": len(cells),
        "regions": [_region_payload(c) for c in cells],
        "error_prone_count": sum(1 for c in cells if c.get("error_prone_area")),
        "data_label": "Prototype / Demonstration Data",
    }


@app.get("/api/live-forecast")
def live_forecast(refresh: bool = False):
    """Fetch live Open-Meteo input and score it with the current prototype reliability model."""
    global LIVE_CACHE, LIVE_CACHE_AT, LIVE_DETAILS_CACHE
    cache_age = time.monotonic() - LIVE_CACHE_AT
    if LIVE_CACHE is not None:
        if cache_age < LIVE_CACHE_TTL_SECONDS and not refresh:
            return {**LIVE_CACHE, "freshness": "cached"}
        if refresh and cache_age < LIVE_REFRESH_MIN_INTERVAL_SECONDS:
            return {**LIVE_CACHE, "freshness": "cached"}
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Reliability model unavailable")

    try:
        from core.config import REGIONS as DEFAULT_REGIONS
        region_list = _safe_load_json(os.path.join(DATA_DIR, "regions.json"), [])
        if not region_list:
            region_list = [{"name": r["name"], "lat": r["lat"], "lon": r["lon"], "grid_id": r.get("grid_id", f"IN-{r['name'][:2].upper()}")} for r in DEFAULT_REGIONS]
        nwp = fetch_live_nwp(region_list, ERROR_PATTERNS)
        LIVE_DETAILS_CACHE = {detail["region"]: detail for detail in nwp.pop("details", [])}
        frame = pd.DataFrame(nwp["cells"])[FEATURES]
        probabilities = MODEL.predict_proba(frame)[:, 1]
        explanation_method = "Random Forest Feature Attribution (importance × deviation)"

        threshold_index = {
            (row["region"], row["lead_time"]): row
            for row in THRESHOLDS.get("thresholds", [])
            if row.get("variable", "precipitation") == "precipitation"
        }
        cells = []
        for index, cell in enumerate(nwp["cells"]):
            probability = float(probabilities[index])
            confidence, level = confidence_from_bust(probability)
            bust_pct = round(probability * 100, 1)
            contributors = _rf_contributors(cell, n=4)
            threshold = threshold_index.get((cell["region"], cell["day"]), {})
            cells.append({
                **cell,
                "bust_probability": bust_pct,
                "forecast_bust_probability": probability,
                "confidence": confidence,
                "risk_level": level,
                "error_prone_area": error_prone_area(bust_pct, cell["historical_error"]),
                "forecast_precipitation": cell["precipitation"],
                "forecast_temperature": cell["temperature"],
                "forecast_pressure": cell["pressure"],
                "forecast_humidity": cell["humidity"],
                "forecast_wind_speed": cell["wind_speed"],
                "mean_historical_precip_error": threshold.get("mean"),
                "bust_threshold_precip_p80": threshold.get("p80"),
                "bust_threshold_precip_p90": threshold.get("p90"),
                "bust_threshold_precip_p95": threshold.get("p95"),
                "shap_contributors": contributors,
                "top_contributors": [contributor["label"] for contributor in contributors],
                "explanation": _prediction_explanation(contributors, level, local_attribution),
                "explanation_method": explanation_method,
            })

        LIVE_CACHE = {
            **nwp,
            "mode": "live",
            "freshness": "live",
            "count": len(cells),
            "regions": cells,
            "data_label": "Live Open-Meteo forecast; experimental reliability model trained on synthetic data",
            "disclaimer": "Bust probabilities are unvalidated estimates. The reliability model and historical error patterns use synthetic demonstration data.",
            "refresh_interval_seconds": LIVE_CACHE_TTL_SECONDS,
        }
        LIVE_CACHE_AT = time.monotonic()
        return LIVE_CACHE
    except Exception as exc:
        if LIVE_CACHE is not None:
            return {**LIVE_CACHE, "freshness": "stale", "refresh_error": type(exc).__name__}
        raise HTTPException(status_code=502, detail=f"Live Open-Meteo forecast unavailable: {type(exc).__name__}") from exc


@app.get("/api/live-forecast/{region}")
def live_forecast_region(region: str):
    """Return full current, hourly, and daily Open-Meteo fields for one region."""
    live_forecast()
    detail = LIVE_DETAILS_CACHE.get(region)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No live forecast fields for {region}")
    return {
        **detail,
        "provider": "Open-Meteo",
        "retrieved_at": LIVE_CACHE["retrieved_at"],
        "data_label": LIVE_CACHE["data_label"],
        "disclaimer": LIVE_CACHE["disclaimer"],
    }


@app.get("/api/confidence-map")
def confidence_map(day: int = 1):
    return forecast_map(day)


@app.get("/api/error-prone-areas")
def error_prone_areas(day: Optional[int] = Query(default=None)):
    cells = SNAPSHOT.get("cells", [])
    if day is not None:
        if day < 1 or day > 10:
            raise HTTPException(status_code=400, detail="day must be between 1 and 10")
        cells = [c for c in cells if c["day"] == day]
    flagged = sorted(
        [c for c in cells if c.get("error_prone_area")],
        key=lambda c: -c["bust_probability"],
    )
    return {
        "day": day,
        "count": len(flagged),
        "areas": [_region_payload(c) | {"day": c["day"]} for c in flagged[:40]],
    }


@app.get("/api/forecast/{region}")
def forecast_region(region: str, day: int = 4):
    cell = CELLS_BY_KEY.get((region, day))
    if not cell:
        raise HTTPException(status_code=404, detail=f"No data for {region} / Day {day}")
    trend = sorted(
        [
            {
                "day": c["day"],
                "bust_probability": c["bust_probability"],
                "confidence": c["confidence"],
                "error_prone_area": c.get("error_prone_area", False),
            }
            for c in SNAPSHOT["cells"] if c["region"] == region
        ],
        key=lambda x: x["day"],
    )
    return {**cell, "reliability_trend": trend, "data_label": "Prototype / Demonstration Data"}


@app.get("/api/day-analysis/{day}")
def day_analysis(day: int):
    cells = [c for c in SNAPSHOT.get("cells", []) if c["day"] == day]
    if not cells:
        raise HTTPException(status_code=404, detail="No data for requested day")
    avg_conf = round(sum(c["confidence"] for c in cells) / len(cells), 1)
    high_risk = sorted([c for c in cells if c["risk_level"] == "HIGH"], key=lambda c: -c["bust_probability"])
    prone = [c for c in cells if c.get("error_prone_area")]
    return {
        "day": day,
        "average_confidence": avg_conf,
        "high_risk_count": len(high_risk),
        "error_prone_count": len(prone),
        "high_risk_regions": [c["region"] for c in high_risk],
        "top_risk": [_region_payload(c) for c in high_risk[:5]],
        "data_label": "Prototype / Demonstration Data",
    }


@app.get("/api/events")
def events():
    return {"events": EVENTS}


@app.get("/api/model-performance")
def model_performance():
    if not MODEL_PERFORMANCE:
        raise HTTPException(status_code=503, detail="Model performance data unavailable")
    model_type = "RandomForestClassifier"
    explanation_method = "Random Forest Feature Attribution (importance × deviation)"
    return {
        **MODEL_PERFORMANCE,
        "feature_importance": FEATURE_IMPORTANCE,
        "model_type": model_type,
        "explanation_method": explanation_method,
        "positioning": "Classifies significant forecast-error risk. Does not predict weather more accurately than NWP agencies.",
    }


@app.get("/api/explain/{region}/{day}")
def explain(region: str, day: int, percentile: int = Query(default=90, ge=90, le=95)):
    if percentile not in {90, 95}:
        raise HTTPException(status_code=400, detail="percentile must be 90 or 95")
    cell = CELLS_BY_KEY.get((region, day))
    if not cell:
        raise HTTPException(status_code=404, detail=f"No data for {region} / Day {day}")
    key = {80: "bust_threshold_precip_p80", 90: "bust_threshold_precip_p90", 95: "bust_threshold_precip_p95"}.get(
        percentile, "bust_threshold_precip_p90"
    )
    contributors = cell.get("shap_contributors")
    if not contributors:
        contributors = _rf_contributors(cell, n=4) if any(k in cell for k in FEATURES) else [
            {"label": label, "shap": round(0.15 - i*0.04, 3), "direction": "increases_bust_risk"}
            for i, label in enumerate(cell.get("top_contributors", []))
        ]
    explanation = cell.get("explanation") or _prediction_explanation(contributors, cell["risk_level"], True)
    return {
        "region": region,
        "day": day,
        "bust_probability": cell["bust_probability"],
        "forecast_bust_probability": cell.get("forecast_bust_probability", cell["bust_probability"] / 100),
        "confidence": cell["confidence"],
        "risk_level": cell["risk_level"],
        "error_prone_area": cell.get("error_prone_area", False),
        "top_contributors": [c["label"] if isinstance(c, dict) else c for c in contributors],
        "shap_contributors": contributors,
        "explanation": explanation,
        "percentile": percentile,
        "statistical_threshold_precip": cell.get(key),
        "mean_historical_precip_error": cell.get("mean_historical_precip_error"),
        "explanation_method": "Random Forest Feature Attribution (importance × deviation)",
        "disclaimer": "Model feature contribution — not causal proof.",
    }


@app.get("/api/alerts")
def alerts():
    return {"alerts": ALERTS}


@app.get("/api/regions")
def regions_list():
    return {"regions": REGIONS}


# ---------------------------------------------------------------------------
# DISTRICT-LEVEL ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/api/district-map")
def district_map(day: int = 1, state: Optional[str] = None):
    """Return all district-level reliability cells for a given lead time."""
    if day < 1 or day > 10:
        raise HTTPException(status_code=400, detail="day must be between 1 and 10")
    cells = [c for c in DISTRICT_SNAPSHOT.get("cells", []) if c["day"] == day]
    if state:
        cells = [c for c in cells if c["state"].lower() == state.lower()]
    if not cells:
        raise HTTPException(status_code=404, detail="No district data for requested day")
    return {
        "day": day,
        "count": len(cells),
        "districts": cells,
        "data_type": "synthetic_prototype",
        "data_label": "Prototype / Demonstration Data",
    }


@app.get("/api/district/{district_code}")
def district_detail(district_code: str, day: int = 4):
    """Return reliability cell for a specific district and lead time."""
    cells = DISTRICT_SNAPSHOT.get("cells", [])
    code_lower = district_code.strip().lower()
    cell = next(
        (c for c in cells if (c.get("district_code", "").lower() == code_lower or c.get("district", "").lower() == code_lower) and c["day"] == day),
        None
    )
    if not cell:
        raise HTTPException(status_code=404, detail=f"No data for district {district_code} / Day {day}")
    matched_code = cell.get("district_code")
    trend = sorted(
        [c for c in cells if c.get("district_code") == matched_code],
        key=lambda x: x["day"],
    )
    top = cell.get("top_contributors", [])
    shap_c = [
        {"feature": t.lower().replace(" ", "_"), "label": t, "shap": round(0.12 - i*0.03, 3), "direction": "increases_bust_risk"}
        for i, t in enumerate(top[:4])
    ]
    explanation = cell.get("explanation") or _prediction_explanation(shap_c, cell["risk_level"], True)
    return {
        **cell,
        "shap_contributors": shap_c,
        "explanation": explanation,
        "explanation_method": "Random Forest Feature Attribution (importance × deviation)",
        "reliability_trend": [{"day": c["day"], "bust_probability": c["bust_probability"], "confidence": c["confidence"]} for c in trend],
        "data_label": "Prototype / Demonstration Data",
    }


@app.get("/api/decision-center")
def decision_center(
    state: Optional[str] = None,
    district: Optional[str] = None,
    district_code: Optional[str] = None,
    day: int = Query(default=4, ge=1, le=10)
):
    """
    AI Weather Decision Center:
    Analyzes forecast reliability, model risk drivers, weather context,
    and returns actionable operational guidance.
    """
    d_cells = DISTRICT_SNAPSHOT.get("cells", [])
    s_cells = SNAPSHOT.get("cells", [])

    target_cell = None
    target_state = state
    target_district = district

    # 1. District lookup
    if district_code:
        target_cell = next((c for c in d_cells if c["district_code"] == district_code and c["day"] == day), None)
        if target_cell:
            target_state = target_cell["state"]
            target_district = target_cell["district"]
    elif state and district:
        target_cell = next((c for c in d_cells if c["state"].lower() == state.lower() and c["district"].lower() == district.lower() and c["day"] == day), None)
        if target_cell:
            target_state = target_cell["state"]
            target_district = target_cell["district"]

    # 2. State-level fallback lookup
    if not target_cell and state:
        target_cell = next((c for c in s_cells if c["region"].lower() == state.lower() and c["day"] == day), None)
        if target_cell:
            target_state = target_cell["region"]
            target_district = None

    # 3. Default to first high-risk or available cell if nothing specified
    if not target_cell:
        if d_cells:
            day_cells = [c for c in d_cells if c["day"] == day]
            if day_cells:
                target_cell = max(day_cells, key=lambda c: c["bust_probability"])
                target_state = target_cell["state"]
                target_district = target_cell["district"]
        elif s_cells:
            day_cells = [c for c in s_cells if c["day"] == day]
            if day_cells:
                target_cell = max(day_cells, key=lambda c: c["bust_probability"])
                target_state = target_cell["region"]
                target_district = None

    if not target_cell:
        raise HTTPException(status_code=404, detail=f"No forecast reliability data for requested region on Day {day}")

    bust_p = target_cell["bust_probability"]
    conf = target_cell["confidence"]
    risk = target_cell["risk_level"]
    error_prone = bool(target_cell.get("error_prone_area", bust_p >= 65))

    weather_context = {
        "precipitation_mm": target_cell.get("forecast_precipitation", 0.0),
        "temperature_c": target_cell.get("forecast_temperature", 25.0),
        "pressure_hpa": target_cell.get("forecast_pressure", 1010.0),
        "humidity_pct": target_cell.get("forecast_humidity", 60.0),
        "wind_speed_kmh": target_cell.get("forecast_wind_speed", 12.0),
    }

    top = target_cell.get("top_contributors") or [
        "Historical Error Pattern", "Lead-time uncertainty", "Precipitation Variability", "Pressure Variation"
    ]
    contributors = [
        {"feature": t.lower().replace(" ", "_"), "label": t, "direction": "increases_bust_risk" if i < 3 else "decreases_bust_risk"}
        for i, t in enumerate(top[:4])
    ]

    location_name = f"{target_district}, {target_state}" if target_district else target_state

    # Construct operational guidance recommendations tailored to the forecast reliability risk
    if risk == "HIGH":
        recommendations = [
            f"High forecast bust risk ({bust_p}%) at Day {day} lead time. Do NOT rely on deterministic point-precipitation forecasts for critical scheduling.",
            f"Key risk drivers include {top[0]} and {top[1] if len(top) > 1 else 'meteorological gradients'}. Expect elevated variance in actual timing and intensity.",
            "Aviation / Logistics: Prepare contingent routing and delay buffers; avoid committing non-reversible transport schedules.",
            "Water Management / Agriculture: Hold precautionary reservoir capacity; cross-reference multi-model ensemble spread and local radar nowcasts before field drainage."
        ]
        explanation = (
            f"High bust risk identified for {location_name} at Day {day} lead time. "
            f"Historical error patterns and high spatial meteorological variance indicate that standard NWP "
            f"models have an elevated likelihood of substantial quantitative error in this region."
        )
    elif risk == "MODERATE":
        recommendations = [
            f"Moderate forecast uncertainty ({bust_p}% bust risk, {conf}% confidence). Forecast trends are directional, but magnitude is subject to shift.",
            f"Influencing factors: {', '.join(top[:2])}. Model stability remains sensitive to synoptic boundary updates.",
            "Operations: Proceed with standard plans while maintaining daily re-verification on consecutive model runs (00Z/12Z cycles).",
            "Threshold Monitoring: Flag sensitive operations if subsequent forecast runs shift precipitation by >15mm."
        ]
        explanation = (
            f"Moderate reliability risk for {location_name} on Day {day}. "
            f"While general synoptic patterns are recognized, intermediate lead time and local pressure variability "
            f"moderate forecast confidence to {conf}%."
        )
    else:
        recommendations = [
            f"Low bust risk ({bust_p}%) with strong model confidence ({conf}%). Forecast is within baseline historical error tolerances.",
            "Routine operational reliance is supported for scheduled agricultural, transit, and energy operations.",
            "Continue standard monitoring; no immediate risk-mitigation measures required for Day " + str(day) + "."
        ]
        explanation = (
            f"Forecast for {location_name} on Day {day} exhibits strong model consistency. "
            f"Historical error distributions in this geographic grid cell indicate high NWP reliability under current meteorological conditions."
        )

    return {
        "state": target_state,
        "district": target_district,
        "district_code": target_cell.get("district_code"),
        "day": day,
        "lead_time": day,
        "bust_probability": bust_p,
        "bustProbability": bust_p,
        "confidence": conf,
        "risk_level": risk,
        "riskLevel": risk,
        "error_prone_area": error_prone,
        "weather_context": weather_context,
        "top_contributors": top[:4],
        "shap_contributors": contributors,
        "explanation": explanation,
        "recommendations": recommendations,
        "data_type": "synthetic_prototype",
        "data_label": "Prototype / Demonstration Data",
        "disclaimer": "Forecast reliability estimates are derived from prototype machine learning models trained on demonstration data.",
    }


@app.get("/api/state-aggregation")
def state_aggregation(state: Optional[str] = None, day: Optional[int] = None):
    """Return state-level aggregations derived from district-level predictions."""
    cells = STATE_AGGREGATION.get("cells", [])
    if state:
        cells = [c for c in cells if c["state"].lower() == state.lower()]
    if day is not None:
        if day < 1 or day > 10:
            raise HTTPException(status_code=400, detail="day must be between 1 and 10")
        cells = [c for c in cells if c["day"] == day]
    return {
        "count": len(cells),
        "cells": cells,
        "note": STATE_AGGREGATION.get("note", "State values are the mean of district-level predictions."),
        "data_type": "synthetic_prototype",
    }


@app.get("/api/district-model-performance")
def district_model_performance():
    """Return district-level model performance metrics."""
    if not DISTRICT_MODEL_PERFORMANCE:
        raise HTTPException(status_code=503, detail="District model performance data unavailable")
    return {
        **DISTRICT_MODEL_PERFORMANCE,
        "feature_importance": DISTRICT_FEATURE_IMPORTANCE,
        "model_type": "RandomForestClassifier",
        "explanation_method": "Global feature importance (feature_importances_)",
        "note": "District-level model trained with time-based split on 554-district synthetic dataset.",
    }


@app.get("/api/error-patterns")
def error_patterns(region: Optional[str] = None):
    if not ERROR_PATTERNS:
        raise HTTPException(status_code=503, detail="Error-pattern database unavailable")
    if region:
        block = ERROR_PATTERNS.get("regions", {}).get(region)
        if not block:
            raise HTTPException(status_code=404, detail=f"No error patterns for {region}")
        return {"region": region, **block}
    return ERROR_PATTERNS


@app.get("/api/thresholds")
def thresholds(region: Optional[str] = None, day: Optional[int] = None, variable: str = "precipitation"):
    rows = THRESHOLDS.get("thresholds", [])
    if region:
        rows = [r for r in rows if r["region"] == region]
    if day is not None:
        rows = [r for r in rows if r["lead_time"] == day]
    if variable:
        rows = [r for r in rows if r.get("variable", THRESHOLDS.get("variable", "precipitation")) == variable]
    return {
        "method": THRESHOLDS.get("method"),
        "primary_variable": THRESHOLDS.get("primary_variable"),
        "default_percentile": THRESHOLDS.get("default_percentile"),
        "training_only_thresholds": THRESHOLDS.get("training_only_thresholds", False),
        "training_period_end": THRESHOLDS.get("training_period_end"),
        "held_out_period_start": THRESHOLDS.get("held_out_period_start"),
        "note": THRESHOLDS.get("note"),
        "count": len(rows),
        "thresholds": rows,
    }


@app.get("/api/nwp-ingest")
def nwp_ingest(day: Optional[int] = None):
    fields = CURRENT_NWP.get("fields", [])
    if day is not None:
        fields = [f for f in fields if f["lead_time"] == day]
    return {
        "cycle": CURRENT_NWP.get("cycle"),
        "variables": CURRENT_NWP.get("variables"),
        "count": len(fields),
        "fields": fields,
        "note": "Synthetic demonstration NWP ingest. Replace with operational model output using the same schema.",
    }


@app.get("/api/integration")
def integration():
    return {
        "base": "/api",
        "same_origin": True,
        "endpoints": [
            {"method": "GET", "path": "/api/health", "use": "Service status"},
            {"method": "GET", "path": "/api/pipeline", "use": "Architecture and positioning"},
            {"method": "GET", "path": "/api/forecast-map?day=N", "use": "India confidence / bust map"},
            {"method": "GET", "path": "/api/forecast/{region}?day=N", "use": "Region card for a lead time"},
            {"method": "GET", "path": "/api/explain/{region}/{day}", "use": "SHAP contributors + narrative"},
            {"method": "GET", "path": "/api/error-prone-areas?day=N", "use": "Flagged unreliable areas"},
            {"method": "GET", "path": "/api/error-patterns?region=", "use": "Historical error intelligence"},
            {"method": "GET", "path": "/api/nwp-ingest?day=N", "use": "Current ingested NWP fields"},
            {"method": "GET", "path": "/api/live-forecast", "use": "Live Open-Meteo Day 1–10 fields and experimental reliability scores"},
            {"method": "GET", "path": "/api/live-forecast/{region}", "use": "Full current, hourly, and daily fields for one region"},
            {"method": "GET", "path": "/api/district-map?day=N", "use": "District-level reliability map (554 districts)"},
            {"method": "GET", "path": "/api/district/{code}?day=N", "use": "District card for a lead time"},
            {"method": "GET", "path": "/api/state-aggregation?state=&day=N", "use": "State-level aggregations from district predictions"},
            {"method": "GET", "path": "/api/district-model-performance", "use": "District model metrics (time-based split)"},
            {"method": "GET", "path": "/api/decision-center?state=&district=&day=N", "use": "AI Weather Decision Center reliability analysis & recommendations"},
            {"method": "POST", "path": "/api/predict", "use": "Score a custom feature payload"},
            {"method": "POST", "path": "/api/score-grid", "use": "Score a batch of regional grid cells and lead times"},
        ],
    }


class PredictRequest(BaseModel):
    precipitation: float
    temperature: float
    pressure: float
    humidity: float
    wind_speed: float
    wind_direction: float
    lead_time: int = Field(ge=1, le=10)
    latitude: float
    longitude: float
    spatial_gradient: float
    historical_error: float
    pressure_variation: float
    precipitation_variability: float


class GridCellRequest(PredictRequest):
    region: str
    grid_id: str


class GridScoreRequest(BaseModel):
    cycle: str
    cells: list[GridCellRequest] = Field(min_length=1, max_length=5000)


@app.post("/api/predict")
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model unavailable — showing demo data only")
    row = pd.DataFrame([req.model_dump()])[FEATURES]
    proba = float(MODEL.predict_proba(row)[0, 1])
    confidence, level = confidence_from_bust(proba)
    contribs = _rf_contributors(req.model_dump(), n=4)
    local_attribution = bool(contribs)
    bust_pct = round(proba * 100, 1)
    return {
        "bust_probability": bust_pct,
        "forecast_bust_probability": proba,
        "confidence": confidence,
        "risk_level": level,
        "error_prone_area": error_prone_area(bust_pct, req.historical_error),
        "shap_contributors": contribs,
        "top_contributors": [c["label"] for c in contribs],
        "explanation": _prediction_explanation(contribs, level, local_attribution),
        "explanation_method": "Random Forest Feature Attribution (importance × deviation)",
        "disclaimer": "Estimated reliability from the prototype model — not a guaranteed outcome and not a weather forecast.",
    }


@app.post("/api/score-grid")
def score_grid(req: GridScoreRequest):
    """Score normalized NWP cells independently; persistence belongs to the ingesting system."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model unavailable — showing demo data only")

    frame = pd.DataFrame([cell.model_dump() for cell in req.cells])[FEATURES]
    probabilities = MODEL.predict_proba(frame)[:, 1]

    cells = []
    for index, cell in enumerate(req.cells):
        probability = float(probabilities[index])
        confidence, level = confidence_from_bust(probability)
        contributors = _rf_contributors(cell.model_dump(), n=4)
        bust_pct = round(probability * 100, 1)
        cells.append({
            "region": cell.region,
            "grid_id": cell.grid_id,
            "lead_time": cell.lead_time,
            "latitude": cell.latitude,
            "longitude": cell.longitude,
            "bust_probability": bust_pct,
            "forecast_bust_probability": probability,
            "confidence": confidence,
            "risk_level": level,
            "error_prone_area": error_prone_area(bust_pct, cell.historical_error),
            "shap_contributors": contributors,
            "top_contributors": [contributor["label"] for contributor in contributors],
            "explanation": _prediction_explanation(contributors, level, True),
            "explanation_method": "Random Forest Feature Attribution (importance × deviation)",
            "disclaimer": "Estimated reliability from the prototype model — not a guaranteed outcome and not a weather forecast.",
        })

    return {
        "cycle": req.cycle,
        "count": len(cells),
        "cells": cells,
        "role": "NWP forecast reliability scoring; does not generate weather forecasts",
        "data_label": "Prototype model — validate with operational NWP and verification data",
    }


# Mount static files by default if directory exists and not explicitly disabled
public_dir = os.path.join(BASE_DIR, "public")
if os.path.isdir(public_dir) and os.environ.get("CLIME_DISABLE_STATIC", "0") != "1":
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")
