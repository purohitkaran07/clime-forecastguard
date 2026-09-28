"""Shared Clime configuration — NWP variables, features, regions."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "model")

FEATURES = [
    "precipitation",
    "temperature",
    "pressure",
    "humidity",
    "wind_speed",
    "wind_direction",
    "lead_time",
    "latitude",
    "longitude",
    "spatial_gradient",
    "historical_error",
    "pressure_variation",
    "precipitation_variability",
]

FEATURE_LABELS = {
    "precipitation": "Precipitation pattern",
    "temperature": "Temperature pattern",
    "pressure": "Pressure field",
    "humidity": "Humidity pattern",
    "wind_speed": "Wind speed",
    "wind_direction": "Wind direction",
    "lead_time": "Lead-time uncertainty",
    "latitude": "Latitude / geography",
    "longitude": "Longitude / geography",
    "spatial_gradient": "Spatial meteorological gradient",
    "historical_error": "Historical forecast error",
    "pressure_variation": "Pressure variation",
    "precipitation_variability": "Precipitation variability",
}

NWP_VARIABLES = [
    "precipitation",
    "temperature",
    "pressure",
    "humidity",
    "wind_speed",
]

# Primary verification variable for the bust label (standard precip-focused demo).
PRIMARY_ERROR_VARIABLE = "precipitation"

# Statistical bust threshold uses the regional × lead-time error distribution.
_BUST_PERCENTILE = int(os.environ.get(
    "CLIME_BUST_PERCENTILE",
    os.environ.get("FORECASTGUARD_BUST_PERCENTILE", "90"),
))
if _BUST_PERCENTILE not in {90, 95}:
    raise ValueError("CLIME_BUST_PERCENTILE must be 90 or 95")
DEFAULT_BUST_PERCENTILE = _BUST_PERCENTILE / 100

HIGH_BUST_RISK_MIN = float(os.environ.get("CLIME_HIGH_BUST_RISK_MIN", "0.65"))
MODERATE_BUST_RISK_MIN = float(os.environ.get("CLIME_MODERATE_BUST_RISK_MIN", "0.35"))
if not 0 <= MODERATE_BUST_RISK_MIN < HIGH_BUST_RISK_MIN <= 1:
    raise ValueError("Clime risk thresholds must satisfy 0 <= moderate < high <= 1")

REGIONS = [
    {"name": "Rajasthan", "lat": 27.0, "lon": 74.2, "base_precip": 15, "volatility": 1.4, "grid_id": "IN-RJ"},
    {"name": "Gujarat", "lat": 22.3, "lon": 71.2, "base_precip": 20, "volatility": 1.2, "grid_id": "IN-GJ"},
    {"name": "Maharashtra", "lat": 19.7, "lon": 75.7, "base_precip": 35, "volatility": 1.3, "grid_id": "IN-MH"},
    {"name": "Madhya Pradesh", "lat": 23.5, "lon": 78.6, "base_precip": 28, "volatility": 1.1, "grid_id": "IN-MP"},
    {"name": "Uttar Pradesh", "lat": 26.8, "lon": 80.9, "base_precip": 25, "volatility": 1.0, "grid_id": "IN-UP"},
    {"name": "Odisha", "lat": 20.9, "lon": 85.1, "base_precip": 45, "volatility": 1.5, "grid_id": "IN-OD"},
    {"name": "West Bengal", "lat": 22.9, "lon": 87.8, "base_precip": 48, "volatility": 1.4, "grid_id": "IN-WB"},
    {"name": "Assam", "lat": 26.2, "lon": 92.9, "base_precip": 55, "volatility": 1.3, "grid_id": "IN-AS"},
    {"name": "Kerala", "lat": 10.5, "lon": 76.2, "base_precip": 50, "volatility": 1.2, "grid_id": "IN-KL"},
    {"name": "Tamil Nadu", "lat": 11.1, "lon": 78.7, "base_precip": 30, "volatility": 1.2, "grid_id": "IN-TN"},
    {"name": "Karnataka", "lat": 15.3, "lon": 75.7, "base_precip": 32, "volatility": 1.1, "grid_id": "IN-KA"},
    {"name": "Delhi", "lat": 28.7, "lon": 77.1, "base_precip": 18, "volatility": 0.9, "grid_id": "IN-DL"},
    {"name": "Bihar", "lat": 25.9, "lon": 85.1, "base_precip": 33, "volatility": 1.2, "grid_id": "IN-BR"},
    {"name": "Punjab", "lat": 31.1, "lon": 75.3, "base_precip": 20, "volatility": 0.9, "grid_id": "IN-PB"},
    {"name": "Andhra Pradesh", "lat": 15.9, "lon": 79.7, "base_precip": 30, "volatility": 1.2, "grid_id": "IN-AP"},
    {"name": "Telangana", "lat": 18.1, "lon": 79.0, "base_precip": 28, "volatility": 1.1, "grid_id": "IN-TG"},
    {"name": "Jharkhand", "lat": 23.6, "lon": 85.3, "base_precip": 34, "volatility": 1.2, "grid_id": "IN-JH"},
    {"name": "Chhattisgarh", "lat": 21.3, "lon": 81.9, "base_precip": 36, "volatility": 1.2, "grid_id": "IN-CG"},
]
