# Clime / ForecastGuard

> **"Forecast reliability, not another weather forecast."**

Existing numerical weather models estimate what the weather may be. **Clime / ForecastGuard** estimates how reliable that forecast is, flagging where, when, and why an existing forecast may experience unusually large errors (forecast busts).

---

## What It Does

Numerical Weather Prediction (NWP) models (such as GFS, ECMWF, and regional models) provide vital medium-range atmospheric projections. However, spatial gradients, complex orography, and rapid atmospheric transitions frequently degrade medium-range forecast skill.

Clime sits as an intelligent reliability and quality-assurance layer on top of NWP forecasts:
* **Quantifies Uncertainty:** Computes a statistical **Bust Probability** (0–100%) and **Forecast Confidence** (100 − bust probability).
* **Identifies Risk Regimes:** Flags **LOW**, **MODERATE**, and **HIGH** forecast uncertainty across Day 1 to Day 10 lead times.
* **Pinpoints Root Causes:** Uses Random Forest feature attribution to identify which meteorological inputs (e.g. pressure gradients, precipitation variability, wind shears) drive forecast instability.
* **Guides Operational Decisions:** Provides actionable risk controls for agriculture, reservoir management, and grid logistics.

---

## Current Coverage

* **18 Monitored States / Meteorological Regions** covering key Indian climatic zones.
* **554 Districts** across India mapped to geographic boundaries (`public/india-districts.geojson`).
* **Day 1–10 Lead Time Horizons**.

*Note: The platform covers 18 states and 554 districts. It does not claim all-India 36-state coverage unless corresponding meteorological boundary pairs are configured.*

---

## Key Features

1. **National Reliability Dashboard**
   * Real-time metrics: Average Confidence, High-Risk Regions, Error-Prone Cells, and Most Uncertain Lead Time.
   * Interactive Day 1 to Day 10 forecast horizon slider.
   * Live Open-Meteo weather telemetry status indicator.

2. **Dual-Layer Interactive Map (Leaflet)**
   * **State Mode:** Macro-scale meteorological regions with confidence radius and risk-coded pins.
   * **District Mode:** High-resolution GeoJSON polygon choropleth rendering bust risk across 554 districts.
   * Filter controls (`ALL`, `HIGH`, `MODERATE`, `LOW`) and real-time text search.

3. **AI Weather Decision Center**
   * Operational console for sector decision-makers.
   * Deep dive by State, District, and Lead Time.
   * Provides bust probability, confidence, NWP weather context grid, model risk drivers, reliability narrative, and sector decision protocols (Agriculture, Water & Reservoirs, Logistics & Power).

4. **Regional & District Deep Dive**
   * 10-day reliability trajectories for every monitored state and district.
   * Terrain classification profiles (High-variance / Complex Terrain vs. Standard Terrain).
   * Feature attribution breakdowns.

5. **Explainable AI (XAI)**
   * Feature importance attribution computed via Random Forest feature weights combined with standardized meteorological deviations.
   * Contextual statistical error thresholds at the 75th, 90th, and 95th percentiles.

6. **Historical Event Replay**
   * Case studies (e.g. Cyclone Biparjoy, Mumbai High Rain Event, Western Disturbance) comparing forecast values vs. observed reference data to demonstrate how Clime flags forecast busts before they occur.

7. **Model Performance Analytics**
   * Evaluation charts rendered with native HTML5 Canvas: ROC Curves (AUC), Precision-Recall Curves (AP), Calibration Curves, and Confusion Matrices for both State and District models.

8. **Interactive Risk Simulator**
   * Parametric testing interface to score custom weather vectors (precipitation, temperature, surface pressure, humidity, wind, spatial gradient) in real time.

9. **Live NWP Telemetry**
   * Direct ingestion of current, hourly, and daily atmospheric variables from Open-Meteo's multi-location API across monitored coordinates.

---

## Machine Learning Architecture

Clime employs supervised ensemble classifiers trained on chronological forecast cycles:

* **State Reliability Model:** `RandomForestClassifier` (300 estimators) trained on macro-region meteorological predictors and historical error patterns.
* **District Reliability Model:** `RandomForestClassifier` (150 estimators) calibrated to district-level spatial features and terrain indicators.
* **Feature Attribution:** Implements localized Random Forest attribution combining feature importance weights with normalized parameter deviations from regional baselines.
* **Data Safeguard:** Models are trained strictly on variables known at forecast initialization time. Reference verification data is used solely offline to derive statistical bust thresholds.

---

## Data Provenance & Integrity

* **Live Weather:** Real-time atmospheric observations and medium-range forecast feeds ingested dynamically via Open-Meteo APIs.
* **District Boundaries:** Standard GeoJSON polygon geometries covering 554 Indian administrative districts.
* **Training / Calibration:** The bundled historical error patterns and training sets represent calibrated demonstration data structured according to operational NWP schemas. In production deployments, these can be replaced directly with operational IMD/NCMRWF/ECMWF forecast-analysis archives.

---

## Installation & Quickstart

### Prerequisites
* Python 3.10 to 3.14
* Git

### 1. Clone the Repository
```bash
git clone https://github.com/purohitkaran07/clime-forecastguard.git
cd clime-forecastguard
```

### 2. Set Up a Virtual Environment

**On Windows (PowerShell / Command Prompt):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python -m uvicorn api.index:app --host 127.0.0.1 --port 8000
```

### 5. Access the Interface
Open your web browser and navigate to:
```
http://127.0.0.1:8000
```

For development with automatic code reloading:
```bash
python -m uvicorn api.index:app --reload --host 127.0.0.1 --port 8000
```

---

## API Reference

The backend provides a comprehensive REST API. Interactive OpenAPI documentation is accessible at `/docs` when running locally.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status, model availability, and version metadata. |
| `GET` | `/api/summary` | India-wide aggregate reliability KPIs (average confidence, alert count). |
| `GET` | `/api/kpis` | Alias to `/api/summary`. |
| `GET` | `/api/regions` | List of 18 monitored states / meteorological regions. |
| `GET` | `/api/forecast-map?day=N` | State-level bust probabilities and confidence scores for lead time Day 1–10. |
| `GET` | `/api/district-map?day=N` | District-level reliability for 554 districts (supports `?state=RegionName` filtering). |
| `GET` | `/api/day-analysis/{day}` | Risk distribution breakdown across monitored areas for lead time `day`. |
| `GET` | `/api/alerts` | Top high-risk operational alert cells across regions and lead times. |
| `GET` | `/api/forecast/{region}?day=N` | Detailed forecast card, variables, and 10-day trend for a state. |
| `GET` | `/api/district/{code}?day=N` | District reliability card, contributors, and 10-day trend by code or name. |
| `GET` | `/api/decision-center` | AI Weather Decision Center analysis payload with risk drivers and sector guidance. |
| `GET` | `/api/live-forecast` | Real-time Open-Meteo weather ingest scored through the reliability model. |
| `GET` | `/api/live-forecast/{region}`| Current, hourly, and daily live telemetry for a specific state. |
| `GET` | `/api/explain/{region}/{day}` | Feature attribution bars and statistical error thresholds (P75, P90, P95). |
| `GET` | `/api/events` | Historical case study replays (cyclones, monsoon events). |
| `GET` | `/api/model-performance` | State model validation metrics (ROC-AUC, Precision, Recall, Confusion Matrix). |
| `GET` | `/api/district-model-performance` | District model validation metrics and calibration data. |
| `GET` | `/api/thresholds` | Fitted statistical bust thresholds by region and lead time. |
| `GET` | `/api/pipeline` | Complete 7-step reliability pipeline metadata and input/output definitions. |
| `GET` | `/api/integration` | Machine-readable API registry and schema specifications. |
| `POST` | `/api/predict` | Score a custom meteorological feature payload in real time. |
| `POST` | `/api/score-grid` | Batch-score arbitrary NWP grid cells and lead times. |

---

## Running the Automated Test Suite

A standalone test suite is provided to verify API integrity and model scoring:

```bash
pytest tests/ -v
```

Or using standard Python:
```bash
python -m unittest discover -s tests
```

---

## License

License: To be determined.
