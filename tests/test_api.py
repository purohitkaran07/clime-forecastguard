import unittest
from fastapi.testclient import TestClient
from api.index import app, MODEL, DISTRICT_MODEL, REGIONS

client = TestClient(app)


class TestClimeForecastGuardAPI(unittest.TestCase):
    """Integration test suite for Clime / ForecastGuard reliability engine."""

    def test_01_models_loaded(self):
        """Verify that state and district models are instantiated RandomForestClassifiers."""
        self.assertIsNotNone(MODEL, "State reliability model failed to load")
        self.assertIsNotNone(DISTRICT_MODEL, "District reliability model failed to load")
        self.assertEqual(MODEL.__class__.__name__, "RandomForestClassifier")
        self.assertEqual(DISTRICT_MODEL.__class__.__name__, "RandomForestClassifier")
        self.assertEqual(len(REGIONS), 18, "Expected exactly 18 monitored meteorological regions")

    def test_02_health_check(self):
        """Test GET /api/health returns 200 OK and model status."""
        res = client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertTrue(data.get("model_loaded"))
        self.assertEqual(data.get("regions"), 18)

    def test_03_summary_and_kpis(self):
        """Test GET /api/summary and alias GET /api/kpis."""
        res_sum = client.get("/api/summary")
        self.assertEqual(res_sum.status_code, 200)
        data_sum = res_sum.json()
        self.assertIn("average_confidence", data_sum)
        self.assertEqual(data_sum.get("regions_monitored"), 554)

        res_kpi = client.get("/api/kpis")
        self.assertEqual(res_kpi.status_code, 200)
        self.assertEqual(res_kpi.json(), data_sum)

    def test_04_regions_list(self):
        """Test GET /api/regions returns 18 distinct regions."""
        res = client.get("/api/regions")
        self.assertEqual(res.status_code, 200)
        regions = res.json().get("regions", [])
        self.assertEqual(len(regions), 18)
        self.assertIn("Maharashtra", regions)
        self.assertIn("Punjab", regions)

    def test_05_forecast_map(self):
        """Test GET /api/forecast-map returns state-level bust probabilities."""
        res = client.get("/api/forecast-map?day=4")
        self.assertEqual(res.status_code, 200)
        regions = res.json().get("regions", [])
        self.assertEqual(len(regions), 18)
        first = regions[0]
        self.assertIn("bust_probability", first)
        self.assertIn("confidence", first)
        self.assertIn("risk_level", first)
        self.assertTrue(0 <= first["bust_probability"] <= 100)
        self.assertTrue(0 <= first["confidence"] <= 100)

    def test_06_district_map(self):
        """Test GET /api/district-map returns all 554 districts and supports state filtering."""
        res_all = client.get("/api/district-map?day=4")
        self.assertEqual(res_all.status_code, 200)
        all_districts = res_all.json().get("districts", [])
        self.assertEqual(len(all_districts), 554)

        # Test state filtering
        res_mh = client.get("/api/district-map?day=4&state=Maharashtra")
        self.assertEqual(res_mh.status_code, 200)
        mh_districts = res_mh.json().get("districts", [])
        self.assertTrue(len(mh_districts) > 0)
        self.assertTrue(all(d["state"] == "Maharashtra" for d in mh_districts))

    def test_07_forecast_region_detail(self):
        """Test GET /api/forecast/{region}?day=N returns regional card."""
        res = client.get("/api/forecast/Maharashtra?day=4")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("region"), "Maharashtra")
        self.assertEqual(data.get("day"), 4)
        self.assertIn("bust_probability", data)
        self.assertIn("confidence", data)

    def test_08_district_detail_code_and_name(self):
        """Test GET /api/district/{code} supports both district code and name."""
        res_code = client.get("/api/district/IN-M-PUN?day=4")
        self.assertEqual(res_code.status_code, 200)
        data_code = res_code.json()
        self.assertEqual(data_code.get("district"), "Pune")
        self.assertEqual(data_code.get("district_code"), "IN-M-PUN")
        self.assertIn("reliability_trend", data_code)

        res_name = client.get("/api/district/Pune?day=4")
        self.assertEqual(res_name.status_code, 200)
        self.assertEqual(res_name.json()["district_code"], "IN-M-PUN")

    def test_09_decision_center(self):
        """Test GET /api/decision-center for state and district scopes."""
        # State scope
        res_state = client.get("/api/decision-center?state=Maharashtra&day=3")
        self.assertEqual(res_state.status_code, 200)
        d_state = res_state.json()
        self.assertEqual(d_state.get("state"), "Maharashtra")
        self.assertIn("bust_probability", d_state)
        self.assertIn("confidence", d_state)
        self.assertIn("weather_context", d_state)
        self.assertIn("recommendations", d_state)
        self.assertTrue(len(d_state["recommendations"]) > 0)

        # District scope
        res_dist = client.get("/api/decision-center?state=Maharashtra&district=Pune&day=4")
        self.assertEqual(res_dist.status_code, 200)
        d_dist = res_dist.json()
        self.assertEqual(d_dist.get("district"), "Pune")
        self.assertTrue("shap_contributors" in d_dist or "top_contributors" in d_dist)

    def test_10_explainability(self):
        """Test GET /api/explain/{region}/{day} returns feature attribution."""
        res = client.get("/api/explain/Maharashtra/4?percentile=90")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("shap_contributors", data)
        self.assertIn("explanation_method", data)
        self.assertIn("Random Forest", data["explanation_method"])

    def test_11_model_performance(self):
        """Test model performance endpoints for state and district models."""
        res_state = client.get("/api/model-performance")
        self.assertEqual(res_state.status_code, 200)
        d_state = res_state.json()
        self.assertIn("roc_auc", d_state)
        self.assertIn("confusion_matrix", d_state)
        self.assertEqual(d_state.get("model_type"), "RandomForestClassifier")

        res_dist = client.get("/api/district-model-performance")
        self.assertEqual(res_dist.status_code, 200)
        d_dist = res_dist.json()
        self.assertIn("roc_auc", d_dist)
        self.assertEqual(d_dist.get("model_type"), "RandomForestClassifier")

    def test_12_events_and_thresholds(self):
        """Test GET /api/events and GET /api/thresholds."""
        res_events = client.get("/api/events")
        self.assertEqual(res_events.status_code, 200)
        events = res_events.json().get("events", [])
        self.assertTrue(len(events) > 0)

        res_thr = client.get("/api/thresholds")
        self.assertEqual(res_thr.status_code, 200)
        thresholds = res_thr.json().get("thresholds", [])
        self.assertTrue(len(thresholds) > 0)

    def test_13_predict_post(self):
        """Test POST /api/predict scoring a custom feature payload."""
        payload = {
            "precipitation": 35.0,
            "temperature": 28.0,
            "pressure": 1012.0,
            "humidity": 75.0,
            "wind_speed": 18.0,
            "wind_direction": 200.0,
            "lead_time": 4,
            "latitude": 19.0,
            "longitude": 73.0,
            "spatial_gradient": 3.8,
            "historical_error": 5.2,
            "pressure_variation": 2.5,
            "precipitation_variability": 9.0,
        }
        res = client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("bust_probability", data)
        self.assertIn("confidence", data)
        self.assertIn("risk_level", data)
        self.assertIn("shap_contributors", data)

    def test_14_score_grid_post(self):
        """Test POST /api/score-grid batch scoring."""
        payload = {
            "cycle": "2026-09-28-00Z",
            "cells": [
                {
                    "region": "Maharashtra",
                    "grid_id": "CELL_1",
                    "latitude": 19.0,
                    "longitude": 73.0,
                    "lead_time": 3,
                    "precipitation": 40.0,
                    "temperature": 29.0,
                    "pressure": 1010.0,
                    "humidity": 80.0,
                    "wind_speed": 22.0,
                    "wind_direction": 240.0,
                    "spatial_gradient": 4.0,
                    "historical_error": 5.0,
                    "pressure_variation": 3.0,
                    "precipitation_variability": 10.0,
                }
            ],
        }
        res = client.post("/api/score-grid", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("count"), 1)
        self.assertEqual(len(data.get("cells", [])), 1)

    def test_15_static_assets(self):
        """Test static assets serving (index.html, styles.css, app.js, geojson)."""
        for path in ["/", "/styles.css", "/app.js", "/india-districts.geojson"]:
            res = client.get(path)
            self.assertEqual(res.status_code, 200, f"Static asset {path} failed")
            self.assertTrue(len(res.content) > 0, f"Static asset {path} is empty")

    def test_16_live_forecast(self):
        """Test GET /api/live-forecast and GET /api/live-forecast/{region}."""
        res = client.get("/api/live-forecast")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("regions", data)
        self.assertEqual(len(data["regions"]), 180)

        res_reg = client.get("/api/live-forecast/Maharashtra")
        self.assertEqual(res_reg.status_code, 200)
        d_reg = res_reg.json()
        self.assertEqual(d_reg.get("region"), "Maharashtra")
        self.assertEqual(d_reg.get("provider"), "Open-Meteo")


if __name__ == "__main__":
    unittest.main()
