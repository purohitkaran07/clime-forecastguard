#!/usr/bin/env bash
# ForecastGuard AI - Full Pipeline Runner
# Runs all data generation, training, and snapshot building steps in order.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="$BASE_DIR/.venv313/bin/python"

echo "=========================================="
echo "  ForecastGuard AI - Full Pipeline"
echo "  Base: $BASE_DIR"
echo "=========================================="

cd "$BASE_DIR"

echo ""
echo "[1/6] Generating state-level training dataset..."
"$PYTHON" scripts/generate_data.py

echo ""
echo "[2/6] Training state-level reliability model..."
"$PYTHON" scripts/train_model.py

echo ""
echo "[3/6] Building state-level demo snapshot..."
"$PYTHON" scripts/build_demo_snapshot.py

echo ""
echo "[4/6] Generating district-level training dataset (may take ~1 min)..."
"$PYTHON" scripts/generate_district_data.py

echo ""
echo "[5/6] Training district-level model (time-based split)..."
"$PYTHON" scripts/train_district_model.py

echo ""
echo "[6/6] Building district snapshot + state aggregation + KPIs/Alerts/Events..."
"$PYTHON" scripts/build_district_snapshot.py
"$PYTHON" scripts/build_extras.py

echo ""
echo "=========================================="
echo "  Pipeline complete! Starting API server..."
echo "  URL: http://localhost:8000"
echo "=========================================="
CLIME_SERVE_STATIC=1 "$PYTHON" -m uvicorn api.index:app --reload --host 0.0.0.0 --port 8000
