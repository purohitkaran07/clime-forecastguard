"""
ForecastGuard AI - Model Training
-----------------------------------
Trains a gradient-boosted classifier (RandomForest, chosen over XGBoost for a
lighter/more reliable serverless deployment footprint on Vercel) to predict
`forecast_bust` probability from forecast-time-known variables only.

Saves:
  model/forecast_model.joblib
  model/feature_importance.json
  model/model_performance.json   (REAL metrics on a held-out test split)
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import joblib

FEATURES = [
    "precipitation", "temperature", "pressure", "humidity", "wind_speed",
    "wind_direction", "lead_time", "latitude", "longitude",
    "spatial_gradient", "historical_error", "pressure_variation",
    "precipitation_variability",
]
TARGET = "forecast_bust"

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    df = pd.read_csv(f"{BASE}/data/training_dataset.csv")
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, preds).tolist()
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")

    # Down-sample curve points for compact API payloads
    def sample_curve(x, y, n=20):
        idx = np.linspace(0, len(x) - 1, min(n, len(x))).astype(int)
        return [round(float(v), 4) for v in x[idx]], [round(float(v), 4) for v in y[idx]]

    fpr_s, tpr_s = sample_curve(fpr, tpr)
    rec_s, prec_s = sample_curve(rec, prec)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds)), 4),
        "recall": round(float(recall_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "brier_score": round(float(brier_score_loss(y_test, proba)), 4),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "roc_curve": {"fpr": fpr_s, "tpr": tpr_s},
        "pr_curve": {"recall": rec_s, "precision": prec_s},
        "calibration_curve": {
            "mean_predicted": [round(float(v), 4) for v in mean_pred],
            "fraction_positive": [round(float(v), 4) for v in frac_pos],
        },
        "test_set_size": int(len(y_test)),
        "train_set_size": int(len(y_train)),
        "note": "Prototype model performance on synthetic demonstration dataset.",
    }

    # Performance by lead time (Day 1-10)
    by_lead = []
    test_df = X_test.copy()
    test_df["y_true"] = y_test.values
    test_df["y_proba"] = proba
    for day in sorted(test_df["lead_time"].unique()):
        sub = test_df[test_df["lead_time"] == day]
        if sub["y_true"].nunique() < 2:
            auc = None
        else:
            auc = round(float(roc_auc_score(sub["y_true"], sub["y_proba"])), 4)
        preds_d = (sub["y_proba"] >= 0.5).astype(int)
        by_lead.append({
            "day": int(day),
            "accuracy": round(float(accuracy_score(sub["y_true"], preds_d)), 4),
            "roc_auc": auc,
            "avg_bust_probability": round(float(sub["y_proba"].mean()), 4),
            "n": int(len(sub)),
        })
    metrics["performance_by_lead_time"] = by_lead

    importances = model.feature_importances_
    feat_imp = sorted(
        [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(FEATURES, importances)],
        key=lambda d: -d["importance"]
    )

    os.makedirs(f"{BASE}/model", exist_ok=True)
    joblib.dump(model, f"{BASE}/model/forecast_model.joblib")
    with open(f"{BASE}/model/model_performance.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{BASE}/model/feature_importance.json", "w") as f:
        json.dump(feat_imp, f, indent=2)

    print("Model trained.")
    print(json.dumps({k: v for k, v in metrics.items() if k not in
                       ("confusion_matrix", "roc_curve", "pr_curve", "calibration_curve", "performance_by_lead_time")}, indent=2))
    print("Feature importance:", feat_imp)


if __name__ == "__main__":
    main()
