"""
ForecastGuard AI - District-Level Model Training
--------------------------------------------------
Trains on the district-level dataset with a TIME-BASED split (not random):
forecasts issued in the first 30 of 40 simulated days go to train/val,
the last 10 days are held out as test - so the reported metrics reflect
genuine forward-looking performance, avoiding future-data leakage.

District identity itself is NEVER used as a numeric feature - only
latitude/longitude + meteorological + historical-error features are used,
exactly as requested (spatial info via coordinates, not a district ID).
"""
import json
import os
import numpy as np
import pandas as pd
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
    df = pd.read_csv(f"{BASE}/data/district_training_dataset.csv", parse_dates=["forecast_issued_date"])

    # Time-based split: last 10 of 40 simulated issue-days are the held-out test set.
    cutoff = df["forecast_issued_date"].quantile(0.75)
    train_df = df[df["forecast_issued_date"] <= cutoff]
    test_df = df[df["forecast_issued_date"] > cutoff]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    model = RandomForestClassifier(
        # n_estimators kept modest (150, vs 300 for the state-level model) since
        # the district dataset is ~30x larger - keeps the serialized model and
        # Vercel cold-start load time small while barely affecting accuracy.
        n_estimators=150, max_depth=9, min_samples_leaf=6,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, preds).tolist()
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")

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
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr_s, "tpr": tpr_s},
        "pr_curve": {"recall": rec_s, "precision": prec_s},
        "calibration_curve": {
            "mean_predicted": [round(float(v), 4) for v in mean_pred],
            "fraction_positive": [round(float(v), 4) for v in frac_pos],
        },
        "test_set_size": int(len(y_test)),
        "train_set_size": int(len(y_train)),
        "split_method": "time-based (forecast_issued_date <= 75th percentile = train, later = test)",
        "note": "Prototype model performance on synthetic demonstration dataset (district-level).",
    }

    by_lead = []
    test_eval = X_test.copy()
    test_eval["y_true"] = y_test.values
    test_eval["y_proba"] = proba
    for day in sorted(test_eval["lead_time"].unique()):
        sub = test_eval[test_eval["lead_time"] == day]
        auc = round(float(roc_auc_score(sub["y_true"], sub["y_proba"])), 4) if sub["y_true"].nunique() > 1 else None
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

    joblib.dump(model, f"{BASE}/model/district_forecast_model.joblib")
    with open(f"{BASE}/model/district_model_performance.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{BASE}/model/district_feature_importance.json", "w") as f:
        json.dump(feat_imp, f, indent=2)

    print("District model trained (time-based split).")
    print(json.dumps({k: v for k, v in metrics.items() if k not in
                       ("confusion_matrix", "roc_curve", "pr_curve", "calibration_curve", "performance_by_lead_time")}, indent=2))
    print("Feature importance:", feat_imp)


if __name__ == "__main__":
    main()
