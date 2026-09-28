"""XGBoost native SHAP (pred_contribs) → ranked, human-readable factors."""
from .config import FEATURES, FEATURE_LABELS


def shap_contributors(contrib_row, n=4):
    """
    contrib_row is length n_features+1 (bias in the last column).
    Positive SHAP on class 1 increases predicted bust probability.
    """
    values = list(contrib_row[: len(FEATURES)])
    ranked = sorted(
        zip(FEATURES, values),
        key=lambda kv: abs(float(kv[1])),
        reverse=True,
    )[:n]
    out = []
    for feature, shap_value in ranked:
        shap_value = float(shap_value)
        out.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "shap": round(shap_value, 4),
            "direction": "increases_bust_risk" if shap_value > 0 else "decreases_bust_risk",
        })
    return out


def explanation_narrative(contributors, risk_level):
    drivers = [
        contributor["label"].lower()
        for contributor in contributors
        if contributor.get("direction") == "increases_bust_risk"
    ][:4]
    if not drivers:
        if risk_level == "LOW":
            return "Low bust risk. The strongest local model contributions do not raise bust probability."
        return "Local SHAP contributors are unavailable for this prediction."

    joined = ", ".join(drivers[:-1]) + (" and " + drivers[-1] if len(drivers) > 1 else drivers[0])
    if risk_level == "HIGH":
        return f"High bust risk. Risk-increasing model contributions: {joined}."
    if risk_level == "MODERATE":
        return f"Moderate bust risk. Risk-increasing model contributions: {joined}."
    return f"Low bust risk; the strongest risk-increasing signals are {joined}, but the combined model probability remains low."
