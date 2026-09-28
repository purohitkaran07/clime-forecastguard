"""Bust-probability classification and confidence conversion."""
from .config import HIGH_BUST_RISK_MIN, MODERATE_BUST_RISK_MIN


def confidence_from_bust(probability: float):
    """Return inverse bust probability (as percentage [0, 100]) and bust-risk level for probability in [0, 1]."""
    p = float(probability)
    conf = round((1.0 - p) * 100, 1)
    if p >= HIGH_BUST_RISK_MIN:
        level = "HIGH"
    elif p >= MODERATE_BUST_RISK_MIN:
        level = "MODERATE"
    else:
        level = "LOW"
    return conf, level


def confidence_percent(confidence: float) -> float:
    """Format fractional confidence as a percentage for display clients."""
    return round(float(confidence) * 100, 1)


def error_prone_area(bust_probability_pct: float, historical_error: float = 0) -> bool:
    """Flag cells classified as high bust risk; historical error is retained for API compatibility."""
    return bust_probability_pct >= HIGH_BUST_RISK_MIN * 100
