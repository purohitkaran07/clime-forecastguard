"""Shared chronological splitting and training-only bust-label helpers."""
import pandas as pd


def chronological_train_test_split(df, test_fraction=0.25):
    """Split on whole forecast cycles so no timestamp appears in both partitions."""
    if "forecast_cycle" not in df:
        raise ValueError("forecast_cycle is required for chronological validation")

    cycle_dates = pd.to_datetime(df["forecast_cycle"], errors="coerce").dt.normalize()
    unique_cycles = sorted(cycle_dates.dropna().unique())
    if len(unique_cycles) < 2:
        raise ValueError("At least two forecast cycles are required")

    split_index = int(len(unique_cycles) * (1 - test_fraction))
    split_index = min(max(split_index, 1), len(unique_cycles) - 1)
    train_end = unique_cycles[split_index - 1]
    test_start = unique_cycles[split_index]
    train = df.loc[cycle_dates <= train_end].copy()
    test = df.loc[cycle_dates >= test_start].copy()
    return train, test, train_end.strftime("%Y-%m-%d"), test_start.strftime("%Y-%m-%d")


def fit_bust_thresholds(train_df, percentile):
    """Fit precipitation absolute-error thresholds using training cycles only."""
    if "absolute_error" not in train_df:
        raise ValueError("absolute_error is required to fit bust thresholds")
    thresholds = train_df.groupby(["region", "lead_time"])["absolute_error"].quantile(percentile)
    return thresholds.rename("bust_threshold")


def apply_bust_labels(df, thresholds):
    """Apply training-derived thresholds to rows without refitting on them."""
    labeled = df.copy()
    keys = pd.MultiIndex.from_frame(labeled[["region", "lead_time"]])
    labeled["bust_threshold"] = thresholds.reindex(keys).to_numpy()
    if labeled["bust_threshold"].isna().any():
        raise ValueError("Missing training-derived threshold for one or more region/lead-time groups")
    labeled["forecast_bust"] = (labeled["absolute_error"] >= labeled["bust_threshold"]).astype(int)
    return labeled
