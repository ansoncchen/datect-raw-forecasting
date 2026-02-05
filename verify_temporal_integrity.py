#!/usr/bin/env python3
"""
Verify temporal integrity for raw-data forecasting.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config
from forecasting.raw_data_forecaster import (
    aggregate_raw_to_weekly,
    build_raw_feature_frame,
    get_site_test_row,
    get_site_training_frame,
    load_raw_da_measurements,
)


def verify_training_cutoff(raw_data: pd.DataFrame, feature_frame: pd.DataFrame, n_samples: int = 50) -> None:
    rng = np.random.RandomState(config.RANDOM_SEED)
    sample_rows = raw_data.sample(min(n_samples, len(raw_data)), random_state=rng)
    for row in sample_rows.itertuples(index=False):
        anchor_date = row.date - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
        train_data = get_site_training_frame(feature_frame, row.site, anchor_date, config.MIN_TRAINING_SAMPLES)
        if train_data is None or train_data.empty:
            raise AssertionError("Training frame missing for sampled row.")
        if train_data["date"].max() > anchor_date:
            raise AssertionError("Training data contains rows after anchor_date.")


def verify_lag_features(feature_frame: pd.DataFrame, lags: list[int]) -> None:
    feature_frame = feature_frame.sort_values(["site", "date"])
    for lag in lags:
        col = f"da_raw_lag_{lag}"
        if col not in feature_frame.columns:
            continue
        expected = feature_frame.groupby("site")["da_raw"].shift(lag)
        actual = feature_frame[col]
        mask = expected.notna() & actual.notna()
        if not np.allclose(expected[mask], actual[mask], equal_nan=True):
            raise AssertionError(f"Lag feature mismatch for {col}.")


def verify_test_features(raw_data: pd.DataFrame, feature_frame: pd.DataFrame, n_samples: int = 30) -> None:
    rng = np.random.RandomState(config.RANDOM_SEED + 1)
    sample_rows = raw_data.sample(min(n_samples, len(raw_data)), random_state=rng)
    for row in sample_rows.itertuples(index=False):
        anchor_date = row.date - pd.Timedelta(days=config.FORECAST_HORIZON_DAYS)
        test_row = get_site_test_row(
            feature_frame,
            row.site,
            row.date,
            anchor_date,
            max_date_diff_days=28,
        )
        if test_row is None:
            continue
        drop_cols = {"date", "site", "da_raw", "da"}
        test_features = test_row.drop(columns=list(drop_cols), errors="ignore")
        if "da_raw" in test_features.columns or "da" in test_features.columns:
            raise AssertionError("Target columns leaked into test features.")
        if test_features.select_dtypes(include=[np.number]).empty:
            raise AssertionError("No numeric features remain after dropping targets.")


def main() -> int:
    raw_data = load_raw_da_measurements()
    processed_data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
    raw_weekly = aggregate_raw_to_weekly(raw_data)
    feature_frame = build_raw_feature_frame(processed_data, raw_weekly)

    verify_training_cutoff(raw_data, feature_frame)
    verify_lag_features(feature_frame, list(config.LAG_FEATURES))
    verify_test_features(raw_data, feature_frame)

    print("Temporal integrity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
