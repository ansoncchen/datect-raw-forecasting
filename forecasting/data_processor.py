"""
Minimal DataProcessor for raw-DA forecasting.
"""

import pandas as pd
import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """
    Minimal processor to create raw lag features without interpolation.
    """

    def create_raw_lag_features(self, df, group_col, value_col, lags):
        logger.info("Creating raw lag features for %s (no interpolation)", value_col)
        df = df.copy()
        df_sorted = df.sort_values([group_col, "date"])
        for lag in lags:
            feature_name = f"{value_col}_lag_{lag}"
            df_sorted[feature_name] = df_sorted.groupby(group_col)[value_col].shift(lag)
        return df_sorted
