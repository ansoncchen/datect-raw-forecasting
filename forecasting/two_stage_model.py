"""
Two-Stage Model for DA Forecasting

Stage 1: Binary classifier (spike vs non-spike)
Stage 2: Separate regressors for spike and non-spike cases

This architecture aims to improve spike detection metrics (F1, Recall)
by explicitly modeling the spike/non-spike decision separately from
the magnitude prediction.
"""

import numpy as np
import xgboost as xgb
from typing import Tuple, Optional


def train_two_stage_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_params: dict,
    spike_threshold: float = 20.0,
    sample_weight: Optional[np.ndarray] = None
) -> Tuple[xgb.XGBClassifier, xgb.XGBRegressor]:
    """
    Train two-stage model:
    Stage 1: Binary classifier (spike vs non-spike) with probability threshold
    Stage 2: Single regressor for magnitude, with spike-aware prediction adjustment

    This simpler architecture avoids data fragmentation while maintaining
    spike detection focus.

    Args:
        X_train: Training features
        y_train: Training targets (raw DA values)
        model_params: Base XGBoost parameters to use
        spike_threshold: Threshold for spike classification (μg/g)
        sample_weight: Optional sample weights (ignored to avoid over-prediction)

    Returns:
        Tuple of (classifier, regressor)
    """
    # Stage 1: Binary spike classifier
    y_binary = (y_train > spike_threshold).astype(int)

    # Handle class imbalance with scale_pos_weight
    n_neg = np.sum(y_binary == 0)
    n_pos = np.sum(y_binary == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # Use model_params but override for classification
    clf_params = {
        **model_params,
        'scale_pos_weight': scale_pos_weight,
        'objective': 'binary:logistic',  # Binary classification
    }
    # Remove regression-specific params
    clf_params.pop('reg_alpha', None)
    clf_params.pop('reg_lambda', None)

    classifier = xgb.XGBClassifier(**clf_params)
    classifier.fit(X_train, y_binary)

    # Stage 2: Single regressor on all data (maintains data volume)
    reg_params = {**model_params}
    regressor = xgb.XGBRegressor(**reg_params)
    regressor.fit(X_train, y_train)

    return classifier, regressor


def predict_two_stage(
    classifier: xgb.XGBClassifier,
    regressor: xgb.XGBRegressor,
    X_test: np.ndarray,
    spike_threshold: float = 20.0
) -> Tuple[float, float]:
    """
    Make prediction using two-stage model

    Strategy:
    1. Classifier determines spike probability
    2. Regressor predicts magnitude
    3. If high spike probability (>0.5), ensure prediction is at least at threshold

    This prevents the classifier from detecting a spike while the regressor
    predicts a low value.

    Args:
        classifier: Trained spike classifier
        regressor: Trained regressor for magnitude
        X_test: Test features (single sample)
        spike_threshold: Threshold used for training

    Returns:
        Tuple of (final_prediction, spike_probability)
    """
    # Stage 1: Predict spike probability
    spike_prob = classifier.predict_proba(X_test)[0, 1]

    # Stage 2: Get magnitude prediction
    base_pred = regressor.predict(X_test)[0]

    # Adjustment: If classifier says spike (prob > 0.5) but regressor predicts low,
    # blend toward threshold to maintain consistency
    if spike_prob > 0.5 and base_pred < spike_threshold:
        # Blend between base prediction and threshold based on confidence
        # spike_prob = 0.5 → no adjustment
        # spike_prob = 1.0 → push halfway to threshold
        adjustment_factor = (spike_prob - 0.5) * 2  # 0 to 1
        final_pred = base_pred + adjustment_factor * (spike_threshold - base_pred) * 0.5
    else:
        final_pred = base_pred

    # Ensure non-negative
    final_pred = max(0.0, final_pred)

    return final_pred, spike_prob
