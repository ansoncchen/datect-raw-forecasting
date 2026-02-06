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
    spike_threshold: float = 20.0,
    sample_weight: Optional[np.ndarray] = None
) -> Tuple[xgb.XGBClassifier, Optional[xgb.XGBRegressor], xgb.XGBRegressor]:
    """
    Train two-stage model:
    Stage 1: Binary classifier (spike vs non-spike)
    Stage 2: Separate regressors for spike and non-spike cases

    Args:
        X_train: Training features
        y_train: Training targets (raw DA values)
        spike_threshold: Threshold for spike classification (μg/g)
        sample_weight: Optional sample weights (ignored to avoid over-prediction)

    Returns:
        Tuple of (classifier, spike_regressor, normal_regressor)
    """
    # Stage 1: Binary spike classifier
    y_binary = (y_train > spike_threshold).astype(int)

    # Handle class imbalance with scale_pos_weight
    n_neg = np.sum(y_binary == 0)
    n_pos = np.sum(y_binary == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    clf_params = {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',
        'random_state': 42
    }

    classifier = xgb.XGBClassifier(**clf_params)
    # Don't use sample_weight - causes over-prediction
    classifier.fit(X_train, y_binary)

    # Stage 2: Train separate regressors
    spike_mask = y_train > spike_threshold

    # Spike regressor (high-DA events)
    reg_spike = None
    if np.sum(spike_mask) >= 10:  # Need minimum samples
        reg_spike_params = {
            'n_estimators': 400,
            'max_depth': 6,  # Deeper tree for complex spike patterns
            'learning_rate': 0.05,
            'tree_method': 'hist',
            'random_state': 42
        }
        reg_spike = xgb.XGBRegressor(**reg_spike_params)
        # Don't use sample_weight - causes over-prediction
        reg_spike.fit(X_train[spike_mask], y_train[spike_mask])

    # Normal regressor (low-DA events)
    reg_normal_params = {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'tree_method': 'hist',
        'random_state': 42
    }
    reg_normal = xgb.XGBRegressor(**reg_normal_params)
    # Don't use sample_weight - causes over-prediction
    reg_normal.fit(X_train[~spike_mask], y_train[~spike_mask])

    return classifier, reg_spike, reg_normal


def predict_two_stage(
    classifier: xgb.XGBClassifier,
    reg_spike: Optional[xgb.XGBRegressor],
    reg_normal: xgb.XGBRegressor,
    X_test: np.ndarray,
    spike_threshold: float = 20.0
) -> Tuple[float, float]:
    """
    Make prediction using two-stage model

    Args:
        classifier: Trained spike classifier
        reg_spike: Trained spike regressor (may be None)
        reg_normal: Trained normal regressor
        X_test: Test features (single sample)
        spike_threshold: Threshold used for training

    Returns:
        Tuple of (final_prediction, spike_probability)
    """
    # Stage 1: Predict spike probability
    spike_prob = classifier.predict_proba(X_test)[0, 1]

    # Stage 2: Get predictions from both regressors
    pred_normal = reg_normal.predict(X_test)[0]

    if reg_spike is not None:
        pred_spike = reg_spike.predict(X_test)[0]
    else:
        # Fallback if no spike training data
        # Use a conservative high value above threshold
        pred_spike = spike_threshold * 1.5

    # Blend based on spike probability
    # Higher spike_prob → weight more toward spike regressor
    final_pred = spike_prob * pred_spike + (1 - spike_prob) * pred_normal

    # Ensure non-negative
    final_pred = max(0.0, final_pred)

    return final_pred, spike_prob
