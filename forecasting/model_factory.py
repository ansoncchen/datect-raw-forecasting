"""
Model factory for DATect raw forecasting.
"""

from __future__ import annotations

from typing import Optional

from xgboost import XGBClassifier, XGBRegressor

import config


def _resolve_xgb_device_params() -> dict:
    """
    Enforce CPU/GPU selection based on config.USE_GPU.
    """
    if config.USE_GPU is True:
        return {"tree_method": "gpu_hist", "device": "cuda"}
    if config.USE_GPU is False:
        return {"tree_method": "hist", "device": "cpu"}
    return {"tree_method": "hist"}


def build_xgb_regressor(param_overrides: Optional[dict] = None) -> XGBRegressor:
    """
    Build an XGBoost regressor with config defaults.
    """
    base_params = dict(config.XGB_REGRESSION_PARAMS)
    params = {**base_params, **(param_overrides or {})}
    params.pop("tree_method", None)
    params.update(_resolve_xgb_device_params())
    return XGBRegressor(**params, random_state=config.RANDOM_SEED, verbosity=0)


def build_xgb_classifier(param_overrides: Optional[dict] = None) -> XGBClassifier:
    """
    Build an XGBoost classifier with config defaults.
    """
    base_params = dict(config.XGB_CLASSIFICATION_PARAMS)
    params = {**base_params, **(param_overrides or {})}
    params.pop("tree_method", None)
    params.update(_resolve_xgb_device_params())
    return XGBClassifier(**params, random_state=config.RANDOM_SEED, verbosity=0)
