"""
Per-site model configurations for DATect raw-data DA forecasting.

Each site can override:
  - xgb_params: XGBoost hyperparameter overrides (merged onto base_params)
  - param_grid: Custom PARAM_GRID for per-anchor tuning (replaces global grid)
  - feature_subset: Explicit list of features to keep (None = use all default features)
  - ensemble_weights: (xgb_weight, naive_weight) tuple (None = use global weights)
  - prediction_clip_q: Custom quantile for prediction clipping (None = use global)
  - prediction_clip_max: Hard ceiling on predictions in ug/g (None = no hard ceiling)
"""

from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Feature group definitions (for readable subset selection)
# --------------------------------------------------------------------------

PERSISTENCE_FEATURES = [
    'last_observed_da_raw',
    'weeks_since_last_spike',
]

LAG_FEATURES_SHORT = [
    'da_raw_lag_1',
    'da_raw_lag_2',
    'da_raw_lag_diff_1',
]

LAG_FEATURES_FULL = [
    'da_raw_lag_1',
    'da_raw_lag_2',
    'da_raw_lag_3',
    'da_raw_lag_4',
    'da_raw_lag_diff_1',
    'da_raw_lag_diff_2',
]

ROLLING_FEATURES_SHORT = [
    'raw_obs_roll_mean_4',
    'raw_obs_roll_max_4',
]

ROLLING_FEATURES_FULL = [
    'raw_obs_roll_mean_4',
    'raw_obs_roll_std_4',
    'raw_obs_roll_max_4',
    'raw_obs_roll_mean_8',
    'raw_obs_roll_std_8',
    'raw_obs_roll_max_8',
    'raw_obs_roll_mean_12',
    'raw_obs_roll_std_12',
    'raw_obs_roll_max_12',
]

ENV_FEATURES_CORE = [
    'modis-sst',
    'pdo',
    'modis-chla',
    'beuti',
]

TEMPORAL_FEATURES_CORE = [
    'sin_day_of_year',
    'cos_day_of_year',
    'month',
]

TEMPORAL_FEATURES_FULL = [
    'sin_day_of_year',
    'cos_day_of_year',
    'month',
    'sin_month',
    'cos_month',
    'sin_week_of_year',
    'cos_week_of_year',
    'days_since_start',
]


# --------------------------------------------------------------------------
# Site-specific configuration dictionary
# --------------------------------------------------------------------------

SITE_SPECIFIC_CONFIGS: Dict[str, Dict[str, Any]] = {

    # ==================================================================
    # CATASTROPHIC SITES -- aggressive regularization & feature reduction
    # ==================================================================

    'Cannon Beach': {
        # N=61, R²=-44. Extreme over-prediction (predictions of 9032, 303).
        # Root cause: max_depth=6 with ~40 training samples = memorization.
        # Strategy: ultra-shallow trees, minimal features, hard prediction cap.
        'xgb_params': {
            'max_depth': 2,
            'n_estimators': 100,
            'learning_rate': 0.03,
            'min_child_weight': 10,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'gamma': 1.0,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
        },
        'param_grid': [
            {'max_depth': 2, 'n_estimators': 100, 'learning_rate': 0.03,
             'min_child_weight': 10},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES
            + LAG_FEATURES_SHORT
            + TEMPORAL_FEATURES_CORE
            + ['modis-sst', 'pdo']
        ),
        'ensemble_weights': (0.35, 0.65),
        'prediction_clip_q': 0.95,
        'prediction_clip_max': 130.0,
    },

    'Coos Bay': {
        # N=67, R²=-0.27. Wide range (0-93+), over-predicts low periods.
        # Strategy: moderate regularization, keep more features for genuine signal.
        'xgb_params': {
            'max_depth': 3,
            'n_estimators': 200,
            'learning_rate': 0.03,
            'min_child_weight': 7,
            'reg_alpha': 0.5,
            'reg_lambda': 3.0,
            'gamma': 0.5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        'param_grid': [
            {'max_depth': 3, 'n_estimators': 200, 'learning_rate': 0.03,
             'min_child_weight': 7},
            {'max_depth': 2, 'n_estimators': 150, 'learning_rate': 0.03,
             'min_child_weight': 10},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES
            + LAG_FEATURES_FULL
            + ROLLING_FEATURES_SHORT
            + ENV_FEATURES_CORE
            + TEMPORAL_FEATURES_CORE
        ),
        'ensemble_weights': (0.50, 0.50),
        'prediction_clip_q': 0.97,
        'prediction_clip_max': None,
    },

    'Gold Beach': {
        # N=144, R²=-0.94. Mostly 0-1 actuals, systematic over-prediction.
        # Strategy: similar to Coos Bay but slightly more conservative.
        'xgb_params': {
            'max_depth': 3,
            'n_estimators': 200,
            'learning_rate': 0.03,
            'min_child_weight': 8,
            'reg_alpha': 0.5,
            'reg_lambda': 3.0,
            'gamma': 0.5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        'param_grid': [
            {'max_depth': 3, 'n_estimators': 200, 'learning_rate': 0.03,
             'min_child_weight': 8},
            {'max_depth': 2, 'n_estimators': 150, 'learning_rate': 0.05,
             'min_child_weight': 10},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES
            + LAG_FEATURES_FULL
            + ROLLING_FEATURES_SHORT
            + ENV_FEATURES_CORE
            + TEMPORAL_FEATURES_CORE
        ),
        'ensemble_weights': (0.45, 0.55),
        'prediction_clip_q': 0.97,
        'prediction_clip_max': None,
    },

    'Newport': {
        # N=142, R²=-0.28. Both XGB and naive struggle equally.
        # Strategy: moderate regularization, more features since N is decent.
        'xgb_params': {
            'max_depth': 3,
            'n_estimators': 250,
            'learning_rate': 0.03,
            'min_child_weight': 7,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'gamma': 0.3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        'param_grid': [
            {'max_depth': 3, 'n_estimators': 250, 'learning_rate': 0.03,
             'min_child_weight': 7},
            {'max_depth': 4, 'n_estimators': 200, 'learning_rate': 0.03,
             'min_child_weight': 5},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES
            + LAG_FEATURES_FULL
            + ROLLING_FEATURES_FULL
            + ENV_FEATURES_CORE
            + TEMPORAL_FEATURES_CORE
        ),
        'ensemble_weights': (0.50, 0.50),
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },

    # ==================================================================
    # HIGH-PERFORMING SITES -- light tuning, preserve performance
    # ==================================================================

    'Copalis': {
        # R²=0.72, N=167. Already excellent -- true pass-through, no intervention.
        'xgb_params': None,
        'param_grid': None,
        'feature_subset': None,
        'ensemble_weights': None,
        'prediction_clip_q': None,
        'prediction_clip_max': None,
    },

    'Kalaloch': {
        # R²=0.67, N=131. Already excellent -- true pass-through, no intervention.
        'xgb_params': None,
        'param_grid': None,
        'feature_subset': None,
        'ensemble_weights': None,
        'prediction_clip_q': None,
        'prediction_clip_max': None,
    },

    'Quinault': {
        # R²=0.64, N=113. Good performance -- true pass-through, no intervention.
        'xgb_params': None,
        'param_grid': None,
        'feature_subset': None,
        'ensemble_weights': None,
        'prediction_clip_q': None,
        'prediction_clip_max': None,
    },

    'Twin Harbors': {
        # R²=0.62, N=138. Good but naive is better (R²=0.76).
        # Only change: ensemble weight favors naive more. No XGB param overrides.
        'xgb_params': None,
        'param_grid': None,
        'feature_subset': None,
        'ensemble_weights': (0.50, 0.50),
        'prediction_clip_q': None,
        'prediction_clip_max': None,
    },

    # ==================================================================
    # MODERATE SITES -- use global defaults
    # ==================================================================

    'Long Beach': {
        'xgb_params': None,
        'param_grid': None,
        'feature_subset': None,
        'ensemble_weights': None,
        'prediction_clip_q': None,
        'prediction_clip_max': None,
    },

    'Clatsop Beach': {
        'xgb_params': None,
        'param_grid': None,
        'feature_subset': None,
        'ensemble_weights': None,
        'prediction_clip_q': None,
        'prediction_clip_max': None,
    },
}


# --------------------------------------------------------------------------
# Default config for sites not in SITE_SPECIFIC_CONFIGS
# --------------------------------------------------------------------------

DEFAULT_SITE_CONFIG: Dict[str, Any] = {
    'xgb_params': None,
    'param_grid': None,
    'feature_subset': None,
    'ensemble_weights': None,
    'prediction_clip_q': None,
    'prediction_clip_max': None,
}


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def get_site_config(site: str) -> Dict[str, Any]:
    """Return per-site configuration, falling back to defaults."""
    cfg = dict(DEFAULT_SITE_CONFIG)
    cfg.update(SITE_SPECIFIC_CONFIGS.get(site, {}))
    return cfg


def apply_site_xgb_params(base_params: dict, site: str) -> dict:
    """Merge site-specific XGB params onto global base_params."""
    site_cfg = get_site_config(site)
    if site_cfg['xgb_params'] is None:
        return dict(base_params)
    return {**base_params, **site_cfg['xgb_params']}


def get_site_param_grid(site: str) -> Optional[List[dict]]:
    """Return site-specific PARAM_GRID, or None to use global grid."""
    return get_site_config(site)['param_grid']


def get_site_ensemble_weights(site: str) -> Tuple[float, float]:
    """Return (xgb_weight, naive_weight) for this site. Default: (0.65, 0.35)."""
    weights = get_site_config(site)['ensemble_weights']
    if weights is not None:
        return weights
    return (0.65, 0.35)


def get_site_clip_params(site: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (clip_quantile, clip_max) for prediction clipping."""
    cfg = get_site_config(site)
    return cfg['prediction_clip_q'], cfg['prediction_clip_max']


def compute_site_drop_cols(
    base_drop_cols: list,
    all_columns: list,
    site: str,
) -> list:
    """Extend drop_cols to enforce site-specific feature subset.

    If the site has a feature_subset, any column NOT in the subset
    (and not already in base_drop_cols) gets added to drop_cols.
    """
    cfg = get_site_config(site)
    subset = cfg['feature_subset']
    if subset is None:
        return list(base_drop_cols)

    drops = set(base_drop_cols)
    subset_set = set(subset)
    for col in all_columns:
        if col not in subset_set and col not in drops:
            drops.add(col)
    return list(drops)
