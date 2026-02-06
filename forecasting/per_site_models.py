"""
Per-site model configurations for DATect raw-data DA forecasting.

Each site can override:
  - xgb_params: XGBoost hyperparameter overrides (merged onto base_params)
  - rf_params: Random Forest hyperparameter overrides (merged onto RF base_params)
  - param_grid: Custom PARAM_GRID for per-anchor XGB tuning (replaces global grid)
  - feature_subset: Explicit list of features to keep (None = use all default features)
  - ensemble_weights: (xgb_weight, rf_weight, naive_weight) tuple (None = use global)
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

# Conservative RF params for sites where RF R² < 0.1
RF_CONSERVATIVE = {
    'n_estimators': 200,
    'max_depth': 6,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 0.5,
}


# --------------------------------------------------------------------------
# Site-specific configuration dictionary
# --------------------------------------------------------------------------

SITE_SPECIFIC_CONFIGS: Dict[str, Dict[str, Any]] = {

    # ==================================================================
    # PERSISTENCE-DOMINANT SITES -- naive >> XGB, strip env features,
    # lean ensemble toward naive
    # ==================================================================

    'Copalis': {
        # N=167, XGB R²=0.732, RF R²=0.765, Naive R²=0.715.
        # All three strong. RF best → lean RF. Ens was 0.761 (good).
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
        'rf_params': None,  # Global RF defaults work well (R²=0.765)
        'param_grid': [
            {'max_depth': 2, 'n_estimators': 100, 'learning_rate': 0.03,
             'min_child_weight': 10},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES
            + LAG_FEATURES_SHORT
            + ROLLING_FEATURES_SHORT
            + TEMPORAL_FEATURES_CORE
        ),
        'ensemble_weights': (0.25, 0.45, 0.30),  # (xgb, rf, naive) — RF leads
        'prediction_clip_q': 0.97,
        'prediction_clip_max': None,
    },

    'Kalaloch': {
        # N=131, XGB R²=0.565, RF R²=0.679, Naive R²=0.669.
        # RF excels. Ens was 0.679 = RF alone. Lean harder into RF.
        'xgb_params': {
            'max_depth': 2,
            'n_estimators': 80,
            'learning_rate': 0.02,
            'min_child_weight': 12,
            'reg_alpha': 2.0,
            'reg_lambda': 10.0,
            'gamma': 2.0,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
        },
        'rf_params': None,  # RF excels here (R²=0.679)
        'param_grid': [
            {'max_depth': 2, 'n_estimators': 80, 'learning_rate': 0.02,
             'min_child_weight': 12},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES
            + LAG_FEATURES_SHORT
            + TEMPORAL_FEATURES_CORE
        ),
        'ensemble_weights': (0.10, 0.50, 0.40),  # (xgb, rf, naive) — RF dominant
        'prediction_clip_q': 0.95,
        'prediction_clip_max': 80.0,
    },

    'Twin Harbors': {
        # N=138, XGB R²=0.597, RF R²=0.604, Naive R²=0.763.
        # Naive dominates. Ens was 0.795 (best site!) — weights already great.
        # Slight bump to naive since it's clearly best individual model.
        'xgb_params': {
            'max_depth': 3,
            'n_estimators': 150,
            'learning_rate': 0.03,
            'min_child_weight': 8,
            'reg_alpha': 0.5,
            'reg_lambda': 3.0,
            'gamma': 0.5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        'rf_params': None,  # RF decent (R²=0.604)
        'param_grid': [
            {'max_depth': 3, 'n_estimators': 150, 'learning_rate': 0.03,
             'min_child_weight': 8},
            {'max_depth': 2, 'n_estimators': 100, 'learning_rate': 0.03,
             'min_child_weight': 10},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES
            + LAG_FEATURES_FULL
            + ROLLING_FEATURES_SHORT
            + ['modis-sst', 'pdo']
            + TEMPORAL_FEATURES_CORE
        ),
        'ensemble_weights': (0.15, 0.25, 0.60),  # (xgb, rf, naive) — naive leads
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },

    'Quinault': {
        # N=113, XGB R²=0.528, RF R²=0.585, Naive R²=0.590.
        # All similar. Ens was 0.654 (great!). Blend works well here.
        # Naive/RF slightly edge XGB → keep balanced, slight naive lean.
        'xgb_params': {
            'max_depth': 3,
            'n_estimators': 200,
            'learning_rate': 0.03,
            'min_child_weight': 7,
            'reg_alpha': 0.3,
            'reg_lambda': 2.0,
            'gamma': 0.3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        'rf_params': None,  # RF good (R²=0.585)
        'param_grid': [
            {'max_depth': 3, 'n_estimators': 200, 'learning_rate': 0.03,
             'min_child_weight': 7},
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
        'ensemble_weights': (0.25, 0.35, 0.40),  # (xgb, rf, naive) — balanced
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },

    # ==================================================================
    # XGB-LEANING SITE -- XGB > naive, lean ensemble toward XGB
    # ==================================================================

    'Long Beach': {
        # N=140, XGB R²=0.638, RF R²=0.615, Naive R²=0.470.
        # XGB best, RF close. Ens was 0.640 ≈ XGB alone. Lean XGB+RF.
        'xgb_params': {
            'max_depth': 3,
            'n_estimators': 250,
            'learning_rate': 0.03,
            'min_child_weight': 7,
            'reg_alpha': 0.3,
            'reg_lambda': 2.0,
            'gamma': 0.3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        'rf_params': None,  # RF strong (R²=0.615)
        'param_grid': [
            {'max_depth': 3, 'n_estimators': 250, 'learning_rate': 0.03,
             'min_child_weight': 7},
            {'max_depth': 4, 'n_estimators': 200, 'learning_rate': 0.05,
             'min_child_weight': 5},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES
            + LAG_FEATURES_FULL
            + ROLLING_FEATURES_FULL
            + ENV_FEATURES_CORE
            + TEMPORAL_FEATURES_CORE
        ),
        'ensemble_weights': (0.50, 0.35, 0.15),  # (xgb, rf, naive) — XGB leads
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },

    # ==================================================================
    # ENVIRONMENT-RESPONSIVE SITES -- XGB > naive, lean ensemble toward XGB
    # ==================================================================

    'Clatsop Beach': {
        # N=218, XGB R²=0.171, RF R²=0.238, Naive R²=-0.015.
        # RF best, XGB decent, naive useless. Ens was 0.213 < RF 0.238.
        # Lean harder into RF, drop naive weight.
        'xgb_params': None,
        'rf_params': None,  # RF decent (R²=0.238)
        'param_grid': None,
        'feature_subset': None,
        'ensemble_weights': (0.40, 0.55, 0.05),  # (xgb, rf, naive) — RF leads
        'prediction_clip_q': None,
        'prediction_clip_max': None,
    },

    'Coos Bay': {
        # N=67, XGB R²=0.337, RF R²=0.305, Naive R²=-0.570.
        # XGB best, RF close behind. Ens was 0.311 < XGB 0.337.
        # Near-pure XGB+RF split, minimize naive drag.
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
        'rf_params': dict(RF_CONSERVATIVE),  # Conservative: RF decent with constraints
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
        'ensemble_weights': (0.55, 0.40, 0.05),  # (xgb, rf, naive) — both ML, drop naive
        'prediction_clip_q': 0.97,
        'prediction_clip_max': None,
    },

    # ==================================================================
    # BOTH-STRUGGLE SITES -- both XGB and naive have negative R²
    # Aggressive regularization, feature reduction, lean XGB (naive is worse)
    # ==================================================================

    'Cannon Beach': {
        # N=61 (smallest), XGB R²=-0.257, RF R²=-0.539, Naive R²=-10.663.
        # All models terrible. XGB least bad. Ens was -0.607 (RF dragged it down).
        # Near-pure XGB — minimize RF/naive contamination.
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
        'rf_params': dict(RF_CONSERVATIVE),  # Conservative: RF terrible here
        'param_grid': [
            {'max_depth': 2, 'n_estimators': 80, 'learning_rate': 0.02,
             'min_child_weight': 12},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES
            + LAG_FEATURES_SHORT
            + TEMPORAL_FEATURES_CORE
            + ['modis-sst', 'pdo']
        ),
        'ensemble_weights': (0.95, 0.03, 0.02),  # (xgb, rf, naive) — near-pure XGB
        'prediction_clip_q': 0.95,
        'prediction_clip_max': 80.0,
    },

    'Gold Beach': {
        # N=144, XGB R²=-0.094, RF R²=-0.091, Naive R²=-1.656.
        # XGB and RF nearly tied (both slightly negative). Naive catastrophic.
        # Ens was -0.148 (naive drag). Split between XGB+RF, kill naive.
        'xgb_params': {
            'max_depth': 2,
            'n_estimators': 150,
            'learning_rate': 0.03,
            'min_child_weight': 10,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'gamma': 1.0,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
        },
        'rf_params': dict(RF_CONSERVATIVE),  # Conservative: RF marginally better than XGB
        'param_grid': [
            {'max_depth': 2, 'n_estimators': 150, 'learning_rate': 0.03,
             'min_child_weight': 10},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES
            + LAG_FEATURES_SHORT
            + ROLLING_FEATURES_SHORT
            + ['modis-sst', 'pdo']
            + TEMPORAL_FEATURES_CORE
        ),
        'ensemble_weights': (0.50, 0.47, 0.03),  # (xgb, rf, naive) — both ML, kill naive
        'prediction_clip_q': 0.95,
        'prediction_clip_max': None,
    },

    'Newport': {
        # N=142, XGB R²=-0.127, RF R²=+0.038, Naive R²=-0.287.
        # RF is the ONLY positive-R² model! Ens was -0.136 (XGB drag).
        # Lean heavily into RF — it's the only one that works.
        'xgb_params': {
            'max_depth': 3,
            'n_estimators': 250,
            'learning_rate': 0.03,
            'min_child_weight': 7,
            'reg_alpha': 0.5,
            'reg_lambda': 3.0,
            'gamma': 0.5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        'rf_params': dict(RF_CONSERVATIVE),  # Conservative but still best model here
        'param_grid': [
            {'max_depth': 3, 'n_estimators': 250, 'learning_rate': 0.03,
             'min_child_weight': 7},
            {'max_depth': 4, 'n_estimators': 200, 'learning_rate': 0.03,
             'min_child_weight': 5},
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
        'ensemble_weights': (0.20, 0.70, 0.10),  # (xgb, rf, naive) — RF dominant
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },
}


# --------------------------------------------------------------------------
# Default config for sites not in SITE_SPECIFIC_CONFIGS
# --------------------------------------------------------------------------

DEFAULT_SITE_CONFIG: Dict[str, Any] = {
    'xgb_params': None,
    'rf_params': None,
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


def apply_site_rf_params(base_params: dict, site: str) -> dict:
    """Merge site-specific RF params onto global base_params."""
    site_cfg = get_site_config(site)
    if site_cfg['rf_params'] is None:
        return dict(base_params)
    return {**base_params, **site_cfg['rf_params']}


def get_site_param_grid(site: str) -> Optional[List[dict]]:
    """Return site-specific PARAM_GRID, or None to use global grid."""
    return get_site_config(site)['param_grid']


def get_site_ensemble_weights(site: str) -> Tuple[float, float, float]:
    """Return (xgb_weight, rf_weight, naive_weight) for this site.

    Default: (0.50, 0.15, 0.35).
    """
    weights = get_site_config(site)['ensemble_weights']
    if weights is not None:
        return weights
    return (0.50, 0.15, 0.35)


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
