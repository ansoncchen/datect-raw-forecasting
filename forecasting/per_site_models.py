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
    # PERSISTENCE-DOMINANT SITES -- naive >> XGB, strip env features,
    # lean ensemble toward naive
    # ==================================================================

    'Copalis': {
        # N=167, XGB R²=0.732, Naive R²=0.715. Largest site.
        # Persistence-only features unlocked strong XGB signal (was 0.097 baseline).
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
            + ROLLING_FEATURES_SHORT
            + TEMPORAL_FEATURES_CORE
        ),
        'ensemble_weights': (0.55, 0.45),
        'prediction_clip_q': 0.97,
        'prediction_clip_max': None,
    },

    'Kalaloch': {
        # N=131, XGB R²=0.565, Naive R²=0.669.
        # Was -1.757 before per-site config. Max regularization + minimal features worked.
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
        'param_grid': [
            {'max_depth': 2, 'n_estimators': 80, 'learning_rate': 0.02,
             'min_child_weight': 12},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES
            + LAG_FEATURES_SHORT
            + TEMPORAL_FEATURES_CORE
        ),
        'ensemble_weights': (0.35, 0.65),
        'prediction_clip_q': 0.95,
        'prediction_clip_max': 80.0,
    },

    'Twin Harbors': {
        # N=138, XGB R²=0.597, Naive R²=0.763.
        # XGB decent but naive clearly better. Limited env (SST+PDO only).
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
        'ensemble_weights': (0.30, 0.70),
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },

    'Quinault': {
        # N=113, XGB R²=0.528, Naive R²=0.590.
        # Modest gap -- keep full env, moderate naive lean.
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
        'ensemble_weights': (0.40, 0.60),
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },

    # ==================================================================
    # XGB-LEANING SITE -- XGB > naive, lean ensemble toward XGB
    # ==================================================================

    'Long Beach': {
        # N=140, XGB R²=0.638, Naive R²=0.470.
        # XGB clearly better -- full features since N supports it.
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
        'ensemble_weights': (0.65, 0.35),
        'prediction_clip_q': 0.98,
        'prediction_clip_max': None,
    },

    # ==================================================================
    # ENVIRONMENT-RESPONSIVE SITES -- XGB > naive, lean ensemble toward XGB
    # ==================================================================

    'Clatsop Beach': {
        # N=218, XGB R²=0.171, Naive R²=-0.015.
        # XGB clearly better -- global params work well, just adjust ensemble.
        'xgb_params': None,
        'param_grid': None,
        'feature_subset': None,
        'ensemble_weights': (0.80, 0.20),
        'prediction_clip_q': None,
        'prediction_clip_max': None,
    },

    'Coos Bay': {
        # N=67, XGB R²=0.337, Naive R²=-0.570.
        # XGB much better than naive. Near-pure XGB ensemble.
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
        'ensemble_weights': (0.90, 0.10),
        'prediction_clip_q': 0.97,
        'prediction_clip_max': None,
    },

    # ==================================================================
    # BOTH-STRUGGLE SITES -- both XGB and naive have negative R²
    # Aggressive regularization, feature reduction, lean XGB (naive is worse)
    # ==================================================================

    'Cannon Beach': {
        # N=61 (smallest), XGB R²=-0.257, Naive R²=-10.663.
        # Was -44 before Iter 1. Naive is catastrophic -- near-pure XGB ensemble.
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
            {'max_depth': 2, 'n_estimators': 80, 'learning_rate': 0.02,
             'min_child_weight': 12},
        ],
        'feature_subset': (
            PERSISTENCE_FEATURES
            + LAG_FEATURES_SHORT
            + TEMPORAL_FEATURES_CORE
            + ['modis-sst', 'pdo']
        ),
        'ensemble_weights': (0.95, 0.05),
        'prediction_clip_q': 0.95,
        'prediction_clip_max': 80.0,
    },

    'Gold Beach': {
        # N=144, XGB R²=-0.094, Naive R²=-1.656.
        # Was -0.94 before Iter 1. Naive is terrible -- near-pure XGB ensemble.
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
        'ensemble_weights': (0.90, 0.10),
        'prediction_clip_q': 0.95,
        'prediction_clip_max': None,
    },

    'Newport': {
        # N=142, XGB R²=-0.163, Naive R²=-0.287.
        # Was -0.28 before Iter 1. Both struggle -- XGB slightly less bad.
        # Tightened regularization and reduced rolling features for Iter 4.
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
        'ensemble_weights': (0.60, 0.40),
        'prediction_clip_q': 0.98,
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
