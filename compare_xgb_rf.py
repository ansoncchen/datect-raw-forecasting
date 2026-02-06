#!/usr/bin/env python3
"""
XGBoost vs Random Forest Comparison
====================================
Runs both models through the same validation pipeline and compares results.
Uses identical data splits, features, and preprocessing for a fair comparison.
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

import config
from forecasting.model_factory import build_xgb_regressor
from forecasting.raw_data_forecaster import (
    aggregate_raw_to_weekly,
    build_raw_feature_frame,
    get_last_known_raw_da,
    get_site_test_row,
    get_site_training_frame,
    recompute_test_row_persistence_features,
)
from forecasting.per_site_models import (
    apply_site_xgb_params,
    compute_site_drop_cols,
    get_site_clip_params,
    get_site_ensemble_weights,
)

# Import data loading and temporal features from the main script
from validate_on_raw_data import (
    load_raw_da_measurements,
    load_processed_data,
    add_temporal_features,
    create_transformer,
    FORECAST_HORIZON_DAYS,
    MIN_TRAINING_SAMPLES,
    SPIKE_THRESHOLD,
    RANDOM_SEED,
    PREDICTION_CLIP_Q,
    USE_LOG_TARGET,
    USE_PER_SITE_MODELS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

ENABLE_PARALLEL = True
N_JOBS = -1

# Random Forest default parameters (comparable to XGBoost defaults)
RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "max_features": 0.85,
    "n_jobs": 1,
}


# =============================================================================
# SINGLE-SAMPLE VALIDATION (both models)
# =============================================================================

def run_single_comparison(raw_measurement, feature_frame, xgb_params):
    """
    Run a single validation point with both XGBoost and Random Forest.
    Returns a dict with predictions from both models.
    """
    test_date = raw_measurement['date']
    site = raw_measurement['site']
    actual_da = raw_measurement['da_raw']

    anchor_date = test_date - pd.Timedelta(days=FORECAST_HORIZON_DAYS)

    train_data = get_site_training_frame(feature_frame, site, anchor_date, MIN_TRAINING_SAMPLES)
    if train_data is None:
        return None

    test_row = get_site_test_row(feature_frame, site, test_date, anchor_date, max_date_diff_days=28)
    if test_row is None:
        return None

    test_row = recompute_test_row_persistence_features(test_row, train_data, SPIKE_THRESHOLD)

    train_data = add_temporal_features(train_data)
    test_row = add_temporal_features(test_row)

    drop_cols = ['date', 'site', 'da_raw', 'da',
                 'lat', 'lon', 'weeks_since_last_raw', 'is_bloom_season', 'quarter', 'da_raw_lag_52']

    if USE_PER_SITE_MODELS:
        drop_cols = compute_site_drop_cols(drop_cols, train_data.columns.tolist(), site)

    try:
        transformer, X_train = create_transformer(train_data, drop_cols)
        y_train_raw = train_data['da_raw'].astype(float)
        y_train = y_train_raw.copy()
        if USE_LOG_TARGET:
            y_train = np.log1p(y_train)

        X_train_processed = transformer.fit_transform(X_train)

        X_test = test_row.drop(columns=drop_cols, errors='ignore')
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        X_test_processed = transformer.transform(X_test)

        # Determine clip params
        if USE_PER_SITE_MODELS:
            site_clip_q, site_clip_max = get_site_clip_params(site)
            clip_q = site_clip_q if site_clip_q is not None else PREDICTION_CLIP_Q
        else:
            clip_q = PREDICTION_CLIP_Q
            site_clip_max = None

        def _postprocess(value):
            if USE_LOG_TARGET:
                value = np.expm1(value)
            value = max(0.0, value)
            if clip_q is not None:
                clip_max = float(np.quantile(train_data['da_raw'], clip_q))
                value = min(value, clip_max)
            if site_clip_max is not None:
                value = min(value, site_clip_max)
            return float(value)

        # --- XGBoost ---
        if USE_PER_SITE_MODELS:
            effective_xgb_params = apply_site_xgb_params(xgb_params, site)
        else:
            effective_xgb_params = xgb_params

        xgb_model = build_xgb_regressor(effective_xgb_params)
        try:
            if len(X_train_processed) > 15:
                val_split = int(0.8 * len(X_train_processed))
                xgb_model.fit(
                    X_train_processed[:val_split], y_train[:val_split],
                    eval_set=[(X_train_processed[val_split:], y_train[val_split:])],
                    verbose=False,
                )
            else:
                xgb_model.fit(X_train_processed, y_train)
        except Exception:
            xgb_model.fit(X_train_processed, y_train)

        xgb_raw = float(xgb_model.predict(X_test_processed)[0])
        xgb_pred = _postprocess(xgb_raw)

        # --- Random Forest ---
        rf_model = RandomForestRegressor(
            n_estimators=RF_PARAMS["n_estimators"],
            max_depth=RF_PARAMS["max_depth"],
            min_samples_split=RF_PARAMS["min_samples_split"],
            min_samples_leaf=RF_PARAMS["min_samples_leaf"],
            max_features=RF_PARAMS["max_features"],
            n_jobs=RF_PARAMS["n_jobs"],
            random_state=RANDOM_SEED,
        )
        rf_model.fit(X_train_processed, y_train)
        rf_raw = float(rf_model.predict(X_test_processed)[0])
        rf_pred = _postprocess(rf_raw)

        # --- Naive baseline ---
        naive_prediction = get_last_known_raw_da(train_data)
        if naive_prediction is None:
            return None

        # --- Feature importance ---
        xgb_importance = {}
        rf_importance = {}
        try:
            feature_names = X_train.columns.tolist()
            if hasattr(xgb_model, 'feature_importances_'):
                xgb_importance = dict(zip(feature_names, xgb_model.feature_importances_))
            rf_importance = dict(zip(feature_names, rf_model.feature_importances_))
        except Exception:
            pass

        return {
            'test_date': test_date,
            'anchor_date': anchor_date,
            'site': site,
            'actual_da_raw': actual_da,
            'xgb_predicted': xgb_pred,
            'rf_predicted': rf_pred,
            'naive_prediction': naive_prediction,
            'training_samples': len(train_data),
            'xgb_importance': xgb_importance,
            'rf_importance': rf_importance,
        }

    except Exception as e:
        return None


# =============================================================================
# METRICS
# =============================================================================

def print_comparison_metrics(results_df):
    """Print side-by-side comparison metrics."""
    actual = results_df['actual_da_raw'].values
    xgb = results_df['xgb_predicted'].values
    rf = results_df['rf_predicted'].values
    naive = results_df['naive_prediction'].values

    # Ensemble predictions (per-site weights)
    xgb_ensemble = []
    rf_ensemble = []
    for _, row in results_df.iterrows():
        w_xgb, w_naive = get_site_ensemble_weights(row['site'])
        xgb_ensemble.append(w_xgb * row['xgb_predicted'] + w_naive * row['naive_prediction'])
        rf_ensemble.append(w_xgb * row['rf_predicted'] + w_naive * row['naive_prediction'])
    xgb_ensemble = np.array(xgb_ensemble)
    rf_ensemble = np.array(rf_ensemble)

    def metrics(y_true, y_pred):
        return {
            'R²': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'Corr': np.corrcoef(y_true, y_pred)[0, 1],
        }

    def spike_metrics(y_true, y_pred):
        actual_spike = (y_true > SPIKE_THRESHOLD).astype(int)
        pred_spike = (y_pred > SPIKE_THRESHOLD).astype(int)
        if actual_spike.sum() == 0:
            return {'Precision': 0, 'Recall': 0, 'F1': 0}
        return {
            'Precision': precision_score(actual_spike, pred_spike, zero_division=0),
            'Recall': recall_score(actual_spike, pred_spike, zero_division=0),
            'F1': f1_score(actual_spike, pred_spike, zero_division=0),
        }

    xgb_m = metrics(actual, xgb)
    rf_m = metrics(actual, rf)
    naive_m = metrics(actual, naive)
    xgb_ens_m = metrics(actual, xgb_ensemble)
    rf_ens_m = metrics(actual, rf_ensemble)

    xgb_s = spike_metrics(actual, xgb)
    rf_s = spike_metrics(actual, rf)
    naive_s = spike_metrics(actual, naive)
    xgb_ens_s = spike_metrics(actual, xgb_ensemble)
    rf_ens_s = spike_metrics(actual, rf_ensemble)

    print(f"\n{'='*80}")
    print("XGBOOST vs RANDOM FOREST COMPARISON")
    print(f"{'='*80}")
    print(f"  Total predictions: {len(results_df)}")
    n_spikes = (actual > SPIKE_THRESHOLD).sum()
    print(f"  Actual spikes: {n_spikes} / {len(actual)} ({100*n_spikes/len(actual):.1f}%)")

    print(f"\n{'─'*80}")
    print(f"  {'Metric':<12} {'XGBoost':>10} {'RandomForest':>14} {'Naive':>10} {'XGB+Naive':>12} {'RF+Naive':>12}")
    print(f"{'─'*80}")
    for key in ['R²', 'MAE', 'RMSE', 'Corr']:
        fmt = '.4f' if key in ('R²', 'Corr') else '.2f'
        print(f"  {key:<12} {xgb_m[key]:>10{fmt}} {rf_m[key]:>14{fmt}} {naive_m[key]:>10{fmt}} {xgb_ens_m[key]:>12{fmt}} {rf_ens_m[key]:>12{fmt}}")

    print(f"\n{'─'*80}")
    print(f"  SPIKE DETECTION (threshold: {SPIKE_THRESHOLD} μg/g)")
    print(f"{'─'*80}")
    for key in ['Precision', 'Recall', 'F1']:
        print(f"  {key:<12} {xgb_s[key]:>10.4f} {rf_s[key]:>14.4f} {naive_s[key]:>10.4f} {xgb_ens_s[key]:>12.4f} {rf_ens_s[key]:>12.4f}")

    # Site-specific breakdown
    print(f"\n{'='*80}")
    print("SITE-SPECIFIC COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Site':<20} {'N':>4} {'XGB R²':>8} {'RF R²':>8} {'Naive R²':>9} {'XGB MAE':>8} {'RF MAE':>8} {'Naive MAE':>10}")
    print(f"  {'─'*86}")

    site_results = {}
    for site in sorted(results_df['site'].unique()):
        sd = results_df[results_df['site'] == site]
        sa = sd['actual_da_raw'].values
        sx = sd['xgb_predicted'].values
        sr = sd['rf_predicted'].values
        sn = sd['naive_prediction'].values

        if len(sa) >= 5:
            xr2 = r2_score(sa, sx)
            rr2 = r2_score(sa, sr)
            nr2 = r2_score(sa, sn)
            xmae = mean_absolute_error(sa, sx)
            rmae = mean_absolute_error(sa, sr)
            nmae = mean_absolute_error(sa, sn)
            print(f"  {site:<20} {len(sd):>4} {xr2:>+8.3f} {rr2:>+8.3f} {nr2:>+9.3f} {xmae:>8.2f} {rmae:>8.2f} {nmae:>10.2f}")
            site_results[site] = {
                'n': len(sd), 'xgb_r2': xr2, 'rf_r2': rr2, 'naive_r2': nr2,
                'xgb_mae': xmae, 'rf_mae': rmae, 'naive_mae': nmae,
            }
        else:
            print(f"  {site:<20} {len(sd):>4} {'(insufficient samples)':>60}")

    # Winner summary
    print(f"\n{'='*80}")
    print("WINNER SUMMARY (by site)")
    print(f"{'='*80}")
    xgb_wins_r2 = 0
    rf_wins_r2 = 0
    xgb_wins_mae = 0
    rf_wins_mae = 0
    for site, sm in site_results.items():
        r2_winner = "XGB" if sm['xgb_r2'] > sm['rf_r2'] else "RF"
        mae_winner = "XGB" if sm['xgb_mae'] < sm['rf_mae'] else "RF"
        if r2_winner == "XGB":
            xgb_wins_r2 += 1
        else:
            rf_wins_r2 += 1
        if mae_winner == "XGB":
            xgb_wins_mae += 1
        else:
            rf_wins_mae += 1
        r2_diff = sm['xgb_r2'] - sm['rf_r2']
        mae_diff = sm['xgb_mae'] - sm['rf_mae']
        print(f"  {site:<20}  R² winner: {r2_winner:<4} (diff: {r2_diff:+.3f})   MAE winner: {mae_winner:<4} (diff: {mae_diff:+.2f})")

    print(f"\n  R² wins:  XGBoost={xgb_wins_r2}, RandomForest={rf_wins_r2}")
    print(f"  MAE wins: XGBoost={xgb_wins_mae}, RandomForest={rf_wins_mae}")

    return {
        'xgb': xgb_m, 'rf': rf_m, 'naive': naive_m,
        'xgb_ens': xgb_ens_m, 'rf_ens': rf_ens_m,
        'xgb_spike': xgb_s, 'rf_spike': rf_s, 'naive_spike': naive_s,
        'site_results': site_results,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80)
    print("XGBOOST vs RANDOM FOREST COMPARISON")
    print("="*80)

    raw_data = load_raw_da_measurements()
    processed_data = load_processed_data()

    # Prepare feature frame
    raw_weekly = aggregate_raw_to_weekly(raw_data)
    feature_frame = build_raw_feature_frame(processed_data, raw_weekly)

    # Select test samples (same logic as validate_on_raw_data.py)
    min_test_date = pd.Timestamp("2003-01-01")
    candidate_raw = raw_data[raw_data['date'] >= min_test_date].copy()
    site_total_counts = raw_data.groupby("site")["date"].size().to_dict()

    valid_for_testing = []
    for _, row in candidate_raw.iterrows():
        anchor_date = row['date'] - pd.Timedelta(days=FORECAST_HORIZON_DAYS)
        site = row["site"]
        total_site_raw = site_total_counts.get(site, 0)
        if total_site_raw == 0:
            continue
        min_required = max(int(np.ceil(0.33 * total_site_raw)), MIN_TRAINING_SAMPLES)
        site_raw_count = len(raw_data[(raw_data["site"] == site) & (raw_data["date"] <= anchor_date)])
        if site_raw_count < min_required:
            continue
        site_history = feature_frame[
            (feature_frame['site'] == site) &
            (feature_frame['date'] <= anchor_date) &
            (feature_frame['da_raw'].notna())
        ]
        site_future = feature_frame[
            (feature_frame['site'] == site) &
            (feature_frame['date'] > anchor_date)
        ]
        if len(site_history) >= MIN_TRAINING_SAMPLES and len(site_future) > 0:
            valid_for_testing.append(row)

    valid_for_testing = pd.DataFrame(valid_for_testing)
    print(f"\nValid test candidates: {len(valid_for_testing)}")

    # Per-site sampling (~20%)
    rng = np.random.RandomState(RANDOM_SEED)
    sampled_rows = []
    for site, site_df in valid_for_testing.groupby("site"):
        site_df = site_df.sort_values("date")
        n_candidates = len(site_df)
        if n_candidates == 0:
            continue
        total_raw = site_total_counts.get(site, n_candidates)
        target = min(int(np.ceil(0.2 * total_raw)), n_candidates)
        if target <= 0:
            continue
        indices = rng.choice(n_candidates, size=target, replace=False)
        sampled_rows.append(site_df.iloc[indices])

    test_samples = pd.concat(sampled_rows, ignore_index=True)
    print(f"Test samples: {len(test_samples)}")

    # XGBoost base params
    base_params = {
        'n_estimators': config.XGB_REGRESSION_PARAMS.get('n_estimators', 400),
        'max_depth': config.XGB_REGRESSION_PARAMS.get('max_depth', 6),
        'learning_rate': config.XGB_REGRESSION_PARAMS.get('learning_rate', 0.05),
        'subsample': config.XGB_REGRESSION_PARAMS.get('subsample', 0.85),
        'colsample_bytree': config.XGB_REGRESSION_PARAMS.get('colsample_bytree', 0.85),
        'reg_alpha': config.XGB_REGRESSION_PARAMS.get('reg_alpha', 0.1),
        'reg_lambda': config.XGB_REGRESSION_PARAMS.get('reg_lambda', 1.0),
        'tree_method': 'hist',
        'n_jobs': 1,
    }

    sample_rows = [
        {'date': row['date'], 'site': row['site'], 'da_raw': row['da_raw']}
        for _, row in test_samples.iterrows()
    ]

    print(f"\nRunning comparison on {len(sample_rows)} samples...")
    print(f"  XGBoost params: {base_params}")
    print(f"  RF params: {RF_PARAMS}")

    start = time.time()
    pbar = tqdm(sample_rows, desc="Comparing XGB vs RF", unit="sample")

    if ENABLE_PARALLEL:
        results = Parallel(n_jobs=N_JOBS)(
            delayed(run_single_comparison)(row, feature_frame, base_params)
            for row in pbar
        )
    else:
        results = [run_single_comparison(row, feature_frame, base_params) for row in pbar]

    results = [r for r in results if r is not None]
    elapsed = time.time() - start
    print(f"\nCompleted {len(results)} / {len(sample_rows)} predictions in {elapsed:.1f}s")

    results_df = pd.DataFrame(results)

    # Print comparison
    comparison = print_comparison_metrics(results_df)

    # Save results
    output_dir = "./raw_validation_plots"
    os.makedirs(output_dir, exist_ok=True)
    save_cols = [c for c in results_df.columns if c not in ('xgb_importance', 'rf_importance')]
    results_df[save_cols].to_csv(os.path.join(output_dir, "xgb_vs_rf_comparison.csv"), index=False)
    print(f"\nResults saved to {output_dir}/xgb_vs_rf_comparison.csv")


if __name__ == "__main__":
    main()
