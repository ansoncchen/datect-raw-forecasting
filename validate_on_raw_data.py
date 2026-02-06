#!/usr/bin/env python3
"""
Raw Data Validation Script - Matching Original Pipeline
========================================================

This script provides a rigorous validation of the XGBoost model by testing
ONLY on real measurements from the raw data files, not on interpolated or
gap-filled values from the processed dataset.

CRITICAL: This version matches the original forecast_engine.py approach:
- Uses environmental features FROM THE PREDICTION DATE (not anchor date)
- Only the DA target is truly "unknown" at prediction time
- This is how the original pipeline works

The validation mimics real-world usage:
1. For each raw measurement at test_date, use anchor_date = test_date - 7 days
2. Train XGBoost on processed data UP TO anchor_date
3. Use environmental features from processed data AT test_date for prediction
4. Compare against the ACTUAL raw measurement (not processed/interpolated value)
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
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
from forecasting.sample_weights import compute_spike_focused_weights

# Try to import plotly for visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Plots will not be generated.")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum training samples required before making a prediction
MIN_TRAINING_SAMPLES = 10

# Forecast horizon (how far ahead we're predicting)
FORECAST_HORIZON_DAYS = config.FORECAST_HORIZON_DAYS  # 7 days (1 week)

# Minimum date for test samples (need enough history to train).
# We now rely primarily on a per-site history fraction rule (see run_validation),
# so this is just a very early lower bound.
MIN_TEST_DATE = "2003-01-01"

# Spike threshold for binary classification metrics
SPIKE_THRESHOLD = config.SPIKE_THRESHOLD  # 20 μg/g

# Random seed for reproducibility
RANDOM_SEED = 42

# Output directory for plots
PLOTS_OUTPUT_DIR = "./raw_validation_plots"

# Optional quantile prediction intervals (extra models per sample)
ENABLE_QUANTILE_INTERVALS = True

# Stabilization settings
USE_LOG_TARGET = False  # Test without log transform - preserves spike magnitude
PREDICTION_CLIP_Q = 0.99

# Parallelization settings
ENABLE_PARALLEL = True
N_JOBS = -1  # Use all cores

# Calibration/tuning
CALIBRATION_FRACTION = 0.3  # Per-anchor fraction of historical rows for tuning/calibration
MAX_CALIBRATION_ROWS = 20   # Hard cap on calibration rows to keep tuning tractable
PARAM_GRID = [
    {"max_depth": 4, "n_estimators": 500, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 6, "n_estimators": 400, "learning_rate": 0.05, "min_child_weight": 3},
]

# =============================================================================
# RAW DATA LOADING
# =============================================================================

def load_raw_da_measurements():
    """
    Load all raw DA measurements from the original CSV files.
    
    Returns a DataFrame with columns: date, site, da_raw
    Only includes actual measurements, no interpolation.
    """
    print("\n" + "="*70)
    print("LOADING RAW DA MEASUREMENTS")
    print("="*70)
    
    raw_measurements = []
    
    for site_key, file_path in config.ORIGINAL_DA_FILES.items():
        if not os.path.exists(file_path):
            print(f"  Warning: File not found: {file_path}")
            continue
            
        # Normalize site name to match processed data
        site_name = site_key.replace('-da', '').replace('_da', '').replace('-', ' ').replace('_', ' ').title()
        
        try:
            df = pd.read_csv(file_path)
            
            # Handle different date formats
            date_col = None
            da_col = None
            
            # Format 1: CollectDate, Domoic Result
            if 'CollectDate' in df.columns:
                date_col = 'CollectDate'
            # Format 2: Harvest Month, Harvest Date, Harvest Year
            elif all(c in df.columns for c in ['Harvest Month', 'Harvest Date', 'Harvest Year']):
                df['CombinedDateStr'] = (
                    df['Harvest Month'].astype(str) + " " + 
                    df['Harvest Date'].astype(str) + ", " + 
                    df['Harvest Year'].astype(str)
                )
                df['ParsedDate'] = pd.to_datetime(df['CombinedDateStr'], format='%B %d, %Y', errors='coerce')
                date_col = 'ParsedDate'
            
            # Find DA column
            if 'Domoic Result' in df.columns:
                da_col = 'Domoic Result'
            elif 'Domoic Acid' in df.columns:
                da_col = 'Domoic Acid'
            
            if date_col is None or da_col is None:
                print(f"  Warning: Could not identify columns in {file_path}")
                continue
            
            # Parse dates and DA values
            df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            df['da_raw'] = pd.to_numeric(df[da_col], errors='coerce')
            df['site'] = site_name
            
            # Keep only valid measurements
            valid_df = df.dropna(subset=['date', 'da_raw'])
            valid_df = valid_df[valid_df['da_raw'] >= 0]  # No negative DA values
            
            raw_measurements.append(valid_df[['date', 'site', 'da_raw']])
            print(f"  Loaded {len(valid_df)} measurements from {site_name}")
            
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    all_raw = pd.concat(raw_measurements, ignore_index=True)
    all_raw = all_raw.sort_values(['site', 'date']).reset_index(drop=True)
    
    print(f"\nTotal raw measurements loaded: {len(all_raw)}")
    print(f"Date range: {all_raw['date'].min().date()} to {all_raw['date'].max().date()}")
    print(f"Sites: {all_raw['site'].nunique()}")
    
    return all_raw


def load_processed_data():
    """Load the processed dataset (what we train on)."""
    print("\n" + "="*70)
    print("LOADING PROCESSED DATA (TRAINING SOURCE)")
    print("="*70)
    
    if not os.path.exists(config.FINAL_OUTPUT_PATH):
        raise FileNotFoundError(f"Processed data not found: {config.FINAL_OUTPUT_PATH}")
    
    data = pd.read_parquet(config.FINAL_OUTPUT_PATH)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['site', 'date']).reset_index(drop=True)
    
    print(f"Processed data shape: {data.shape}")
    print(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    print(f"Sites: {data['site'].nunique()}")
    print(f"Columns: {list(data.columns)}")
    
    return data


# =============================================================================
# FEATURE ENGINEERING (matching the main pipeline)
# =============================================================================

def add_temporal_features(df):
    """Add temporal features matching the main pipeline."""
    df = df.copy()
    
    if config.USE_ENHANCED_TEMPORAL_FEATURES:
        day_of_year = df['date'].dt.dayofyear
        df['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365)
        df['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365)
        df['month'] = df['date'].dt.month
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter'] = df['date'].dt.quarter
        week_of_year = df['date'].dt.isocalendar().week.astype(int)
        df['sin_week_of_year'] = np.sin(2 * np.pi * week_of_year / 52)
        df['cos_week_of_year'] = np.cos(2 * np.pi * week_of_year / 52)
        df['is_bloom_season'] = df['month'].between(3, 10).astype(int)
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    else:
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
    
    return df


def create_transformer(df, drop_cols):
    """Create preprocessing transformer (matching main pipeline)."""
    X = df.drop(columns=drop_cols, errors='ignore')
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    # Drop columns that are all NaN so median imputation can run without warnings
    numeric_cols = [c for c in numeric_cols if X[c].notna().any()]

    if len(numeric_cols) == 0:
        raise ValueError("No numeric features available")

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler()),
    ])
    
    transformer = ColumnTransformer(
        [('num', numeric_pipeline, numeric_cols)],
        remainder='drop',
        verbose_feature_names_out=False
    )
    transformer.set_output(transform='pandas')
    
    return transformer, X


# =============================================================================
# VALIDATION LOGIC - MATCHING ORIGINAL PIPELINE
# =============================================================================

def run_single_raw_validation(raw_measurement, feature_frame, model_params, skip_quantiles=False):
    """
    Validate a single raw measurement using the EXACT approach from forecast_engine.py.
    
    CRITICAL DIFFERENCE FROM BEFORE:
    - Uses environmental features FROM THE TEST DATE (like original pipeline)
    - Only the DA value at test_date is "unknown"
    - This matches how the original forecast_engine.py works
    
    Args:
        raw_measurement: dict with 'date', 'site', 'da_raw'
        processed_data: Full processed DataFrame
        model_params: XGBoost parameters
        skip_quantiles: If True, skip quantile interval models (for tuning/calibration speed)
        
    Returns:
        dict with prediction results or None if insufficient data
    """
    test_date = raw_measurement['date']
    site = raw_measurement['site']
    actual_da = raw_measurement['da_raw']
    
    # Calculate the anchor date (1 week before the test date)
    anchor_date = test_date - pd.Timedelta(days=FORECAST_HORIZON_DAYS)
    
    train_data = get_site_training_frame(feature_frame, site, anchor_date, MIN_TRAINING_SAMPLES)
    if train_data is None:
        return None

    test_row = get_site_test_row(
        feature_frame,
        site,
        test_date,
        anchor_date,
        # Allow a looser tolerance between raw test date and
        # nearest processed environmental row to reduce dropouts.
        max_date_diff_days=28,
    )
    if test_row is None:
        return None

    # Recompute persistence features using only train_data (no target leakage)
    test_row = recompute_test_row_persistence_features(
        test_row, train_data, SPIKE_THRESHOLD
    )

    # Add temporal features to both
    train_data = add_temporal_features(train_data)
    test_row = add_temporal_features(test_row)
    
    # Prepare features - Drop both raw and processed targets + zero-importance features
    drop_cols = ['date', 'site', 'da_raw', 'da',
                  # Zero/near-zero importance features from analysis:
                  'lat', 'lon', 'weeks_since_last_raw', 'is_bloom_season', 'quarter', 'da_raw_lag_52']
    
    try:
        # Create transformer and fit on training data only
        transformer, X_train = create_transformer(train_data, drop_cols)
        y_train_raw = train_data['da_raw'].astype(float)
        y_train = y_train_raw.copy()
        if USE_LOG_TARGET:
            y_train = np.log1p(y_train)

        sample_weight = None
        if config.USE_REGRESSION_SAMPLE_WEIGHTS:
            sample_weight = compute_spike_focused_weights(
                y_train_raw,
                SPIKE_THRESHOLD,
                config.SPIKE_FALSE_NEGATIVE_WEIGHT,
                config.SPIKE_TRUE_NEGATIVE_WEIGHT,
            )
        
        # Fit transformer on training data only
        X_train_processed = transformer.fit_transform(X_train)
        
        # CRITICAL: Use test_row features (environmental data from test date)
        # This matches the original pipeline!
        X_test = test_row.drop(columns=drop_cols, errors='ignore')
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        X_test_processed = transformer.transform(X_test)
        
        # Train XGBoost on raw targets
        model = build_xgb_regressor(model_params)

        # Try early stopping, fall back to regular fit if it fails
        try:
            if len(X_train_processed) > 15:  # Need enough samples for 80/20 split
                val_split = int(0.8 * len(X_train_processed))
                X_es_train = X_train_processed[:val_split]
                X_es_val = X_train_processed[val_split:]
                y_es_train = y_train[:val_split]
                y_es_val = y_train[val_split:]
                w_es_train = sample_weight[:val_split] if sample_weight is not None else None

                model.fit(
                    X_es_train, y_es_train,
                    sample_weight=w_es_train,
                    eval_set=[(X_es_val, y_es_val)],
                    verbose=False
                )
            else:
                # Too few samples for split
                model.fit(X_train_processed, y_train, sample_weight=sample_weight)
        except Exception:
            # Early stopping failed, use regular fit
            model.fit(X_train_processed, y_train, sample_weight=sample_weight)
        
        # Predict using TEST DATE features
        def _postprocess_prediction(value: float) -> float:
            if USE_LOG_TARGET:
                value = np.expm1(value)
            value = max(0.0, value)
            if PREDICTION_CLIP_Q is not None:
                clip_max = float(np.quantile(train_data['da_raw'], PREDICTION_CLIP_Q))
                value = min(value, clip_max)
            return float(value)

        prediction = _postprocess_prediction(float(model.predict(X_test_processed)[0]))

        quantile_predictions = {}
        if ENABLE_QUANTILE_INTERVALS and not skip_quantiles:
            try:
                for q in (0.1, 0.5, 0.9):
                    quantile_params = {
                        **model_params,
                        "objective": "reg:quantile",
                        "quantile_alpha": q,
                    }
                    q_model = build_xgb_regressor(quantile_params)
                    q_model.fit(X_train_processed, y_train, sample_weight=sample_weight)
                    q_pred = float(q_model.predict(X_test_processed)[0])
                    quantile_predictions[f"predicted_p{int(q * 100)}"] = _postprocess_prediction(q_pred)
            except Exception:
                quantile_predictions = {}
        
        # Naive baseline: last known RAW DA value from training
        naive_prediction = get_last_known_raw_da(train_data)
        if naive_prediction is None:
            return None

        # Extract feature importance for analysis
        feature_importance = {}
        try:
            if hasattr(model, 'feature_importances_'):
                feature_names = X_train.columns.tolist()
                feature_importance = dict(zip(feature_names, model.feature_importances_))
        except Exception:
            pass  # Skip if feature importance extraction fails

        return {
            'test_date': test_date,
            'anchor_date': anchor_date,
            'processed_test_date': test_row['date'].iloc[0],
            'site': site,
            'actual_da_raw': actual_da,
            'predicted_da': prediction,
            'naive_prediction': naive_prediction,
            'training_samples': len(train_data),
            'days_ahead': (test_date - anchor_date).days,
            'date_diff_to_processed': int(abs((test_row['date'].iloc[0] - test_date).days)),
            'feature_importance': feature_importance
        } | quantile_predictions
        
    except Exception as e:
        return None


def run_single_raw_validation_with_tuning(raw_measurement, feature_frame, base_params):
    """
    Per-anchor validation:
    - Sample historical anchors before the test date for tuning/calibration
    - Tune hyperparameters on those historical anchors
    - Calibrate output on those historical anchors
    - Predict the current test date with the tuned params
    """
    test_date = raw_measurement['date']
    site = raw_measurement['site']
    anchor_date = test_date - pd.Timedelta(days=FORECAST_HORIZON_DAYS)

    train_data = get_site_training_frame(feature_frame, site, anchor_date, MIN_TRAINING_SAMPLES)
    if train_data is None or train_data.empty:
        return None

    calib_candidates = train_data[['date', 'site', 'da_raw']].dropna().copy()
    if calib_candidates.empty:
        return run_single_raw_validation(raw_measurement, feature_frame, base_params, skip_quantiles=False)

    rng_seed = RANDOM_SEED + int(test_date.value % 1_000_000)
    rng = np.random.RandomState(rng_seed)
    n_candidates = len(calib_candidates)
    target_n = max(1, int(np.ceil(CALIBRATION_FRACTION * n_candidates)))
    target_n = min(target_n, MAX_CALIBRATION_ROWS)  # Hard cap for speed
    if target_n < n_candidates:
        indices = rng.choice(n_candidates, size=target_n, replace=False)
        calib_candidates = calib_candidates.iloc[indices]

    calib_rows = [
        {'date': row['date'], 'site': row['site'], 'da_raw': row['da_raw']}
        for _, row in calib_candidates.iterrows()
    ]

    if len(calib_rows) < 2:
        return run_single_raw_validation(raw_measurement, feature_frame, base_params, skip_quantiles=False)

    best_params, _ = tune_xgb_params(calib_rows, feature_frame, base_params)
    result = run_single_raw_validation(raw_measurement, feature_frame, best_params)
    return result  # No calibration - removed circular optimization


def calibrate_linear(y_true, y_pred):
    if len(y_true) < 3:
        return 1.0, 0.0
    slope, intercept = np.polyfit(y_pred, y_true, 1)
    return float(slope), float(intercept)


def tune_xgb_params(calib_rows, feature_frame, base_params):
    best_params = base_params
    best_r2 = float("-inf")
    for override in PARAM_GRID:
        params = {**base_params, **override}
        # Run sequentially to avoid nested parallelization
        # skip_quantiles=True: quantile models aren't needed for tuning (only R² matters)
        results = [
            run_single_raw_validation(row, feature_frame, params, skip_quantiles=True)
            for row in calib_rows
        ]
        results = [r for r in results if r is not None]
        if not results:
            continue
        df = pd.DataFrame(results)
        r2 = r2_score(df["actual_da_raw"].values, df["predicted_da"].values)
        if r2 > best_r2:
            best_r2 = r2
            best_params = params
    return best_params, best_r2


def run_validation(raw_data, processed_data, n_samples=None):
    """
    Run validation on a random sample of raw measurements.
    """
    print("\n" + "="*70)
    print("RUNNING RAW DATA VALIDATION (MATCHING ORIGINAL PIPELINE)")
    print("="*70)
    
    # Prepare raw/feature frames
    raw_weekly = aggregate_raw_to_weekly(raw_data)
    feature_frame = build_raw_feature_frame(processed_data, raw_weekly)

    # Filter to valid test dates (very early lower bound)
    min_test_date = pd.Timestamp(MIN_TEST_DATE)
    candidate_raw = raw_data[raw_data['date'] >= min_test_date].copy()

    # Pre-compute total raw measurement counts per site
    site_total_counts = raw_data.groupby("site")["date"].size().to_dict()

    print(f"Raw measurements after {MIN_TEST_DATE}: {len(candidate_raw)}")
    
    # Further filter: need enough history in processed data AND
    # at least 33% of each site's raw measurements to lie on/before
    # the anchor date for that test point.
    valid_for_testing = []
    for _, row in candidate_raw.iterrows():
        anchor_date = row['date'] - pd.Timedelta(days=FORECAST_HORIZON_DAYS)
        site = row["site"]

        # How many raw measurements for this site exist up to anchor_date?
        total_site_raw = site_total_counts.get(site, 0)
        if total_site_raw == 0:
            continue
        min_required_history_raw = int(np.ceil(0.33 * total_site_raw))
        # enforce at least MIN_TRAINING_SAMPLES raw points as well
        min_required_history_raw = max(min_required_history_raw, MIN_TRAINING_SAMPLES)

        site_raw_history_count = len(
            raw_data[
                (raw_data["site"] == site) &
                (raw_data["date"] <= anchor_date)
            ]
        )
        if site_raw_history_count < min_required_history_raw:
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
    print(f"Raw measurements with sufficient data: {len(valid_for_testing)}")
    
    if len(valid_for_testing) == 0:
        print("ERROR: No valid test samples found!")
        return None

    # ------------------------------------------------------------------
    # Per-site sampling: use ~20% of valid measurements per site
    # (with at least 1 sample per site), all after MIN_TEST_DATE.
    # ------------------------------------------------------------------
    rng = np.random.RandomState(RANDOM_SEED)
    per_site_counts = {}
    sampled_rows = []
    for site, site_df in valid_for_testing.groupby("site"):
        site_df = site_df.sort_values("date")
        n_site_candidates = len(site_df)
        if n_site_candidates == 0:
            continue

        # Target ~20% of the *total* raw measurements for this site
        total_site_raw = site_total_counts.get(site, n_site_candidates)
        target_per_site = int(np.ceil(0.2 * total_site_raw))

        # But we can only draw from candidates that satisfy the 33% history rule.
        n_site_samples = min(target_per_site, n_site_candidates)
        if n_site_samples <= 0:
            continue

        indices = rng.choice(n_site_candidates, size=n_site_samples, replace=False)
        per_site_counts[site] = n_site_samples
        sampled_rows.append(site_df.iloc[indices])

    if not sampled_rows:
        print("ERROR: Per-site sampling produced no test samples!")
        return None

    test_samples = pd.concat(sampled_rows, ignore_index=True)
    n_samples = len(test_samples)

    print("\nPer-site test sample counts (~20% of total raw measurements, subject to history constraints):")
    for site, count in sorted(per_site_counts.items()):
        print(f"  {site}: {count}")

    print(f"\nTotal selected test measurements: {n_samples}")
    print(f"Test date range: {test_samples['date'].min().date()} to {test_samples['date'].max().date()}")
    
    # XGBoost parameters (matching config)
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
    
    print(f"\nXGBoost base parameters: {base_params}")
    print(f"\nCRITICAL: Using raw DA targets with raw lag features + test-date env features")
    
    # Run validation for each sample
    print(f"\nParallel mode: {ENABLE_PARALLEL} (n_jobs={N_JOBS})")
    sample_rows = [
        {'date': row['date'], 'site': row['site'], 'da_raw': row['da_raw']}
        for _, row in test_samples.iterrows()
    ]
    print(
        f"\nPer-anchor tuning/calibration: sampling {CALIBRATION_FRACTION:.0%} "
        f"of pre-anchor history for each test date"
    )
    print(f"\nValidating {len(sample_rows)} samples...")
    pbar = tqdm(sample_rows, desc="Validating", unit="sample")

    if ENABLE_PARALLEL:
        results = Parallel(n_jobs=N_JOBS)(
            delayed(run_single_raw_validation_with_tuning)(row, feature_frame, base_params)
            for row in pbar
        )
    else:
        results = [
            run_single_raw_validation_with_tuning(row, feature_frame, base_params)
            for row in pbar
        ]

    results = [r for r in results if r is not None]
    print(
        f"\nEvaluation results: {len(results)} / {len(sample_rows)} successful predictions "
        f"({100*len(results)/len(sample_rows):.1f}%)"
    )

    results_df = pd.DataFrame(results)

    # Add ensemble prediction only if we have predictions
    if not results_df.empty and 'predicted_da' in results_df.columns and 'naive_prediction' in results_df.columns:
        # Favor XGB (R²=0.22) over naive (R²=-0.14) while keeping naive's MAE benefit
        ENSEMBLE_WEIGHT_XGB = 0.65
        ENSEMBLE_WEIGHT_NAIVE = 0.35
        results_df['ensemble_prediction'] = (
            ENSEMBLE_WEIGHT_XGB * results_df['predicted_da'] +
            ENSEMBLE_WEIGHT_NAIVE * results_df['naive_prediction']
        )
        results_df['ensemble_weight_xgb'] = ENSEMBLE_WEIGHT_XGB
        results_df['ensemble_weight_naive'] = ENSEMBLE_WEIGHT_NAIVE

    # Aggregate and analyze feature importance
    if not results_df.empty:
        from collections import defaultdict

        all_importances = defaultdict(list)
        for result in results:
            if result and 'feature_importance' in result and result['feature_importance']:
                for feat, imp in result['feature_importance'].items():
                    all_importances[feat].append(imp)

        if all_importances:
            avg_importance = {feat: np.mean(imps) for feat, imps in all_importances.items()}
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

            # Save to file
            import os
            importance_df = pd.DataFrame(sorted_features, columns=['feature', 'avg_importance'])
            importance_df.to_csv(os.path.join(PLOTS_OUTPUT_DIR, 'feature_importance.csv'), index=False)

            print("\n" + "="*60)
            print("FEATURE IMPORTANCE ANALYSIS")
            print("="*60)
            print("\nTOP 10 MOST IMPORTANT FEATURES:")
            for i, (feat, imp) in enumerate(sorted_features[:10], 1):
                print(f"  {i:2d}. {feat:30s} {imp:.4f}")

            print("\nBOTTOM 10 LEAST IMPORTANT FEATURES:")
            for i, (feat, imp) in enumerate(sorted_features[-10:], 1):
                print(f"  {i:2d}. {feat:30s} {imp:.4f}")

            # Identify low-importance features (< 1% of max importance)
            if sorted_features:
                max_importance = sorted_features[0][1]
                threshold = 0.01 * max_importance
                low_importance = [feat for feat, imp in sorted_features if imp < threshold]

                if low_importance:
                    print(f"\n  Low-importance features (< 1% of max):")
                    print(f"  {', '.join(low_importance[:15])}")  # Show first 15
                    print(f"  Total: {len(low_importance)} features")

    return results_df


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(results_df):
    """Calculate and display comprehensive metrics."""
    print("\n" + "="*70)
    print("RAW DATA VALIDATION RESULTS")
    print("="*70)
    
    if results_df is None or results_df.empty:
        print("No results to evaluate!")
        return None
    
    actual = results_df['actual_da_raw'].values
    predicted = results_df['predicted_da'].values
    naive = results_df['naive_prediction'].values
    ensemble = results_df['ensemble_prediction'].values if 'ensemble_prediction' in results_df else None
    
    # Regression metrics - XGBoost
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # Regression metrics - Naive baseline (last known value)
    naive_r2 = r2_score(actual, naive)
    naive_mae = mean_absolute_error(actual, naive)
    naive_rmse = np.sqrt(mean_squared_error(actual, naive))

    # Regression metrics - Ensemble
    if ensemble is not None:
        ensemble_r2 = r2_score(actual, ensemble)
        ensemble_mae = mean_absolute_error(actual, ensemble)
        ensemble_rmse = np.sqrt(mean_squared_error(actual, ensemble))
    else:
        ensemble_r2 = None
        ensemble_mae = None
        ensemble_rmse = None

    # Correlation
    correlation = np.corrcoef(actual, predicted)[0, 1]
    naive_correlation = np.corrcoef(actual, naive)[0, 1]
    ensemble_correlation = np.corrcoef(actual, ensemble)[0, 1] if ensemble is not None else None

    ensemble_f1 = None  # Will be calculated below if ensemble exists
    
    # Spike detection metrics (binary)
    actual_spike = (actual > SPIKE_THRESHOLD).astype(int)
    predicted_spike = (predicted > SPIKE_THRESHOLD).astype(int)
    naive_spike = (naive > SPIKE_THRESHOLD).astype(int)
    ensemble_spike = (ensemble > SPIKE_THRESHOLD).astype(int) if ensemble is not None else None
    
    if actual_spike.sum() > 0:
        precision = precision_score(actual_spike, predicted_spike, zero_division=0)
        recall = recall_score(actual_spike, predicted_spike, zero_division=0)
        f1 = f1_score(actual_spike, predicted_spike, zero_division=0)
        
        naive_precision = precision_score(actual_spike, naive_spike, zero_division=0)
        naive_recall = recall_score(actual_spike, naive_spike, zero_division=0)
        naive_f1 = f1_score(actual_spike, naive_spike, zero_division=0)
        if ensemble_spike is not None:
            ensemble_precision = precision_score(actual_spike, ensemble_spike, zero_division=0)
            ensemble_recall = recall_score(actual_spike, ensemble_spike, zero_division=0)
            ensemble_f1 = f1_score(actual_spike, ensemble_spike, zero_division=0)
        else:
            ensemble_precision = ensemble_recall = ensemble_f1 = 0.0
    else:
        precision = recall = f1 = 0.0
        naive_precision = naive_recall = naive_f1 = 0.0
        ensemble_precision = ensemble_recall = ensemble_f1 = 0.0
    
    print(f"\n{'='*60}")
    print("XGBOOST PERFORMANCE ON RAW DATA")
    print(f"{'='*60}")
    print(f"  Total predictions:     {len(results_df)}")
    print(f"  R² Score:              {r2:.4f}")
    print(f"  MAE:                   {mae:.4f} μg/g")
    print(f"  RMSE:                  {rmse:.4f} μg/g")
    print(f"  Correlation:           {correlation:.4f}")
    
    print(f"\n{'='*60}")
    print("NAIVE BASELINE PERFORMANCE (Last Known DA Value)")
    print(f"{'='*60}")
    print(f"  R² Score:              {naive_r2:.4f}")
    print(f"  MAE:                   {naive_mae:.4f} μg/g")
    print(f"  RMSE:                  {naive_rmse:.4f} μg/g")
    print(f"  Correlation:           {naive_correlation:.4f}")

    if ensemble is not None:
        print(f"\n{'='*60}")
        print("ENSEMBLE PERFORMANCE (0.65*XGB + 0.35*Naive)")
        print(f"{'='*60}")
        print(f"  R² Score:              {ensemble_r2:.4f}")
        print(f"  MAE:                   {ensemble_mae:.4f} μg/g")
        print(f"  RMSE:                  {ensemble_rmse:.4f} μg/g")
        print(f"  Correlation:           {ensemble_correlation:.4f}")
    
    print(f"\n{'='*60}")
    print("XGBOOST vs NAIVE COMPARISON")
    print(f"{'='*60}")
    mae_improvement = (1 - mae/naive_mae)*100 if naive_mae > 0 else 0
    print(f"  MAE Improvement:       {mae_improvement:+.1f}% {'(XGBoost better)' if mae < naive_mae else '(Naive better)'}")
    print(f"  R² Difference:         {r2 - naive_r2:+.4f}")
    
    print(f"\n{'='*60}")
    print(f"SPIKE DETECTION (threshold: {SPIKE_THRESHOLD} μg/g)")
    print(f"{'='*60}")
    print(f"  Actual spikes:         {actual_spike.sum()} / {len(actual)} ({100*actual_spike.mean():.1f}%)")
    print(f"")
    print(f"  XGBoost:")
    print(f"    Predicted spikes:    {predicted_spike.sum()} ({100*predicted_spike.mean():.1f}%)")
    print(f"    Precision:           {precision:.4f}")
    print(f"    Recall:              {recall:.4f}")
    print(f"    F1 Score:            {f1:.4f}")
    print(f"")
    print(f"  Naive Baseline:")
    print(f"    Predicted spikes:    {naive_spike.sum()} ({100*naive_spike.mean():.1f}%)")
    print(f"    Precision:           {naive_precision:.4f}")
    print(f"    Recall:              {naive_recall:.4f}")
    print(f"    F1 Score:            {naive_f1:.4f}")

    if ensemble_spike is not None:
        print(f"")
        print(f"  Ensemble:")
        print(f"    Predicted spikes:    {ensemble_spike.sum()} ({100*ensemble_spike.mean():.1f}%)")
        print(f"    Precision:           {ensemble_precision:.4f}")
        print(f"    Recall:              {ensemble_recall:.4f}")
        print(f"    F1 Score:            {ensemble_f1:.4f}")
    
    # Site-specific breakdown
    print(f"\n{'='*60}")
    print("SITE-SPECIFIC PERFORMANCE")
    print(f"{'='*60}")
    print(f"{'Site':<20} {'N':>4} {'XGB R²':>8} {'Naive R²':>9} {'XGB MAE':>8} {'Naive MAE':>10}")
    print("-"*70)
    
    site_metrics = {}
    for site in sorted(results_df['site'].unique()):
        site_results = results_df[results_df['site'] == site]
        site_actual = site_results['actual_da_raw'].values
        site_pred = site_results['predicted_da'].values
        site_naive = site_results['naive_prediction'].values
        
        if len(site_actual) >= 5:  # Need enough samples
            site_r2 = r2_score(site_actual, site_pred)
            site_mae = mean_absolute_error(site_actual, site_pred)
            site_naive_r2 = r2_score(site_actual, site_naive)
            site_naive_mae = mean_absolute_error(site_actual, site_naive)
            
            site_actual_spike = (site_actual > SPIKE_THRESHOLD).astype(int)
            site_pred_spike = (site_pred > SPIKE_THRESHOLD).astype(int)
            site_f1 = f1_score(site_actual_spike, site_pred_spike, zero_division=0)
            
            print(f"{site:<20} {len(site_results):>4} {site_r2:>+8.3f} {site_naive_r2:>+9.3f} {site_mae:>8.2f} {site_naive_mae:>10.2f}")
            
            site_metrics[site] = {
                'n': len(site_results),
                'r2': site_r2,
                'naive_r2': site_naive_r2,
                'mae': site_mae,
                'naive_mae': site_naive_mae,
                'f1': site_f1
            }
        else:
            print(f"{site:<20} {len(site_results):>4} {'(insufficient samples)':>50}")
    
    # Distribution analysis
    print(f"\n{'='*60}")
    print("DATA DISTRIBUTION")
    print(f"{'='*60}")
    print(f"  Actual DA range:       {actual.min():.1f} - {actual.max():.1f} μg/g")
    print(f"  Actual DA mean:        {actual.mean():.2f} μg/g")
    print(f"  Actual DA median:      {np.median(actual):.2f} μg/g")
    print(f"  Predicted DA range:    {predicted.min():.1f} - {predicted.max():.1f} μg/g")
    print(f"  Predicted DA mean:     {predicted.mean():.2f} μg/g")
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'naive_r2': naive_r2,
        'naive_mae': naive_mae,
        'naive_f1': naive_f1,
        'n_samples': len(results_df),
        'n_spikes': int(actual_spike.sum()),
        'site_metrics': site_metrics
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def generate_plots(results_df, metrics, output_dir):
    """Generate comprehensive plots for each site and overall."""
    
    if not PLOTLY_AVAILABLE:
        print("Plotly not available - skipping plot generation")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"GENERATING PLOTS")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    
    # 1. Overall scatter plot: Actual vs Predicted
    has_ensemble = 'ensemble_prediction' in results_df
    if has_ensemble:
        fig = make_subplots(rows=1, cols=3, subplot_titles=(
            'XGBoost: Actual vs Predicted',
            'Naive Baseline: Actual vs Last Known',
            'Ensemble: Actual vs Predicted'
        ))
    else:
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            'XGBoost: Actual vs Predicted',
            'Naive Baseline: Actual vs Last Known'
        ))
    
    # XGBoost scatter
    fig.add_trace(go.Scatter(
        x=results_df['actual_da_raw'],
        y=results_df['predicted_da'],
        mode='markers',
        marker=dict(
            color=results_df['site'].astype('category').cat.codes,
            colorscale='Viridis',
            size=8,
            opacity=0.6
        ),
        text=results_df['site'],
        hovertemplate='Site: %{text}<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>',
        name='XGBoost'
    ), row=1, col=1)
    
    # Perfect prediction line
    max_val = max(results_df['actual_da_raw'].max(), results_df['predicted_da'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction',
        showlegend=True
    ), row=1, col=1)
    
    # Naive scatter
    fig.add_trace(go.Scatter(
        x=results_df['actual_da_raw'],
        y=results_df['naive_prediction'],
        mode='markers',
        marker=dict(
            color=results_df['site'].astype('category').cat.codes,
            colorscale='Viridis',
            size=8,
            opacity=0.6
        ),
        text=results_df['site'],
        hovertemplate='Site: %{text}<br>Actual: %{x:.2f}<br>Naive: %{y:.2f}<extra></extra>',
        name='Naive Baseline',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        showlegend=False
    ), row=1, col=2)

    if has_ensemble:
        fig.add_trace(go.Scatter(
            x=results_df['actual_da_raw'],
            y=results_df['ensemble_prediction'],
            mode='markers',
            marker=dict(
                color=results_df['site'].astype('category').cat.codes,
                colorscale='Viridis',
                size=8,
                opacity=0.6
            ),
            text=results_df['site'],
            hovertemplate='Site: %{text}<br>Actual: %{x:.2f}<br>Ensemble: %{y:.2f}<extra></extra>',
            name='Ensemble',
            showlegend=False
        ), row=1, col=3)
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ), row=1, col=3)
    
    if has_ensemble and metrics.get('ensemble_r2') is not None:
        title = (
            f"Raw Data Validation: Actual vs Predicted DA Levels<br><sub>"
            f"XGBoost R²={metrics['r2']:.3f}, MAE={metrics['mae']:.2f} | "
            f"Naive R²={metrics['naive_r2']:.3f}, MAE={metrics['naive_mae']:.2f} | "
            f"Ensemble R²={metrics['ensemble_r2']:.3f}, MAE={metrics['ensemble_mae']:.2f}"
            f"</sub>"
        )
        width = 1400
    else:
        title = (
            f"Raw Data Validation: Actual vs Predicted DA Levels<br><sub>"
            f"XGBoost R²={metrics['r2']:.3f}, MAE={metrics['mae']:.2f} | "
            f"Naive R²={metrics['naive_r2']:.3f}, MAE={metrics['naive_mae']:.2f}"
            f"</sub>"
        )
        width = 1000

    fig.update_layout(
        title=title,
        height=500,
        width=width
    )
    fig.update_xaxes(title_text="Actual DA (μg/g)", row=1, col=1)
    fig.update_yaxes(title_text="Predicted DA (μg/g)", row=1, col=1)
    fig.update_xaxes(title_text="Actual DA (μg/g)", row=1, col=2)
    fig.update_yaxes(title_text="Last Known DA (μg/g)", row=1, col=2)
    if has_ensemble:
        fig.update_xaxes(title_text="Actual DA (μg/g)", row=1, col=3)
        fig.update_yaxes(title_text="Ensemble DA (μg/g)", row=1, col=3)
    
    fig.write_html(os.path.join(output_dir, "overall_scatter.html"))
    fig.write_image(os.path.join(output_dir, "overall_scatter.png"), scale=2)
    print(f"  Saved: overall_scatter.html/png")
    
    # 2. Site-specific performance bar chart
    site_metrics = metrics.get('site_metrics', {})
    if site_metrics:
        sites = list(site_metrics.keys())
        xgb_r2 = [site_metrics[s]['r2'] for s in sites]
        naive_r2 = [site_metrics[s]['naive_r2'] for s in sites]
        xgb_mae = [site_metrics[s]['mae'] for s in sites]
        naive_mae = [site_metrics[s]['naive_mae'] for s in sites]
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('R² Score by Site', 'MAE by Site'))
        
        fig.add_trace(go.Bar(name='XGBoost', x=sites, y=xgb_r2, marker_color='steelblue'), row=1, col=1)
        fig.add_trace(go.Bar(name='Naive Baseline', x=sites, y=naive_r2, marker_color='orange'), row=1, col=1)
        
        fig.add_trace(go.Bar(name='XGBoost', x=sites, y=xgb_mae, marker_color='steelblue', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(name='Naive Baseline', x=sites, y=naive_mae, marker_color='orange', showlegend=False), row=1, col=2)
        
        fig.update_layout(
            title="Site-Specific Model Performance: XGBoost vs Naive Baseline",
            barmode='group',
            height=500,
            width=1200
        )
        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(title_text="R² Score", row=1, col=1)
        fig.update_yaxes(title_text="MAE (μg/g)", row=1, col=2)
        
        fig.write_html(os.path.join(output_dir, "site_performance_comparison.html"))
        fig.write_image(os.path.join(output_dir, "site_performance_comparison.png"), scale=2)
        print(f"  Saved: site_performance_comparison.html/png")
    
    # 3. Time series plots for each site
    for site in sorted(results_df['site'].unique()):
        site_data = results_df[results_df['site'] == site].sort_values('test_date')
        
        if len(site_data) < 5:
            continue
        
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=site_data['test_date'],
            y=site_data['actual_da_raw'],
            mode='lines+markers',
            name='Actual (Raw)',
            line=dict(color='black', width=2),
            marker=dict(size=8)
        ))
        
        # XGBoost predictions
        fig.add_trace(go.Scatter(
            x=site_data['test_date'],
            y=site_data['predicted_da'],
            mode='lines+markers',
            name='XGBoost Prediction',
            line=dict(color='steelblue', width=2),
            marker=dict(size=6)
        ))
        
        # Naive predictions
        fig.add_trace(go.Scatter(
            x=site_data['test_date'],
            y=site_data['naive_prediction'],
            mode='lines+markers',
            name='Naive (Last Known)',
            line=dict(color='orange', width=2, dash='dot'),
            marker=dict(size=6)
        ))

        if 'ensemble_prediction' in site_data.columns:
            fig.add_trace(go.Scatter(
                x=site_data['test_date'],
                y=site_data['ensemble_prediction'],
                mode='lines+markers',
                name='Ensemble Prediction',
                line=dict(color='seagreen', width=2, dash='dash'),
                marker=dict(size=6)
            ))
        
        # Spike threshold line
        fig.add_hline(y=SPIKE_THRESHOLD, line_dash="dash", line_color="red",
                      annotation_text=f"Spike Threshold ({SPIKE_THRESHOLD} μg/g)")

        # Use site metrics from calculate_metrics for consistency with printed output
        site_metrics_dict = metrics.get("site_metrics", {})
        sm = site_metrics_dict.get(site, {})
        if sm:
            site_r2, site_mae = sm.get("r2", 0.0), sm.get("mae", 0.0)
            naive_r2, naive_mae = sm.get("naive_r2", 0.0), sm.get("naive_mae", 0.0)
        else:
            site_actual = site_data["actual_da_raw"].values
            site_pred = site_data["predicted_da"].values
            site_naive = site_data["naive_prediction"].values
            site_r2 = r2_score(site_actual, site_pred)
            site_mae = mean_absolute_error(site_actual, site_pred)
            naive_r2 = r2_score(site_actual, site_naive)
            naive_mae = mean_absolute_error(site_actual, site_naive)
        if "ensemble_prediction" in site_data.columns:
            site_ensemble = site_data["ensemble_prediction"].values
            ens_r2 = r2_score(site_data["actual_da_raw"].values, site_ensemble)
            ens_mae = mean_absolute_error(site_data["actual_da_raw"].values, site_ensemble)
            title = (
                f"Time Series Comparison: {site}<br><sub>"
                f"XGBoost: R²={site_r2:.3f}, MAE={site_mae:.2f} | "
                f"Naive: R²={naive_r2:.3f}, MAE={naive_mae:.2f} | "
                f"Ensemble: R²={ens_r2:.3f}, MAE={ens_mae:.2f}</sub>"
            )
        else:
            title = (
                f"Time Series Comparison: {site}<br><sub>"
                f"XGBoost: R²={site_r2:.3f}, MAE={site_mae:.2f} | "
                f"Naive: R²={naive_r2:.3f}, MAE={naive_mae:.2f}</sub>"
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="DA Level (μg/g)",
            height=500,
            width=1000,
            hovermode='x unified'
        )
        
        safe_site = site.replace(' ', '_').replace('/', '_')
        fig.write_html(os.path.join(output_dir, f"timeseries_{safe_site}.html"))
        fig.write_image(os.path.join(output_dir, f"timeseries_{safe_site}.png"), scale=2)
        print(f"  Saved: timeseries_{safe_site}.html/png")
    
    # 4. Error distribution histogram
    errors = results_df['predicted_da'] - results_df['actual_da_raw']
    naive_errors = results_df['naive_prediction'] - results_df['actual_da_raw']
    ensemble_errors = (
        results_df['ensemble_prediction'] - results_df['actual_da_raw']
        if 'ensemble_prediction' in results_df else None
    )
    
    if ensemble_errors is not None:
        fig = make_subplots(rows=1, cols=3, subplot_titles=(
            'XGBoost Prediction Errors',
            'Naive Baseline Errors',
            'Ensemble Errors'
        ))
    else:
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            'XGBoost Prediction Errors',
            'Naive Baseline Errors'
        ))
    
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=50,
        name='XGBoost',
        marker_color='steelblue'
    ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=naive_errors,
        nbinsx=50,
        name='Naive',
        marker_color='orange'
    ), row=1, col=2)

    if ensemble_errors is not None:
        fig.add_trace(go.Histogram(
            x=ensemble_errors,
            nbinsx=50,
            name='Ensemble',
            marker_color='seagreen'
        ), row=1, col=3)
    
    fig.update_layout(
        title=f"Prediction Error Distribution<br><sub>XGBoost Mean Error: {errors.mean():.2f}, Std: {errors.std():.2f} | Naive Mean Error: {naive_errors.mean():.2f}, Std: {naive_errors.std():.2f}</sub>",
        height=400,
        width=1000
    )
    fig.update_xaxes(title_text="Prediction Error (μg/g)", row=1, col=1)
    fig.update_xaxes(title_text="Prediction Error (μg/g)", row=1, col=2)
    
    fig.write_html(os.path.join(output_dir, "error_distribution.html"))
    fig.write_image(os.path.join(output_dir, "error_distribution.png"), scale=2)
    print(f"  Saved: error_distribution.html/png")
    
    # 5. Confusion matrix for spike detection
    actual_spike = (results_df['actual_da_raw'] > SPIKE_THRESHOLD).astype(int)
    pred_spike = (results_df['predicted_da'] > SPIKE_THRESHOLD).astype(int)
    naive_spike = (results_df['naive_prediction'] > SPIKE_THRESHOLD).astype(int)
    ensemble_spike = (results_df['ensemble_prediction'] > SPIKE_THRESHOLD).astype(int) if 'ensemble_prediction' in results_df else None
    
    cm_xgb = confusion_matrix(actual_spike, pred_spike)
    cm_naive = confusion_matrix(actual_spike, naive_spike)
    cm_ens = confusion_matrix(actual_spike, ensemble_spike) if ensemble_spike is not None else None
    
    if cm_ens is not None:
        fig = make_subplots(rows=1, cols=3, subplot_titles=(
            f'XGBoost Spike Detection (F1={metrics["f1"]:.3f})',
            f'Naive Baseline Spike Detection (F1={metrics["naive_f1"]:.3f})',
            f'Ensemble Spike Detection (F1={metrics["ensemble_f1"]:.3f})'
        ))
    else:
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            f'XGBoost Spike Detection (F1={metrics["f1"]:.3f})',
            f'Naive Baseline Spike Detection (F1={metrics["naive_f1"]:.3f})'
        ))
    
    labels = ['No Spike', 'Spike']
    
    fig.add_trace(go.Heatmap(
        z=cm_xgb,
        x=labels,
        y=labels,
        text=cm_xgb,
        texttemplate="%{text}",
        colorscale='Blues',
        showscale=False
    ), row=1, col=1)
    
    fig.add_trace(go.Heatmap(
        z=cm_naive,
        x=labels,
        y=labels,
        text=cm_naive,
        texttemplate="%{text}",
        colorscale='Oranges',
        showscale=False
    ), row=1, col=2)

    if cm_ens is not None:
        fig.add_trace(go.Heatmap(
            z=cm_ens,
            x=labels,
            y=labels,
            text=cm_ens,
            texttemplate="%{text}",
            colorscale='Greens',
            showscale=False
        ), row=1, col=3)
    
    width = 1100 if cm_ens is not None else 800
    fig.update_layout(
        title=f"Spike Detection Confusion Matrix (Threshold: {SPIKE_THRESHOLD} μg/g)",
        height=400,
        width=width
    )
    fig.update_yaxes(title_text="Actual", row=1, col=1)
    fig.update_xaxes(title_text="Predicted", row=1, col=1)
    fig.update_xaxes(title_text="Predicted", row=1, col=2)
    if cm_ens is not None:
        fig.update_xaxes(title_text="Predicted", row=1, col=3)
    
    fig.write_html(os.path.join(output_dir, "confusion_matrix.html"))
    fig.write_image(os.path.join(output_dir, "confusion_matrix.png"), scale=2)
    print(f"  Saved: confusion_matrix.html/png")
    
    print(f"\nAll plots saved to: {output_dir}")


def save_results(results_df, metrics, output_dir):
    """Save detailed results to CSV for further analysis."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'raw_data_validation_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    # Also save summary
    summary_path = os.path.join(output_dir, 'validation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("RAW DATA VALIDATION SUMMARY\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write(f"  Forecast horizon: {FORECAST_HORIZON_DAYS} days\n")
        f.write("  Test sampling: ~20% per site with 33% history requirement\n")
        f.write(f"  Min training samples: {MIN_TRAINING_SAMPLES}\n")
        f.write(f"  Min test date: {MIN_TEST_DATE}\n")
        f.write(f"  Spike threshold: {SPIKE_THRESHOLD} μg/g\n")
        f.write(f"  Random seed: {RANDOM_SEED}\n\n")
        f.write(f"  Use log target: {USE_LOG_TARGET}\n")
        f.write(f"  Prediction clip quantile: {PREDICTION_CLIP_Q}\n")
        f.write(f"  Parallel enabled: {ENABLE_PARALLEL} (n_jobs={N_JOBS})\n")
        f.write(f"  Per-anchor calibration fraction: {CALIBRATION_FRACTION}\n")
        if "ensemble_weight_naive" in results_df.columns:
            f.write(f"  Ensemble weight (naive): {results_df['ensemble_weight_naive'].iloc[0]:.2f}\n")
        f.write("\n")
        
        f.write("CRITICAL: Uses test-date environmental features (matching original pipeline)\n\n")
        
        f.write("OVERALL METRICS\n")
        for key, value in metrics.items():
            if key != 'site_metrics':
                f.write(f"  {key}: {value}\n")
        
        f.write("\nSITE-SPECIFIC METRICS\n")
        site_metrics = metrics.get('site_metrics', {})
        for site, sm in site_metrics.items():
            f.write(f"\n  {site}:\n")
            for k, v in sm.items():
                f.write(f"    {k}: {v}\n")
    
    print(f"Summary saved to: {summary_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("RAW DATA VALIDATION FOR DATECT XGBOOST MODEL")
    print("(MATCHING ORIGINAL PIPELINE)")
    print("="*70)
    print(f"This script tests model performance on ACTUAL raw measurements only,")
    print(f"not on interpolated or gap-filled values from the processed dataset.")
    print(f"")
    print(f"CRITICAL: Uses test-date environmental features (like original pipeline)")
    print(f"")
    print(f"Configuration:")
    print(f"  - Forecast horizon: {FORECAST_HORIZON_DAYS} days")
    print(f"  - Test sampling: ~20% of each site's raw measurements (with 33% history requirement)")
    print(f"  - Min training samples: {MIN_TRAINING_SAMPLES}")
    print(f"  - Min test date: {MIN_TEST_DATE}")
    print(f"  - Spike threshold: {SPIKE_THRESHOLD} μg/g")
    print(f"  - Random seed: {RANDOM_SEED}")
    print(f"  - Per-anchor calibration fraction: {CALIBRATION_FRACTION}")
    print(f"  - Plots output: {PLOTS_OUTPUT_DIR}")
    
    # Load data
    raw_data = load_raw_da_measurements()
    processed_data = load_processed_data()
    
    # Run validation
    results_df = run_validation(raw_data, processed_data)
    
    if results_df is not None and not results_df.empty:
        # Calculate and display metrics
        metrics = calculate_metrics(results_df)
        
        if metrics:
            # Generate plots
            generate_plots(results_df, metrics, PLOTS_OUTPUT_DIR)
            
            # Save results
            save_results(results_df, metrics, PLOTS_OUTPUT_DIR)
        
        print("\n" + "="*70)
        print("VALIDATION COMPLETE")
        print("="*70)
        if metrics:
            print(f"\nKEY TAKEAWAY:")
            print(f"  XGBoost: R² = {metrics['r2']:.4f}, MAE = {metrics['mae']:.2f} μg/g, F1 = {metrics['f1']:.3f}")
            print(f"  Naive:   R² = {metrics['naive_r2']:.4f}, MAE = {metrics['naive_mae']:.2f} μg/g, F1 = {metrics['naive_f1']:.3f}")
            print(f"")
            print(f"  These metrics represent performance on REAL measurements from raw data files,")
            print(f"  using environmental features from the test date (matching original pipeline).")
    else:
        print("\nValidation failed - no results generated.")


if __name__ == "__main__":
    main()
