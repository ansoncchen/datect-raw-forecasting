# DATect Raw-Data Forecasting - Project Status

**Last Updated:** 2024-02-05
**Current State:** Baseline model optimized, ready for per-site improvements

---

## Executive Summary

This project implements 1-week-ahead domoic acid (DA) forecasting using XGBoost on raw measurements. After extensive debugging and optimization, we've achieved a working baseline that beats naive persistence.

### Current Best Performance (Single-Stage XGBoost + Ensemble)

| Metric | XGBoost | Ensemble (0.65*XGB + 0.35*Naive) | Naive Baseline | Status |
|--------|---------|-----------------------------------|----------------|--------|
| R² | 0.224 | 0.200 | -0.139 | ✅ XGB beats naive |
| MAE | 10.10 μg/g | 9.08 μg/g | 8.43 μg/g | ⚠️ Naive better |
| F1 (spike detection) | 0.526 | 0.578 | 0.606 | ⚠️ Below naive |
| Recall | 0.733 | 0.785 | 0.806 | ⚠️ Below naive |

**Key Insight:** XGBoost provides predictive value (R²=0.224 vs -0.139) but spike detection remains challenging. Ensemble improves F1 from 0.526→0.578, approaching naive's 0.606.

---

## Configuration (Current Working State)

### File: `validate_on_raw_data.py`

```python
# Line 90: Target transformation
USE_LOG_TARGET = False  # Log transform hurts spike detection

# Line 106: Two-stage model toggle
USE_TWO_STAGE_MODEL = False  # DISABLED - underperforms (R²=0.100 vs 0.224)

# Lines 313-316: Removed zero-importance features
drop_cols = ['date', 'site', 'da_raw', 'da',
             'lat', 'lon', 'weeks_since_last_raw',
             'is_bloom_season', 'quarter', 'da_raw_lag_52']

# Lines 656-662: Ensemble weights
ENSEMBLE_WEIGHT_XGB = 0.65   # Favor XGB (R²=0.22)
ENSEMBLE_WEIGHT_NAIVE = 0.35  # Include naive for MAE benefit
```

### File: `config.py`

```python
# Line 276: Sample weights (CRITICAL)
USE_REGRESSION_SAMPLE_WEIGHTS = False  # ANY weighting causes over-prediction

# Lines 267-268: Spike weights (not used due to above)
SPIKE_FALSE_NEGATIVE_WEIGHT = 50.0
SPIKE_TRUE_NEGATIVE_WEIGHT = 1.0
```

---

## Critical Bug Fixes Applied

### 1. Per-Anchor Calibration Bug (FIXED)
**Problem:** Lines 437-454 used same `calib_rows` for both hyperparameter tuning AND linear calibration, creating circular optimization → R² = -104.6

**Fix:** Removed entire calibration block (lines 437-454), kept only hyperparameter tuning
```python
# Lines 433-435 (simplified)
best_params, _ = tune_xgb_params(calib_rows, feature_frame, base_params)
result = run_single_raw_validation(raw_measurement, feature_frame, best_params)
return result  # No calibration
```

### 2. Sample Weights Cause Over-Prediction (FIXED)
**Problem:** ANY spike weighting (tested 50x, 1500x, 10000x ratios) caused systematic over-prediction → mean 18-20 μg/g vs actual 10.24 μg/g

**Fix:** Disabled sample weights entirely in `config.py` (line 276)

### 3. Early Stopping Failures (FIXED)
**Problem:** `early_stopping_rounds=50` parameter failed in newer XGBoost → 0% successful predictions

**Fix:** Removed explicit parameter, wrapped in try-except (lines 342-366), increased min samples 10→15

### 4. Feature Importance Analysis (COMPLETED)
**Top Features Identified:**
1. `weeks_since_last_spike` (24.5%)
2. `last_observed_da_raw` (18.4%)
3. `days_since_start` (10.1%)

**Removed Features:** 6 zero-importance features (lat, lon, weeks_since_last_raw, is_bloom_season, quarter, da_raw_lag_52)

### 5. Ensemble Weight Optimization (FIXED)
**Problem:** Initial weights 0.80*naive + 0.20*XGB favored worse model (naive R²=-0.14)

**Fix:** Flipped to 0.65*XGB + 0.35*naive → Ensemble R²=0.200, F1=0.578

---

## Two-Stage Model Experiment (FAILED)

### Attempt 1: Data Fragmentation Architecture
- **Approach:** Separate regressors for spike vs non-spike events
- **Results:** R²=0.058, last_observed_da_raw dominated at 96% importance
- **Root Cause:** Spike regressor only saw 13% of data → collapsed to naive

### Attempt 2: Unified Regressor Architecture
- **Approach:** Classifier for spike detection + single regressor for all data
- **Results:** R²=0.100, F1=0.605
- **Conclusion:** Better than Attempt 1 but still worse than single-stage (R²=0.224)

**Decision:** Disable two-stage model (`USE_TWO_STAGE_MODEL = False`), use ensemble baseline

---

## Site-Specific Performance Issues

### Catastrophic Sites (Need Per-Site Models - Phase 9)

| Site | N | XGB R² | Naive R² | Issue |
|------|---|--------|----------|-------|
| **Cannon Beach** | 61 | **-44.0** | -10.7 | Extreme over-prediction |
| **Coos Bay** | 67 | -0.27 | -0.57 | High variance, poor fit |
| **Gold Beach** | 144 | -0.94 | -1.66 | Consistent under-prediction |
| **Newport** | 142 | -0.28 | -0.29 | Both models struggle |

### High-Performing Sites (Could Benefit from Per-Site Tuning)

| Site | N | XGB R² | Naive R² |
|------|---|--------|----------|
| **Copalis** | 167 | 0.72 | 0.72 |
| **Kalaloch** | 131 | 0.67 | 0.67 |
| **Quinault** | 113 | 0.64 | 0.59 |
| **Twin Harbors** | 138 | 0.62 | 0.76 |

---

## Remaining Work: Phases 8-9-10

### Phase 8: Two-Stage Model ❌ COMPLETED (FAILED)
**Status:** Tested both architectures, neither beat single-stage baseline
**Recommendation:** Skip this approach, focus on per-site models instead

### Phase 9: Per-Site Models 🔄 NEXT PRIORITY

**Goal:** Fix catastrophic sites and boost high-performers

#### Implementation Plan

1. **Create `forecasting/per_site_models.py`:**
   ```python
   SITE_SPECIFIC_CONFIGS = {
       'Cannon Beach': {
           'use_site_model': True,
           'max_depth': 3,  # Shallow to prevent overfitting on N=61
           'n_estimators': 200,
           'learning_rate': 0.03,
           'feature_subset': ['last_observed_da_raw', 'weeks_since_last_spike',
                             'modis-sst', 'pdo']  # Reduce feature set
       },
       'Coos Bay': {
           'use_site_model': True,
           'max_depth': 5,
           'reg_lambda': 2.0,  # Strong L2 for high variance
           'min_child_weight': 10  # Require more samples per leaf
       },
       # ... other sites
   }
   ```

2. **Modify `validate_on_raw_data.py`:**
   - Add `USE_PER_SITE_MODELS = True` toggle (after line 106)
   - In `run_single_raw_validation_with_tuning()`, check if site has custom config
   - Override `base_params` with site-specific params before tuning

3. **Testing Strategy:**
   - Start with Cannon Beach (worst site, R²=-44)
   - Try: shallow trees (depth=2-3), reduced features, higher regularization
   - Success metric: R² > -10 (get closer to naive's -10.7)

#### Success Criteria
- Cannon Beach R² > -10 (currently -44)
- Coos Bay R² > 0 (currently -0.27)
- High-performers (Copalis, Kalaloch) maintain R² > 0.65

### Phase 10: Configuration Centralization & Integrity 🔄 FINAL

**Goal:** Move all magic numbers to `config.py` and add temporal leak checks

#### Implementation Plan

1. **Centralize to `config.py`:**
   ```python
   # Move from validate_on_raw_data.py lines 90-106
   USE_LOG_TARGET = False
   USE_TWO_STAGE_MODEL = False
   USE_PER_SITE_MODELS = True  # After Phase 9

   ENSEMBLE_WEIGHT_XGB = 0.65
   ENSEMBLE_WEIGHT_NAIVE = 0.35

   HISTORY_REQUIREMENT_FRACTION = 0.33  # Currently line 503
   CALIBRATION_FRACTION = 0.3
   MIN_TRAINING_SAMPLES = 10

   # Zero-importance features to drop
   ZERO_IMPORTANCE_FEATURES = ['lat', 'lon', 'weeks_since_last_raw',
                               'is_bloom_season', 'quarter', 'da_raw_lag_52']
   ```

2. **Add Temporal Integrity Checks:**
   ```python
   def verify_no_data_leakage(train_data, test_date, anchor_date):
       """Assert no training data after anchor_date"""
       assert train_data['date'].max() <= anchor_date, \
           f"TEMPORAL LEAK: Training data {train_data['date'].max()} > anchor {anchor_date}"

       # Check lag features don't use future data
       for col in train_data.columns:
           if 'lag' in col or 'last_observed' in col:
               # Verify these are computed from data <= anchor_date only
               pass
   ```

3. **Update imports in `validate_on_raw_data.py`:**
   ```python
   from config import (
       USE_LOG_TARGET, USE_TWO_STAGE_MODEL, USE_PER_SITE_MODELS,
       ENSEMBLE_WEIGHT_XGB, ENSEMBLE_WEIGHT_NAIVE,
       ZERO_IMPORTANCE_FEATURES, verify_no_data_leakage
   )
   ```

#### Success Criteria
- All configuration in one place (`config.py`)
- Temporal integrity checks pass on all 1321 test samples
- No magic numbers in `validate_on_raw_data.py`

---

## Files Reference

### Core Implementation
- `validate_on_raw_data.py` - Main validation loop, ensemble logic
- `forecasting/raw_data_forecaster.py` - Feature building, temporal utilities
- `forecasting/two_stage_model.py` - Two-stage architecture (currently unused)
- `config.py` - Model parameters, spike detection settings

### Data Requirements
- `data/raw/da-input/*.csv` - Raw DA measurements (10 sites, 6592 samples)
- `data/processed/final_output.parquet` - Weekly environmental features (2003-2023)

### Outputs
- `raw_validation_plots/` - Visualizations and results
- `raw_validation_plots/feature_importance.csv` - Feature importance analysis
- `raw_validation_plots/raw_data_validation_results.csv` - Full predictions table

---

## Quick Start for New Session

### To Resume Work on Phase 9 (Per-Site Models):

1. **Verify current baseline:**
   ```bash
   python3 validate_on_raw_data.py
   # Should see: XGB R²=0.224, Ensemble F1=0.578
   ```

2. **Create per-site config file:**
   ```bash
   touch forecasting/per_site_models.py
   # Add SITE_SPECIFIC_CONFIGS dict
   ```

3. **Start with Cannon Beach:**
   - Test shallow trees (max_depth=2,3)
   - Reduce feature set to top 5-10 features
   - Increase regularization (reg_lambda=2.0)

4. **Iterate until R² > -10**

### To Disable Two-Stage (if not already):
Set `USE_TWO_STAGE_MODEL = False` in `validate_on_raw_data.py` line 106

### To Change Ensemble Weights:
Modify lines 656-662 in `validate_on_raw_data.py`

---

## Key Learnings & Constraints

### What Works
✅ Expanding window validation (no future leakage)
✅ Per-anchor hyperparameter tuning (grid search on 30% calibration set)
✅ Raw DA targets (no log transform)
✅ Feature importance-based selection (removed 6 zero features)
✅ Ensemble averaging (0.65*XGB + 0.35*naive)
✅ Early stopping with try-except wrapper

### What Doesn't Work
❌ Sample weights (ANY ratio causes over-prediction)
❌ Per-anchor linear calibration (circular optimization)
❌ Log transform (compresses spike signals)
❌ Two-stage classifier→regressor architecture (worse than single-stage)
❌ Data fragmentation (separate spike/normal models)

### Critical Constraints
- **No temporal leakage:** Train only on date ≤ anchor_date
- **Raw measurements only:** Test on actual observations, not interpolated values
- **33% history requirement:** anchor_date must have ≥33% of site's total history
- **Sample size limits:** Some sites only have N=61-67 measurements

---

## Success Metrics

### Current vs. Target

| Metric | Current (Ensemble) | Original Target | Realistic Target |
|--------|-------------------|-----------------|------------------|
| R² | 0.200 | 0.30 | 0.25 |
| MAE | 9.08 μg/g | Minimize | 8.5 μg/g |
| F1 (spike) | 0.578 | Maximize | 0.61 |
| Recall (spike) | 0.785 | Maximize | 0.80 |

**Rationale for Revised Targets:**
- Environmental signal is weak (top non-persistence feature only 10% importance)
- R²=0.30 may be unrealistic without additional data sources
- Focus should be on spike detection (F1, Recall) for public health impact

### Phase 9 Success Metrics
- Cannon Beach R² > -10 (from -44)
- At least 3 catastrophic sites achieve R² > -1.0
- Ensemble R² improves to 0.22-0.25 range
- Ensemble F1 reaches 0.60-0.61 (beats naive's 0.606)

---

## Contact & Repository

**Repository:** `/Users/ansonchen/Downloads/GitHub/datect-raw-forecasting/`
**Validation Script:** `python3 validate_on_raw_data.py` (run on remote cluster)
**Results Directory:** `./raw_validation_plots/`

**Last Successful Run:** 2024-02-05
**Configuration:** Single-stage XGBoost + Ensemble, two-stage disabled
**Next Step:** Phase 9 - Per-site models for Cannon Beach, Coos Bay, Gold Beach, Newport
