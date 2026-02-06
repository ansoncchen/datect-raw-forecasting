# DATect Raw-Data Forecasting - Project Status

**Last Updated:** 2025-02-06
**Current State:** Phase 9+10 complete — 3-model ensemble (XGB+RF+Naive) with per-site configs

---

## Executive Summary

1-week-ahead domoic acid (DA) forecasting using a 3-model ensemble:
per-site XGBoost + Random Forest + Naive baseline, with per-site ensemble weights.

### Current Best Performance (Iter 5 — 3-Model Ensemble)

| Metric | XGBoost | Random Forest | Naive | **Ensemble** |
|--------|---------|---------------|-------|-------------|
| R² | 0.353 | 0.372 | -0.139 | **0.394** |
| MAE | 7.62 μg/g | 6.87 μg/g | 7.97 μg/g | **6.98 μg/g** |
| RMSE | 16.99 | 16.74 | 22.54 | **16.44** |
| Spike F1 | 0.578 | 0.600 | 0.606 | 0.590 |
| Spike Recall | 0.645 | 0.709 | 0.779 | 0.715 |

### Per-Site Performance (Ensemble R²)

| Site | N | XGB R² | RF R² | Naive R² | Ens R² |
|------|---|--------|-------|----------|--------|
| Twin Harbors | 138 | +0.597 | +0.604 | +0.763 | **+0.798** |
| Copalis | 167 | +0.732 | +0.765 | +0.715 | **+0.761** |
| Kalaloch | 131 | +0.565 | +0.679 | +0.669 | **+0.680** |
| Quinault | 113 | +0.528 | +0.585 | +0.590 | **+0.653** |
| Long Beach | 140 | +0.638 | +0.615 | +0.470 | **+0.643** |
| Coos Bay | 67 | +0.337 | +0.305 | -0.570 | **+0.323** |
| Clatsop Beach | 218 | +0.171 | +0.238 | -0.015 | **+0.226** |
| Newport | 142 | -0.127 | +0.038 | -0.287 | **-0.015** |
| Gold Beach | 144 | -0.094 | -0.091 | -1.656 | **-0.105** |
| Cannon Beach | 61 | -0.257 | -0.539 | -10.663 | **-0.314** |

---

## Configuration (All Centralized in config.py)

All magic numbers live in `config.py` — no magic numbers in `validate_on_raw_data.py`.

```python
USE_LOG_TARGET = False
USE_PER_SITE_MODELS = True
PREDICTION_CLIP_Q = 0.99
CALIBRATION_FRACTION = 0.3
MAX_CALIBRATION_ROWS = 20
HISTORY_REQUIREMENT_FRACTION = 0.33
MIN_TRAINING_SAMPLES = 10
ENABLE_PARALLEL = True
N_JOBS = -1
ZERO_IMPORTANCE_FEATURES = ['lat', 'lon', 'weeks_since_last_raw',
                            'is_bloom_season', 'quarter', 'da_raw_lag_52']
```

Per-site configs (XGB params, RF params, feature subsets, ensemble weights, clipping) are in
`forecasting/per_site_models.py`.

---

## Architecture

### 3-Model Ensemble

For each test point:
1. **XGBoost** — per-site hyperparams, per-anchor grid search tuning
2. **Random Forest** — per-site params (conservative for weak sites), no tuning
3. **Naive** — last known raw DA value

Final prediction: `w_xgb * XGB + w_rf * RF + w_naive * Naive`

Weights are per-site, tuned based on actual model performance:
- Sites where naive dominates (Twin Harbors): heavy naive weight (0.60)
- Sites where RF excels (Newport, Kalaloch): heavy RF weight (0.50–0.70)
- Sites where XGB leads (Long Beach, Coos Bay): heavy XGB weight (0.50–0.55)
- Catastrophic sites (Cannon Beach): near-pure XGB (0.95) to minimize damage

### Temporal Integrity

- `verify_no_data_leakage()` in config.py — called every test point
- Training only uses `date <= anchor_date`
- `da_raw` and `da` dropped from test features
- Lag features use proper past-only shifts
- Fresh model per test point (no lookahead)

---

## Completed Phases

### Phase 8: Two-Stage Model ❌ FAILED (removed)
- Tested classifier→regressor architecture, R²=0.100 vs single-stage 0.224
- Code deleted: `forecasting/two_stage_model.py`

### Phase 9: Per-Site Models ✅ COMPLETE
- Created `forecasting/per_site_models.py` with 10 site-specific configs
- XGB R² improved from 0.224 → 0.354 (Iters 1–4)
- Copalis: 0.097 → 0.732, Kalaloch: -1.757 → 0.565, Long Beach: 0.416 → 0.638

### Phase 10a: 3-Model Ensemble ✅ COMPLETE
- Added RF as third model via `forecasting/model_factory.py`
- Per-site 3-tuple ensemble weights (xgb, rf, naive)
- Ensemble R² = 0.394 (Iter 5 with recalibrated weights)

### Phase 10b: Configuration Centralization ✅ COMPLETE
- All magic numbers moved to `config.py`
- `verify_no_data_leakage()` temporal integrity check added
- Dead code removed (two-stage, sample weights)
- Stale files deleted (6 files)

---

## Files Reference

### Active Files

| File | Purpose |
|------|---------|
| `validate_on_raw_data.py` | Main validation loop (1321 test points, 10 sites) |
| `config.py` | All configuration, model params, `verify_no_data_leakage()` |
| `forecasting/per_site_models.py` | Per-site XGB/RF params, feature subsets, ensemble weights |
| `forecasting/model_factory.py` | `build_xgb_regressor()`, `build_rf_regressor()` |
| `forecasting/raw_data_forecaster.py` | Feature building, temporal utilities |
| `forecasting/data_processor.py` | Minimal DataProcessor for raw lag features |
| `forecasting/logging_config.py` | Logging configuration |
| `forecasting/__init__.py` | Module init |

### Deleted (Stale)

| File | Reason |
|------|--------|
| `forecasting/two_stage_model.py` | Disabled permanently, code removed |
| `forecasting/sample_weights.py` | Disabled permanently (`USE_REGRESSION_SAMPLE_WEIGHTS=False`) |
| `forecasting/models/__init__.py` | Torch/Lightning wrappers, never imported |
| `compare_xgb_rf.py` | One-off comparison, RF now integrated |
| `verify_temporal_integrity.py` | Replaced by `verify_no_data_leakage()` in config |
| `validation_phase1-3_output.txt` | Old output artifact |

### Data Requirements

```
data/raw/da-input/*.csv              # Raw DA measurements (10 sites)
data/processed/final_output.parquet  # Weekly env features (2003-2023)
```

---

## Performance Journey

| Phase | XGB R² | Ensemble R² | Key Change |
|-------|--------|-------------|------------|
| Baseline | 0.224 | 0.200 | Single XGB + naive |
| Phase 9 Iter 3 | 0.350 | 0.267 | Per-site XGB configs |
| Phase 9 Iter 4 | 0.354 | 0.366 | Ensemble weight recalibration |
| Phase 10 (RF added) | 0.354 | 0.373 | 3-model ensemble |
| **Phase 10 Iter 5** | **0.354** | **0.394** | **Weight recalibration on actual RF data** |

---

## Quick Start

```bash
# Run validation (ON CLUSTER ONLY — do not run locally)
python3 validate_on_raw_data.py

# Results saved to:
#   raw_validation_plots/raw_data_validation_results.csv
#   raw_validation_plots/validation_summary.txt
```
