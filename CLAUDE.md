# CLAUDE.md

Quick reference for the DATect raw-data DA forecasting sandbox.

## Commands

```bash
python3 -m pip install -r requirements.txt   # Install deps
python3 validate_on_raw_data.py              # Run validation
```

## Core Concepts

### What It Does
- **1-week-ahead DA forecasting** using XGBoost
- Targets **raw DA measurements** (no interpolation)
- Uses **environmental features** from processed dataset + **DA lag features**

### How Validation Works
For each test point at date T:
1. Anchor = T - 7 days
2. Train on **all site history** from 2003 to anchor (expanding window)
3. Predict using **env features at T** + **DA lags up to anchor**
4. Compare to actual raw measurement at T

### Sampling Rules
- **20% history requirement**: anchor must have ≥20% of site's data behind it
- **30% per-site sampling**: ~30% of each site's measurements used as test points
- **70/30 calibration split**: 70% for hyperparameter tuning, 30% for final metrics

## Data Requirements

```
data/raw/da-input/*.csv              # Raw DA measurements (10 sites)
data/processed/final_output.parquet  # Weekly env features (2003-2023)
```

## Key Files

| File | Purpose |
|------|---------|
| `validate_on_raw_data.py` | Main validation loop |
| `forecasting/raw_data_forecaster.py` | Core utilities (training frames, feature building) |
| `config.py` | Paths, model params, thresholds |

## Configuration (in validate_on_raw_data.py)

```python
FORECAST_HORIZON_DAYS = 7      # Prediction horizon
MIN_TRAINING_SAMPLES = 10      # Min history required
SPIKE_THRESHOLD = 20.0         # μg/g for binary classification
CALIBRATION_FRACTION = 0.7     # Train/eval split
USE_LOG_TARGET = True          # log(1+DA) transform
```

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| R² | ~0.41 | Maximize |
| MAE | ~5.5 μg/g | Minimize |
| Spike F1 | ~0.69 | Maximize recall |

## No Data Leakage Guarantees

- ✅ Training only uses `date ≤ anchor_date`
- ✅ `da_raw` and `da` dropped from test features
- ✅ Lag features use proper past-only shifts
- ✅ Fresh model per test point (no lookahead)

## Outputs

Results saved to `raw_validation_plots/`:
- Scatter plots, time series, confusion matrices
- `raw_data_validation_results.csv` — full predictions table
- `validation_summary.txt` — metrics summary
