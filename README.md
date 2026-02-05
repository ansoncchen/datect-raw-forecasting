# DATect Raw Forecasting

A focused ML pipeline for **1-week-ahead domoic acid (DA) forecasting** using raw measurements and environmental features. This repo isolates the validation and forecasting logic from the full [DATect-Forecasting-Domoic-Acid](https://github.com/ansoncchen/DATect-Forecasting-Domoic-Acid) web application for focused ML iteration.

## Overview

- **10 Pacific Coast monitoring sites** (Oregon to Washington)
- **21 years of data** (2002–2023)
- **~6,600 raw DA measurements** across all sites
- **XGBoost regression** with temporal validation
- **No data leakage** — strict expanding-window evaluation

### Key Results (Current Configuration)

| Metric | XGBoost | Naive Baseline |
|--------|---------|----------------|
| R² | **0.41** | 0.02 |
| MAE | **5.5 μg/g** | 7.3 μg/g |
| Spike F1 | **0.69** | 0.56 |

*Results on 596 evaluation samples using 30% per-site sampling with 20% history requirement.*

---

## Quick Start

```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# Run validation
python3 validate_on_raw_data.py
```

Outputs (plots and CSVs) are saved to `raw_validation_plots/`.

---

## Data Requirements

This repo does **not** include large data files. Copy or symlink from the original DATect project:

```
data/
├── raw/
│   └── da-input/
│       ├── cannon-beach-da.csv      (306 measurements)
│       ├── clatsop-beach-da.csv     (1087 measurements)
│       ├── coos-bay-da.csv          (335 measurements)
│       ├── copalis-da.csv           (933 measurements)
│       ├── gold-beach-da.csv        (717 measurements)
│       ├── kalaloch-da.csv          (710 measurements)
│       ├── long-beach-da.csv        (774 measurements)
│       ├── newport-da.csv           (708 measurements)
│       ├── quinault-da.csv          (602 measurements)
│       └── twin-harbors-da.csv      (760 measurements)
└── processed/
    └── final_output.parquet         (weekly env features, 2003–2023)
```

---

## How the Forecasting Works

### Prediction Task

For a given **site** and **target date** \(T\):
- **Predict**: DA concentration at date \(T\)
- **Using**: Environmental features at \(T\) + historical DA up to \(T - 7\) days
- **Horizon**: 1 week ahead (configurable via `FORECAST_HORIZON_DAYS`)

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAW DA MEASUREMENTS                              │
│   (data/raw/da-input/*.csv — actual lab measurements, irregular dates)  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Aggregate to Weekly (Monday) │
                    │  Using MAX per site-week      │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PROCESSED ENVIRONMENTAL DATA                          │
│        (data/processed/final_output.parquet — weekly, 2003–2023)        │
│                                                                          │
│  Features: SST, Chlorophyll-a, PAR, BEUTI, PDO, ONI, Streamflow, etc.   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Merge + Add Lag Features  │
                    │  da_raw_lag_1, lag_2, lag_3,  │
                    │  lag_52, last_observed_da,    │
                    │  weeks_since_last_raw         │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       FEATURE FRAME           │
                    │  (per site-week: env + DA)    │
                    └───────────────────────────────┘
```

### Features Used

**DA-derived features** (from raw measurements only, no interpolation):
- `da_raw_lag_1`, `da_raw_lag_2`, `da_raw_lag_3`, `da_raw_lag_52` — lagged DA values
- `last_observed_da_raw` — most recent measured DA (forward-filled)
- `weeks_since_last_raw` — time since last measurement

**Environmental features** (from `final_output.parquet`):
- Satellite: `modis-chla`, `modis-sst`, `modis-par`, `modis-flr`, `modis-k490`
- Climate indices: `pdo`, `oni`
- Upwelling: `beuti`
- River flow: `discharge`
- Anomalies: `chla-anom`, `sst-anom`

**Temporal features** (computed at runtime):
- `sin_day_of_year`, `cos_day_of_year`
- `sin_month`, `cos_month`
- `quarter`, `days_since_start`

---

## Validation Methodology

### Expanding Window (Walk-Forward) Validation

For each test measurement at date \(T\):

1. **Set anchor date**: \(A = T - 7\) days
2. **Build training set**: All site data from 2003 through \(A\) where `da_raw` exists
3. **Get test features**: Environmental features at/near date \(T\)
4. **Train XGBoost**: Fresh model on training set
5. **Predict**: DA for date \(T\)
6. **Compare**: Prediction vs actual raw measurement

```
Timeline for a test point at T = 2018-06-15:

2003 ─────────────────────────────── 2018-06-08 ─── 2018-06-15
│                                         │              │
│◄──────── TRAINING DATA ────────────────►│              │
│   (all historical DA + env features)    │              │
│                                         │              │
                                    Anchor Date     Test Date
                                    (T - 7 days)   (prediction target)
```

**Key principle**: The model only ever sees data from before the anchor date. Environmental features from the test date are allowed because in real deployment you would have/forecast those conditions.

### Sampling Strategy

The validation uses a systematic per-site sampling approach:

1. **20% History Requirement**: A test point is only valid if at least 20% of that site's total raw measurements occur on or before the anchor date.

2. **30% Per-Site Sampling**: From valid candidates, sample ~30% of each site's total measurements as test points.

3. **Calibration/Evaluation Split**: 
   - 70% of samples → hyperparameter tuning (grid search over XGBoost params)
   - 30% of samples → final evaluation (reported metrics)

### No Data Leakage

The pipeline ensures temporal integrity:
- ✅ Training only uses DA measurements from `date ≤ anchor_date`
- ✅ DA target (`da_raw`) is explicitly dropped from test features
- ✅ Lag features are computed with proper time shifts (past only)
- ✅ Each test point gets a freshly trained model on its specific history
- ✅ Environmental features in `final_output.parquet` use only trailing windows

---

## Configuration

Key settings in `validate_on_raw_data.py`:

```python
# Forecast horizon
FORECAST_HORIZON_DAYS = 7  # 1 week ahead

# Minimum training requirements
MIN_TRAINING_SAMPLES = 10  # At least 10 DA measurements needed

# Spike detection threshold
SPIKE_THRESHOLD = 20.0  # μg/g (FDA action level)

# Train/test split
CALIBRATION_FRACTION = 0.7  # 70% for tuning, 30% for evaluation

# Model settings
USE_LOG_TARGET = True  # log(1+DA) transformation
PREDICTION_CLIP_Q = 0.99  # Clip extreme predictions
```

XGBoost hyperparameters are tuned via grid search on the calibration set.

---

## Output Files

After running `validate_on_raw_data.py`:

```
raw_validation_plots/
├── overall_scatter.html/png        # Actual vs predicted (all sites)
├── site_performance_comparison.*   # R² and MAE by site
├── timeseries_<Site>.html/png      # Time series per site
├── error_distribution.*            # Prediction error histograms
├── confusion_matrix.*              # Spike detection performance
├── raw_data_validation_results.csv # Full results table
└── validation_summary.txt          # Configuration + metrics summary
```

---

## Interpreting Results

### Regression Metrics

| Metric | Meaning | Good Value |
|--------|---------|------------|
| **R²** | Variance explained | > 0.3 is decent, > 0.5 is good |
| **MAE** | Average absolute error | Lower is better; compare to naive baseline |
| **RMSE** | Root mean squared error | Penalizes large errors more than MAE |
| **Correlation** | Linear relationship strength | > 0.5 indicates useful signal |

### Spike Detection (Classification)

| Metric | Meaning |
|--------|---------|
| **Precision** | Of predicted spikes, how many were real? |
| **Recall** | Of real spikes, how many did we catch? |
| **F1** | Harmonic mean of precision and recall |

For public health applications, **high recall** (catching true spikes) is typically more important than precision.

### Naive Baseline

The naive baseline predicts the **last known raw DA value** — essentially "tomorrow will be like today." Beating this baseline demonstrates the model learns meaningful patterns beyond simple persistence.

---

## Project Structure

```
datect-raw-forecasting/
├── validate_on_raw_data.py      # Main validation script
├── config.py                    # Data paths, model parameters
├── requirements.txt             # Python dependencies
├── forecasting/
│   ├── __init__.py
│   ├── data_processor.py        # Lag feature creation
│   ├── raw_data_forecaster.py   # Core forecasting utilities
│   ├── logging_config.py
│   ├── model_factory.py         # XGBoost model builder
│   ├── sample_weights.py        # Spike-focused sample weighting
│   └── models/
│       └── __init__.py          # Device/accelerator utilities
├── data/
│   ├── README.md
│   ├── raw/da-input/*.csv       # Raw DA measurements
│   └── processed/final_output.parquet  # Env features
└── raw_validation_plots/        # Output directory
```

---

## Extending the System

### Adding New Features

1. Add columns to `final_output.parquet` (or create new env data source)
2. Features are automatically picked up as numeric columns
3. Run validation to assess impact

### Changing Forecast Horizon

In `config.py`:
```python
FORECAST_HORIZON_WEEKS = 2  # Change from 1 to 2 weeks
FORECAST_HORIZON_DAYS = FORECAST_HORIZON_WEEKS * 7
```

---

## References

- Original project: [DATect-Forecasting-Domoic-Acid](https://github.com/ansoncchen/DATect-Forecasting-Domoic-Acid)
- Data sources: NOAA CoastWatch, USGS, Olympic Region HAB Partnership
- Methodology: Expanding window validation with strict temporal integrity

---

## License

Scientific research project. Please cite if used in publications.
