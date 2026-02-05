# DATect Raw Forecasting - Agent Build Instructions

## Quick Commands

```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# Run validation (MANDATORY after every change)
python3 validate_on_raw_data.py

# Check git status
git status
```

## Project Structure

```
datect-raw-forecasting/
├── validate_on_raw_data.py     # Main validation script (run this!)
├── config.py                    # Configuration and paths
├── forecasting/
│   ├── raw_data_forecaster.py  # Core utilities
│   ├── data_processor.py       # Data processing
│   └── models/                 # Model implementations
├── data/
│   ├── raw/da-input/*.csv      # Raw DA measurements
│   └── processed/              # Processed features (parquet)
└── raw_validation_plots/       # Output plots and results
```

## Testing Workflow

1. Make changes to model or features
2. Run: `python3 validate_on_raw_data.py`
3. Check R² in output (target: ≥0.50, baseline: 0.41)
4. Commit results: `git add -A && git commit -m "Ralph: [description] - R² = X.XXX"`

## Output Files

After running validation:
- `raw_validation_plots/validation_summary.txt` - Metrics summary
- `raw_validation_plots/raw_data_validation_results.csv` - Full predictions
- `raw_validation_plots/*.png` - Visualization plots
