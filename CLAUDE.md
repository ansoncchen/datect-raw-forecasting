# CLAUDE.md

This repo is a **focused ML sandbox** for raw-data DA forecasting.

## Quick Commands

```bash
python3 -m pip install -r requirements.txt
python3 validate_on_raw_data.py
```

## Core Flow
- `validate_on_raw_data.py` is the primary evaluation loop
- Raw DA is used as target, no interpolation
- Predictions use test-date environmental features

## Data Requirements
Place or link data here:
```
data/raw/da-input/*.csv
data/processed/final_output.parquet
```

## Success Metric
Primary goal: **maximize R²** on raw measurements while maintaining temporal integrity.
