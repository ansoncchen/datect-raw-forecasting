# DATect Raw Forecasting (Focused Repo)

This repo isolates the **raw-data validation + forecasting** pipeline for domoic acid (DA) prediction. It is intended for focused ML iteration without the full web/backend stack.

## What's Included
- Raw DA validation script: `validate_on_raw_data.py`
- Forecasting utilities: `forecasting/`
- Config: `config.py`
- Ralph/Claude setup: `.ralph/`, `.ralphrc`, `CLAUDE.md`

## Data Setup (Required)
This repo does **not** include large data files. Copy or symlink from the original project:

```
data/raw/da-input/*.csv
data/processed/final_output.parquet
```

Expected paths:
- `data/raw/da-input/`
- `data/processed/final_output.parquet`

## Quick Start
```
python3 -m pip install -r requirements.txt
python3 validate_on_raw_data.py
```

## Outputs
Plots and CSV results are saved to:
```
raw_validation_plots/
```
