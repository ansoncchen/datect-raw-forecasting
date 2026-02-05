# DATect Raw Forecasting - Fix Plan

## Current Metrics to Beat
- **🎯 PRIMARY GOAL**: Raw Data R² = 0.41 → **Target: R² ≥ 0.50**
- Secondary: MAE ~5.5 μg/g → Minimize
- Secondary: Spike F1 ~0.69 → Maximize

**📊 MANDATORY TESTING**: Always run `python3 validate_on_raw_data.py` after changes.

---

## Phase 1: Model Performance Improvement (PRIORITY)

### 1.1 Hyperparameter Optimization
- [ ] Expand PARAM_GRID with more configurations (deeper trees, more estimators)
- [ ] Test different learning rate schedules (0.01, 0.02, 0.03, 0.05, 0.1)
- [ ] Experiment with subsample and colsample_bytree (0.7-0.95 range)
- [ ] Try min_child_weight values (1, 3, 5, 10, 20)
- [ ] Test reg_alpha and reg_lambda combinations for regularization

### 1.2 Feature Engineering
- [ ] Analyze current feature importances from XGBoost
- [ ] Test additional lag features (da_lag_2w, da_lag_4w if not present)
- [ ] Add rolling statistics (7-day, 14-day, 30-day rolling mean/std of DA)
- [ ] Test interaction features (SST × chlorophyll, etc.)
- [ ] Experiment with polynomial features on top predictors

### 1.3 Target Transformation
- [ ] Compare USE_LOG_TARGET=True vs False performance
- [ ] Try Box-Cox transformation on DA values
- [ ] Test quantile transformation
- [ ] Experiment with different PREDICTION_CLIP_Q values (0.95, 0.99, None)

### 1.4 Alternative Models
- [ ] Test LightGBM as drop-in replacement for XGBoost
- [ ] Try CatBoost (handles categorical features natively)
- [ ] Test Random Forest for comparison
- [ ] Implement simple ensemble (average XGBoost + naive baseline)

### 1.5 Calibration Improvements
- [ ] Test isotonic regression calibration instead of linear
- [ ] Try Platt scaling for probability calibration
- [ ] Experiment with different CALIBRATION_FRACTION values (0.3, 0.5, 0.7)

---

## Phase 2: Data Quality & Preprocessing

### 2.1 Missing Value Handling
- [ ] Analyze which features have most missing values
- [ ] Test different imputation strategies (median, mean, KNN)
- [ ] Consider dropping features with >50% missing

### 2.2 Outlier Handling
- [ ] Identify extreme DA values and their impact
- [ ] Test winsorization at different percentiles
- [ ] Try robust scaling instead of MinMaxScaler

---

## Completed Tasks
- [x] Project enabled for Ralph
- [x] Initial PROMPT.md configured with project goals
- [x] fix_plan.md created with prioritized tasks

---

## Experiment Log

### Template for Recording Results
```
Date: YYYY-MM-DD
Experiment: [brief description]
Changes: [what was modified]
Results:
  - R²: X.XXX (baseline: 0.41, delta: +/-X.XXX)
  - MAE: X.XX (baseline: 5.5)
  - F1: X.XXX (baseline: 0.69)
Conclusion: [keep/revert and why]
Next idea: [what to try next]
```

### Experiments Conducted
(Ralph will update this section with results)

---

## Notes & Learnings

- This project tests on **real raw measurements only** - more rigorous than interpolated data
- Data is noisy and spikey - real-world HAB data from 10 Pacific Coast sites
- Current validation uses 50/50 split: half for hyperparameter tuning, half for evaluation
- Log transform (USE_LOG_TARGET=True) is currently enabled
- Linear calibration is applied after predictions
- **Focus on R² regression** - that's the primary metric to improve

---

## Git Workflow Reminder

**EVERY iteration must end with a commit and push!**

```bash
git add -A
git commit -m "Ralph: [description] - R² = X.XXX"
git push origin Ralph-Cycle
```

This ensures all progress is tracked on GitHub for review.
