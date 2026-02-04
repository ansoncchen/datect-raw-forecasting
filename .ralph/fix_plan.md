# DATect Raw Forecasting - Fix Plan

## Current Metrics to Beat
- **🎯 PRIMARY GOAL**: Raw Data R² = 0.41 → **Target: R² ≥ 0.55** (Stretch: **R² ≥ 0.65**)
- Secondary: MAE ~5.5 μg/g → Target: < 4.5 (Stretch: < 3.5)
- Secondary: Spike F1 ~0.69 → Target: > 0.75 (Stretch: > 0.85)

**⚠️ CRITICAL**: Current R² of 0.41 is NOT acceptable. Historical testing shows XGBoost can achieve R² > 0.80 on similar data. Push for breakthrough improvements, not incremental gains!

**📊 MANDATORY TESTING**: Always run `python3 validate_on_raw_data.py` after changes.

---

## ⚠️ IMPORTANT: These Tasks Are Just IDEAS

**You have COMPLETE freedom to:**
- ✅ Skip any/all of these tasks if they don't seem promising
- ✅ Add your own tasks based on creative ideas
- ✅ Try completely different approaches not listed here
- ✅ Experiment with anything that might improve R² - no restrictions!
- ✅ Work on tasks in any order (or ignore the order entirely)

**Don't feel constrained by this list. It's guidance, not rules.**

---

## Phase 1: Quick Wins (Bug Fixes & Config)

### 1.1 Configuration Fixes (High Confidence)
- [ ] Check if all XGBoost params from config.py are actually being used (colsample_bylevel, gamma, etc.)
- [ ] Verify lag features match config.py LAG_FEATURES = [1, 2, 3, 4, 52]
- [ ] Test PREDICTION_CLIP_Q = None (removing clipping might help high-DA predictions)
- [ ] Try USE_LOG_TARGET = False to see if log transform helps or hurts

### 1.2 Hyperparameter Optimization
- [ ] Expand PARAM_GRID from 4 to 8+ configurations
- [ ] Test different learning rate schedules (0.01, 0.02, 0.03, 0.05, 0.1)
- [ ] Experiment with subsample and colsample_bytree (0.7-0.95 range)
- [ ] Try min_child_weight values (1, 3, 5, 10, 20)
- [ ] Test reg_alpha and reg_lambda combinations for regularization
- [ ] Consider implementing Optuna/Bayesian optimization

---

## Phase 2: Feature Engineering

### 2.1 Lag Features
- [ ] Verify current lag features are working correctly
- [ ] Test additional lags (2-week, 4-week, 8-week DA lags)
- [ ] Add rolling statistics (4/8/13-week rolling mean/max/std of raw DA)

### 2.2 Environmental Interactions
- [ ] Add interaction features (chla × sst, beuti × sst, chla × beuti)
- [ ] Test polynomial features on top predictors
- [ ] Try site-specific features or embeddings

### 2.3 Temporal Features
- [ ] Verify sin/cos encoding is working properly
- [ ] Test day-of-year, week-of-year features
- [ ] Add seasonal indicators (spring bloom, fall bloom periods)

---

## Phase 3: Alternative Models

### 3.1 Gradient Boosting Variants
- [ ] Test LightGBM as drop-in replacement
- [ ] Try CatBoost (handles categorical features natively)
- [ ] Compare Random Forest baseline

### 3.2 Ensemble Methods
- [ ] Blend XGBoost + naive baseline (weighted average)
- [ ] Try stacking ensemble (XGBoost + LightGBM + RF)
- [ ] Test voting regressor

---

## Phase 4: Calibration & Post-Processing

### 4.1 Calibration Improvements
- [ ] Try isotonic regression calibration instead of linear
- [ ] Test Platt scaling
- [ ] Experiment with different CALIBRATION_FRACTION values (0.3, 0.5, 0.7)

### 4.2 Prediction Post-Processing
- [ ] Analyze prediction errors by site, season, DA level
- [ ] Consider site-specific models or adjustments
- [ ] Test prediction smoothing or ensembling across time

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
- Many features may HURT performance - feel free to disable things that don't help
- Try anything that might work - you have full experimental freedom!

### Why Current Performance is Low (Opportunities!)
The current R² of 0.41 suggests there's **massive room for improvement**:
1. **Hyperparameters may be suboptimal** - The PARAM_GRID is small (only 4 configs)
2. **Features may be wrong** - Some features might hurt; others might be missing
3. **Model choice** - XGBoost is good, but ensembles or other boosters might be better
4. **Calibration** - Linear calibration is naive; better methods exist
5. **Data preprocessing** - Outlier handling, scaling choices matter

**Don't accept incremental 0.01-0.02 gains. Look for the changes that give 0.05-0.10+ jumps!**

---

## Git Workflow Reminder

**EVERY iteration must end with a commit and push!**

```bash
git add -A
git commit -m "Ralph: [description] - R² = X.XXX"
git push origin Ralph-Cycle
```

This ensures all progress is tracked on GitHub for review.

---

## High-Impact Ideas Worth Exploring

These could give **big jumps** in R² - prioritize bold experiments over safe ones:

1. **🔥 Ensemble everything**: XGBoost + LightGBM + CatBoost + RF → blend predictions
2. **🔥 Fix the hyperparameters**: Current PARAM_GRID is tiny. Try 20+ configs or Optuna
3. **🔥 Disable prediction clipping**: PREDICTION_CLIP_Q = 0.99 might be suppressing good high-DA predictions
4. **🔥 Better lag features**: Add 2/4/8-week rolling stats on raw DA
5. **Site-specific models**: Train separate models per site, or add site embeddings
6. **Environmental interactions**: chla × sst, beuti × temp - capture nonlinear relationships
7. **Extreme value focus**: High DA events are rare but critical. Weight them more?
8. **Simpler might be better**: Feature ablation - remove features that hurt
9. **Seasonal encoding**: HABs are highly seasonal. sin/cos of day-of-year might help
10. **Blend with naive baseline**: Weighted average of XGBoost + last-known-value

**Remember: A 0.10 R² improvement means going from mediocre (0.41) to good (0.51) to great (0.61). That's the goal!**
