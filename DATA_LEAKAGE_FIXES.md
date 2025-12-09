# Data Leakage Fixes - Implementation Summary

**Date**: December 9, 2025  
**Status**: ✅ All Critical Fixes Implemented

## Executive Summary

All critical data leakage issues identified in the overfitting action plan have been addressed. The models should now show realistic accuracy metrics (50-55% for spread, 55-65% for moneyline) instead of the impossible 100% accuracies.

## Fixes Implemented

### 1. ✅ Removed `spread_value` and `totals_value` Calculation
**Location**: `app/training/features.py`, lines 533-548

**Problem**: These features directly encoded the prediction target:
- `spread_value > 0` = "cover", `spread_value < 0` = "don't cover" (directly encodes label)
- `totals_value > 0` = "over", `totals_value < 0` = "under" (directly encodes label)

**Fix**: Completely removed the calculation of these features. They were already excluded from the feature list, but now they're not calculated at all to avoid any confusion.

### 2. ✅ Removed Point Differential Features for Spread Market
**Location**: `app/training/features.py`, `get_feature_columns()`

**Removed Features**:
- `point_diff_avg_{window}` - Allows model to reconstruct edge
- `h2h_avg_margin` - Uses point differential
- `strength_diff`, `home_strength`, `away_strength` - Derived from point differentials
- `opponent_strength` - Uses point differential, can reconstruct edge
- `momentum_{window}` - Uses point differential

**Rationale**: For spread betting, these features allow the model to trivially predict outcomes by learning: if `strength_diff > threshold`, predict cover. This is especially problematic when spreads are missing/null (filled with 0), making the label "did home win?" which these features can perfectly predict.

### 3. ✅ Fixed Label Preparation for Spread Market
**Location**: `app/training/pipeline.py`, `prepare_labels()`

**Problem**: When spreads were null, the function fell back to "did home win?" which any win-predicting feature could predict perfectly.

**Fix**: Now returns NaN for games with null spreads, excluding them from training instead of using a fallback that causes leakage.

### 4. ✅ Added Temporal Split Validation
**Location**: `app/training/pipeline.py`, after train/val split

**Fix**: Added validation to ensure train dates are strictly before validation dates:
```python
if train_max_date >= val_min_date:
    print(f"⚠️  WARNING: Temporal leakage detected!")
else:
    print(f"✓ Temporal split verified: Train ends before Val starts")
```

### 5. ✅ Added XGBoost Regularization
**Location**: `app/training/pipeline.py`, model initialization

**Added Parameters**:
- `max_depth`: Limited to 3 (prevent deep memorization)
- `min_child_weight`: 5 (require 5+ samples per leaf)
- `subsample`: 0.8 (use 80% of rows per tree)
- `colsample_bytree`: 0.8 (use 80% of features per tree)
- `reg_alpha`: 1.0 (L1 regularization)
- `reg_lambda`: 1.0 (L2 regularization)

**Rationale**: Prevents overfitting on small datasets (e.g., NFL with only 1,110 training samples).

### 6. ✅ Added Leakage Validation Function
**Location**: `app/training/pipeline.py`, `validate_no_leakage()`

**Checks**:
- High correlations (>0.90) between features and target
- Outcome variable keywords in feature names
- Known leakage features (spread_value, totals_value, etc.)

**Usage**: Automatically runs before training to catch leakage issues early.

### 7. ✅ Enhanced Debug Logging
**Location**: `app/training/pipeline.py`, feature selection

**Added**:
- Logs number of features used
- Warns about potentially problematic features
- Shows first 20 features being used

## Verification Checklist

After re-training, verify these benchmarks:

| Metric | Before (BROKEN) | Expected (FIXED) | Status |
|--------|----------------|------------------|--------|
| NFL Spread Accuracy | 100% | 50-53% | ⏳ Testing |
| NHL Spread Accuracy | 100% | 50-53% | ⏳ Testing |
| NFL Moneyline Accuracy | 100% | 55-62% | ⏳ Testing |
| NFL Totals Accuracy | 63% | 50-55% | ✅ Already realistic |
| MLB Moneyline Accuracy | 98% | 55-62% | ⏳ Testing |
| MLB Spread Accuracy | 99.89% | 50-53% | ⏳ Testing |

**Success Criteria**:
- ✅ No features use actual game outcomes (home_score, away_score, score_diff)
- ✅ All rolling stats use `g["date"] < game_date` to exclude current game
- ✅ Train/val dates have no overlap (temporal split confirmed)
- ⏳ Spread accuracy drops to 50-55% (realistic for betting markets)
- ⏳ Moneyline accuracy in 55-65% range
- ⏳ Log loss increases (less overconfident predictions)

## Rolling Statistics Verification

Our implementation uses a different approach than `.shift(1)` but is equally correct:

```python
# Get recent games (before this game, with valid scores)
recent_games = [
    g for g in team_history[team] 
    if g["date"] < game_date and g.get("has_score", False)
]
```

This ensures:
1. Only games **before** the current game are used (`g["date"] < game_date`)
2. Only games with valid scores are included
3. Games are sorted by date (most recent first)

This is equivalent to `.shift(1)` but more explicit and easier to verify.

## Next Steps

1. **Re-train all models** using the fixed code
2. **Verify metrics** match expected ranges (50-55% for spread, 55-65% for moneyline)
3. **Monitor debug logs** for any remaining leakage warnings
4. **If metrics are still too high**, investigate:
   - Feature correlations with target
   - Whether any features indirectly encode outcomes
   - Whether temporal split is working correctly

## Testing Command

```powershell
$url = "https://moose-picks-api-production.up.railway.app"
Invoke-RestMethod -Uri "$url/api/trigger-daily-workflow?train=true&predict=false" -Method POST
```

## Notes

- **Lower accuracy after fixes is SUCCESS, not failure**. Sports betting markets are efficient - 53% accuracy with good bankroll management can be profitable. 100% accuracy means data leakage.
- The fixes maintain all legitimate features (win rates, points for/against averages, rest days, etc.) while removing only the problematic ones that directly or indirectly encode outcomes.
- All changes are backward compatible - existing models will continue to work, but new training will use the fixed features.
