# Over/Under (O/U) Data Setup Guide

This guide explains how to set up and backfill Over/Under data for totals model training.

## Overview

The O/U system extracts closing totals from ESPN, calculates actual totals from final scores, and determines OVER/UNDER/PUSH results. This fixes the "all labels are 1.0" error in totals model training.

## Components

1. **Database Schema**: Added `closing_total`, `actual_total`, and `ou_result` columns to `games` table
2. **OUCalculator**: Utility class for extracting and calculating O/U data
3. **ESPN Parser**: Updated to automatically extract O/U data when scraping games
4. **Backfill Script**: Populates historical O/U data for existing games
5. **Training Pipeline**: Updated to use `ou_result` column for more reliable labels

## Setup Steps

### 1. Run Database Migration

Add the new columns to your database:

```powershell
python scripts/migrate_add_ou_columns.py
```

This will:
- Add `closing_total` (FLOAT) column
- Add `actual_total` (INTEGER) column  
- Add `ou_result` (VARCHAR) column
- Create index on `(sport, ou_result)` for faster queries

### 2. Backfill Historical Data

Backfill O/U data for existing games:

```powershell
# Backfill all sports
python scripts/backfill_ou_data.py --sport ALL

# Backfill specific sport
python scripts/backfill_ou_data.py --sport NFL

# Backfill date range
python scripts/backfill_ou_data.py --sport NHL --start 2024-01-01 --end 2024-12-31

# Validate coverage (no backfill)
python scripts/backfill_ou_data.py --validate
```

### 3. Verify Coverage

Check if you have enough O/U data for training:

```powershell
python scripts/backfill_ou_data.py --validate
```

Expected output:
```
O/U Coverage NFL: 85.2% (1234/1447)
  Distribution: {'OVER': 612, 'UNDER': 589, 'PUSH': 33}
  Can train: True
```

**Minimum requirements:**
- At least 200 games with O/U data per sport
- Balanced distribution (not all OVER or all UNDER)
- Less than 5% pushes (normal)

### 4. Retrain Totals Models

Once you have sufficient O/U data, retrain your totals models:

```powershell
# On Railway
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=true&predict=false" -Method POST

# Locally
python scripts/train_all.py
```

## How It Works

### Automatic Extraction

When new games are scraped from ESPN:
1. `OUCalculator.extract_closing_total()` extracts the closing O/U line
2. `OUCalculator.calculate_actual_total()` calculates total points from scores
3. `OUCalculator.determine_ou_result()` determines OVER/UNDER/PUSH
4. Data is stored in `closing_total`, `actual_total`, and `ou_result` columns

### Training Labels

The training pipeline now:
1. **Prefers `ou_result` column** (if available) - most reliable
2. **Falls back to calculation** from scores and `closing_total` if needed
3. **Excludes pushes** (returns NaN) - these can't be used for binary classification
4. **Logs distribution** - shows over/under/push counts for debugging

### Data Quality

The system handles:
- ✅ Missing closing totals (skips game)
- ✅ Missing scores (skips game)
- ✅ Pushes (excludes from training)
- ✅ Invalid data (graceful error handling)

## Troubleshooting

### "All labels are 1.0" Error

**Cause**: All games went OVER (or all went UNDER) - data quality issue

**Solutions**:
1. Check if you have enough games: `python scripts/backfill_ou_data.py --validate`
2. Verify distribution is balanced (not all one outcome)
3. Check if `closing_total` values are correct (not all 0 or very low)
4. Ensure you're using final games with actual scores

### "No games with O/U data" Error

**Cause**: Missing `closing_total` or `ou_result` data

**Solutions**:
1. Run backfill: `python scripts/backfill_ou_data.py --sport <SPORT>`
2. Check if ESPN is providing `overUnder` in odds data
3. Verify database migration ran successfully

### Low Coverage Percentage

**Cause**: Many games missing O/U data

**Solutions**:
1. Backfill more historical data
2. Check if ESPN API is returning odds data
3. Some games may not have closing totals (normal for some sports/leagues)

## Files Modified

- `app/models/db_models.py` - Added O/U columns to Game model
- `app/utils/ou_calculator.py` - New utility class for O/U calculations
- `app/espn_client/parser.py` - Integrated O/U extraction
- `app/training/pipeline.py` - Updated to use `ou_result` column
- `scripts/backfill_ou_data.py` - Backfill script
- `scripts/migrate_add_ou_columns.py` - Database migration

## Next Steps

1. ✅ Run migration
2. ✅ Backfill historical data
3. ✅ Validate coverage
4. ✅ Retrain totals models
5. ✅ Verify models train successfully

After completing these steps, your totals models should train successfully with proper OVER/UNDER labels!
