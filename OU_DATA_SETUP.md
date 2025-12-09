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

### Prerequisites

No prerequisites needed! All commands use the Railway API via PowerShell `Invoke-RestMethod`.

**Base URL:** `https://moose-picks-api-production.up.railway.app`

### 1. Run Database Migration

Add the new columns to your database via API:

```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/migrate-ou-columns" -Method POST
```

This will:
- Add `closing_total` (FLOAT) column
- Add `actual_total` (INTEGER) column  
- Add `ou_result` (VARCHAR) column
- Create index on `(sport, ou_result)` for faster queries

**Note:** The migration runs in the background. Check Railway logs to verify completion.

### 2. Backfill Historical Data

Backfill O/U data for existing games via API:

```powershell
# Backfill all sports
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/backfill-ou-data?sport=ALL" -Method POST

# Backfill specific sport
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/backfill-ou-data?sport=NFL" -Method POST

# Backfill date range for specific sport
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/backfill-ou-data?sport=NHL&start_date=2024-01-01&end_date=2024-12-31" -Method POST
```

**Note:** Backfill runs in the background. Check Railway logs to monitor progress.

**Important:** The backfill can only set `ou_result` for games that have `over_under` values. If your games have scores but no `over_under` values, you need to backfill odds first:

```powershell
# First, backfill odds to get over_under values
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/backfill-odds?sport=NFL&start_date=2021-09-01&end_date=2024-12-31" -Method POST

# Then, backfill O/U data to calculate ou_result
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/backfill-ou-data?sport=NFL" -Method POST
```

Games without `over_under` will get `actual_total` calculated but NOT `ou_result` (can't determine OVER/UNDER without the line).

### 3. Verify Coverage

Check if you have enough O/U data for training:

```powershell
# Validate all sports
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/validate-ou-coverage" -Method GET

# Validate specific sport
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/validate-ou-coverage?sport=NFL" -Method GET
```

Expected output (JSON):
```json
{
  "sport": "NFL",
  "coverage": {
    "sport": "NFL",
    "total_completed": 1447,
    "with_ou_data": 1234,
    "coverage_pct": 85.2,
    "distribution": {
      "OVER": 612,
      "UNDER": 589,
      "PUSH": 33
    },
    "can_train": true
  }
}
```

**Note:** If you see low coverage (e.g., 2-5%), you need to run the backfill first. The distribution will be empty until games have been processed with O/U data.

**Minimum requirements:**
- At least 200 games with O/U data per sport
- Balanced distribution (not all OVER or all UNDER)
- Less than 5% pushes (normal)

**Current Status:** Based on your API response, you have:
- NFL: 30 games with O/U data (2.22% coverage) - **Need to backfill**
- Distribution is empty because most games haven't been processed yet

**Next Step:** Run the backfill to populate O/U data for historical games.

### 4. Retrain Totals Models

Once you have sufficient O/U data, retrain your totals models:

```powershell
# Trigger training via Railway API (recommended)
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=true&predict=false" -Method POST

# Or run training script directly on Railway
railway run python scripts/train_all.py
```

**Note:** The API endpoint runs in the background and returns immediately. Check Railway logs to monitor progress.

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
1. Check if you have enough games: `Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/validate-ou-coverage" -Method GET`
2. Verify distribution is balanced (not all one outcome)
3. Check if `closing_total` values are correct (not all 0 or very low)
4. Ensure you're using final games with actual scores

### "No games with O/U data" Error

**Cause**: Missing `closing_total` or `ou_result` data

**Solutions**:
1. **Check if games have `over_under` values**: Games need `over_under` to calculate `ou_result`. If your games have scores but no `over_under`, backfill odds first:
   ```powershell
   # Backfill odds to get over_under values
   Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/backfill-odds?sport=<SPORT>&start_date=2021-01-01&end_date=2024-12-31" -Method POST
   ```
2. **Then run O/U backfill**: `Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/backfill-ou-data?sport=<SPORT>" -Method POST`
3. Check if ESPN is providing `overUnder` in odds data
4. Verify database migration ran successfully: `Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/migrate-ou-columns" -Method POST`

### Backfill Not Updating Old Records

**Cause**: Games may not have `over_under` values, so they can't get `ou_result` set

**What the backfill does**:
- ✅ Sets `actual_total` for ALL games with scores (even without `over_under`)
- ✅ Sets `closing_total` from `over_under` (if `over_under` exists)
- ✅ Sets `ou_result` ONLY if both `actual_total` and `closing_total` exist

**If games have scores but no `over_under`**:
1. They will get `actual_total` set ✅
2. They will NOT get `ou_result` set ❌ (can't determine OVER/UNDER without the line)
3. They won't count toward coverage (validation only counts games with `ou_result`)

**Solution**: Backfill odds first to populate `over_under` values, then run O/U backfill again.

### Low Coverage Percentage

**Cause**: Many games missing O/U data

**Solutions**:
1. Backfill more historical data: `Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/backfill-ou-data?sport=<SPORT>" -Method POST`
2. Check if ESPN API is returning odds data
3. Some games may not have closing totals (normal for some sports/leagues)

### API Response Issues

**"Connection refused" or timeout**:
- Verify Railway service is running (check Railway dashboard)
- Check if the API URL is correct
- Wait a few minutes and retry

**Background tasks not completing**:
- Check Railway logs for errors
- Large backfills may take 10-30 minutes
- Consider backfilling one sport at a time

## Files Modified

- `app/models/db_models.py` - Added O/U columns to Game model
- `app/utils/ou_calculator.py` - New utility class for O/U calculations
- `app/espn_client/parser.py` - Integrated O/U extraction
- `app/training/pipeline.py` - Updated to use `ou_result` column
- `scripts/backfill_ou_data.py` - Backfill script
- `scripts/migrate_add_ou_columns.py` - Database migration

## Quick Reference: API PowerShell Commands

### Complete Setup Workflow

```powershell
# Base URL (set as variable for convenience)
$baseUrl = "https://moose-picks-api-production.up.railway.app"

# 1. Run migration
Invoke-RestMethod -Uri "$baseUrl/api/migrate-ou-columns" -Method POST

# 2. Backfill all sports
Invoke-RestMethod -Uri "$baseUrl/api/backfill-ou-data?sport=ALL" -Method POST

# 3. Validate coverage
Invoke-RestMethod -Uri "$baseUrl/api/validate-ou-coverage" -Method GET

# 4. Retrain models
Invoke-RestMethod -Uri "$baseUrl/api/trigger-daily-workflow?train=true&predict=false" -Method POST
```

### Individual Commands

```powershell
# Migration
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/migrate-ou-columns" -Method POST

# Backfill all sports
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/backfill-ou-data?sport=ALL" -Method POST

# Backfill specific sport
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/backfill-ou-data?sport=NFL" -Method POST

# Backfill with date range
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/backfill-ou-data?sport=NHL&start_date=2024-01-01&end_date=2024-12-31" -Method POST

# Validate all sports
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/validate-ou-coverage" -Method GET

# Validate specific sport
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/validate-ou-coverage?sport=NFL" -Method GET

# Retrain models
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=true&predict=false" -Method POST
```

### Monitoring Progress

Check Railway logs to monitor backfill progress:
- Railway Dashboard → Your Service → Logs
- Look for "Updated X games with O/U data" messages
- Check for any errors or warnings
- API endpoints return immediately; tasks run in background

## Next Steps

1. ✅ Run migration: `Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/migrate-ou-columns" -Method POST`
2. ✅ Backfill historical data: `Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/backfill-ou-data?sport=ALL" -Method POST`
3. ✅ Validate coverage: `Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/validate-ou-coverage" -Method GET`
4. ✅ Retrain totals models: `Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=true&predict=false" -Method POST`
5. ✅ Verify models train successfully (check Railway logs)

After completing these steps, your totals models should train successfully with proper OVER/UNDER labels!
