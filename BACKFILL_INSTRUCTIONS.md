# Historical Data Backfill Instructions

## Overview

The backfill script fetches historical game data from ESPN for the past 5 seasons (2020-2024) for all 4 sports (NFL, NHL, NBA, MLB). This data is required to train the ML models.

## Quick Start

### Run Locally

```bash
# Backfill all 4 sports, past 5 seasons
python scripts/backfill_historical_data.py
```

This will:
- Fetch games day-by-day for each season
- Only store games with **final scores** (completed games)
- Handle duplicates (upserts existing games)
- Show progress as it runs

**Time Estimate:** 
- ~2-4 hours for all 4 sports
- ~30-60 minutes per sport
- Depends on API response times

### Run on Railway

**Option 1: Via Railway CLI**
```bash
railway run python scripts/backfill_historical_data.py
```

**Option 2: Via Railway Console**
1. Go to Railway dashboard
2. Select your service
3. Open "Deployments" → "View Logs"
4. Use Railway's console/terminal feature to run:
   ```bash
   python scripts/backfill_historical_data.py
   ```

**Option 3: Add as One-Time Task**
1. Create a new Railway service (temporary)
2. Set it to run: `python scripts/backfill_historical_data.py`
3. Run it once, then delete the service

## Command Options

### Backfill Specific Sports

```bash
# Just NHL
python scripts/backfill_historical_data.py --sports NHL

# NFL and NHL only
python scripts/backfill_historical_data.py --sports NFL NHL
```

### Backfill Specific Seasons

```bash
# Just 2023 and 2024 seasons
python scripts/backfill_historical_data.py --seasons 2023 2024

# Single sport, specific seasons
python scripts/backfill_historical_data.py --sports NHL --seasons 2023 2024
```

### Adjust API Call Rate

```bash
# Faster (0.05s delay - may hit rate limits)
python scripts/backfill_historical_data.py --delay 0.05

# Slower (0.2s delay - safer)
python scripts/backfill_historical_data.py --delay 0.2
```

## What Gets Stored

The script only stores games with:
- ✅ **Final status** (game completed)
- ✅ **Final scores** (home_score and away_score)
- ✅ All games are upserted (duplicates handled)

This ensures only completed games are used for training.

## Progress Monitoring

The script shows:
- Progress every 30 days processed
- Games stored per date
- Summary per season
- Final totals per sport

Example output:
```
Backfilling NHL 2023 Season
Date Range: 2023-10-01 to 2024-06-30

  2023-10-01: Stored 2 games
  2023-10-02: Stored 3 games
  ...
  Progress: 30 days processed, 87 games stored...
  
✓ NHL 2023: 1,312 games stored
```

## After Backfilling

Once historical data is loaded:

1. **Train Models:**
   ```bash
   python scripts/train_all.py
   ```
   Or via API:
   ```powershell
   Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=true&predict=false" -Method POST
   ```

2. **Generate Predictions:**
   ```powershell
   Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=true&sports=NFL,NHL,NBA,MLB" -Method POST
   ```

## Troubleshooting

### Rate Limiting

If you get rate limit errors:
- Increase delay: `--delay 0.2` or `--delay 0.5`
- Run one sport at a time
- Wait and resume later

### Missing Games

Some dates may have no games (off-days, lockouts, etc.). This is normal.

### Database Connection Issues

On Railway, ensure:
- `DATABASE_URL` environment variable is set
- Database service is running
- Connection string is correct

### Partial Backfill

If the script stops partway:
- It's safe to re-run - duplicates are handled
- The script will continue from where it left off
- Already-stored games are updated, not duplicated

## Season Date Ranges

The script uses these season ranges:

- **NFL**: September 1 - February 28 (next year)
- **NHL**: October 1 - June 30 (next year)
- **NBA**: October 1 - June 30 (next year)
- **MLB**: March 1 - November 30 (same year)

## Expected Game Counts

Approximate games per season:
- **NFL**: ~272 games/season
- **NHL**: ~1,312 games/season
- **NBA**: ~1,230 games/season
- **MLB**: ~2,430 games/season

For 5 seasons (2020-2024):
- **NFL**: ~1,360 games
- **NHL**: ~6,560 games
- **NBA**: ~6,150 games
- **MLB**: ~12,150 games
- **Total**: ~26,220 games

## Next Steps

After backfilling:
1. ✅ Historical data loaded
2. ✅ Train models (will now succeed)
3. ✅ Generate predictions
4. ✅ Set up daily automation
