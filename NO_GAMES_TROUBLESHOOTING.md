# "No Games Found" Troubleshooting Guide

## Issue: Training completed but "No games found for [sport] on [date]"

This happens when the prediction generation step can't find games in the database for today's date.

## Quick Fix

The code has been updated to automatically fetch games from ESPN if none are found in the database. However, here are manual steps if needed:

### 1. Verify Games Were Fetched

Check if games were stored in Step 2 of the daily workflow:

```powershell
# Check health endpoint
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/health"
```

Look for `games_in_db` - should be > 0 if games were stored.

### 2. Manually Fetch Today's Games

If games weren't fetched, trigger just the fetch step:

```powershell
# This will fetch games but not generate predictions
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=false" -Method POST
```

Then generate predictions:

```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=true" -Method POST
```

### 3. Check Railway Logs

Look for these messages in Railway logs:

**Step 2 should show:**
```
[2/5] Fetching today's games...
Fetching NFL games...
  ✓ Stored 12 NFL games
```

**Step 5 should show:**
```
[5/5] Generating predictions...
Found 12 games for NFL on 2025-12-08
```

If Step 2 shows "No games found", it means ESPN doesn't have games scheduled for today.

### 4. Check Date/Time Issues

The system uses UTC time. If you're in a different timezone:
- Games might be scheduled for "tomorrow" in your timezone
- Check what date the system thinks is "today"

**Check current date in logs:**
```
Date: 2025-12-08 21:50:59
```

### 5. Verify Games Exist on ESPN

Check ESPN directly to see if games are scheduled:
- NFL: https://www.espn.com/nfl/schedule
- NHL: https://www.espn.com/nhl/schedule
- NBA: https://www.espn.com/nba/schedule
- MLB: https://www.espn.com/mlb/schedule

### 6. Try Predicting for Tomorrow

If no games today, try tomorrow:

```powershell
# Get tomorrow's date (PowerShell)
$tomorrow = (Get-Date).AddDays(1).ToString("yyyy-MM-dd")
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=true&predict_date=$tomorrow" -Method POST
```

**Note:** The API endpoint doesn't support `predict_date` parameter yet. You'd need to modify the code or use the next-days endpoint:

```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/next-days?sport=NFL&days=7"
```

## What Was Fixed

The `export_predictions_for_date` function now:

1. ✅ **Checks all games first** (not just scheduled) for better debugging
2. ✅ **Auto-fetches from ESPN** if no games found in database
3. ✅ **Better error messages** showing what was found
4. ✅ **More lenient status filtering** (excludes only "final" games)

## Expected Behavior After Fix

When you run the daily workflow:

1. **Step 2:** Fetches games from ESPN and stores them
2. **Step 5:** 
   - First checks database for games
   - If none found, automatically fetches from ESPN
   - Then generates predictions for scheduled/in-progress games

## Still Having Issues?

1. **Check Railway logs** for specific error messages
2. **Verify ESPN has games** for today's date
3. **Check timezone** - system uses UTC
4. **Try a different date** if today has no games

## Common Causes

1. **No games scheduled today** - Some sports have off-days
2. **Games already finished** - All games are "final" status
3. **Timezone mismatch** - Games scheduled for different day in UTC
4. **ESPN API issue** - Games not available from ESPN API
5. **Database not storing** - Games fetched but not saved (check logs for errors)
