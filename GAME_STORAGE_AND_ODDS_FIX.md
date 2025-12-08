# Game Storage and Odds Fix

## Issues Fixed

### 1. Games Not Being Stored ✅

**Problem:** New games fetched from ESPN weren't being stored in the database, only backfilled historical games were.

**Root Cause:** The upsert logic was using `vars(game)` which doesn't work correctly with SQLAlchemy models.

**Fix Applied:**
- ✅ Fixed upsert to explicitly update each field
- ✅ Added commit confirmation logging
- ✅ Added better error handling

**File:** `app/espn_client/parser.py`

### 2. Odds Not Being Stored ✅

**Problem:** Odds API key was set up, but odds weren't being stored in the database.

**Root Causes:**
1. Team name matching was too strict (only fuzzy `ilike` match)
2. Date matching might fail due to timezone issues
3. No debugging to see why matches failed

**Fixes Applied:**
- ✅ **Multiple matching strategies:**
  1. Exact match (case insensitive)
  2. Contains match (fuzzy)
  3. Reverse match (in case teams are swapped)
- ✅ **Better date matching:** Uses `func.date()` to compare dates only (ignores timezone)
- ✅ **Comprehensive logging:**
  - Shows matched games
  - Shows unmatched odds
  - Shows existing games in DB for debugging

**File:** `app/odds_api/client.py`

### 3. Better Error Handling ✅

**Added:**
- ✅ Better logging in `fetch_todays_games()` to show:
  - How many games found from ESPN
  - How many successfully stored
  - Errors if storage fails

**File:** `scripts/daily_automation.py`

---

## What to Check

After deploying these fixes, check Railway logs for:

### Game Storage:
```
Fetching NFL games...
  Found 12 NFL games from ESPN
  ✓ Stored 12 NFL games in database
  ✓ Committed 12 games to database
```

### Odds Storage:
```
Fetching NFL odds...
  Attempting to match 12 odds entries to games...
  ✓ Updated odds for 12 NFL games
    Matched games: Team A @ Team B, Team C @ Team D
```

If you see warnings:
```
⚠️  No games matched for odds update
    Unmatched odds: Team A @ Team B
    Existing games in DB: ['Team A @ Team B']
```

This means team names don't match exactly. The system will try multiple matching strategies, but if team names are very different between ESPN and The Odds API, manual mapping might be needed.

---

## Testing

### 1. Test Game Storage:
```powershell
# Fetch games only
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=false" -Method POST
```

Check logs for:
- `Found X games from ESPN`
- `✓ Stored X games in database`
- `✓ Committed X games to database`

### 2. Test Odds Storage:
```powershell
# Fetch games and odds
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=false" -Method POST
```

Check logs for:
- `Attempting to match X odds entries to games...`
- `✓ Updated odds for X games`
- Or warnings if matching fails

### 3. Verify in Database:
```powershell
# Check health endpoint
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/health"
```

Look for `games_in_db` - should increase after fetching.

---

## Common Issues

### Games Still Not Storing?

1. **Check database connection:**
   - Verify `DATABASE_URL` is set in Railway
   - Check logs for database errors

2. **Check ESPN API:**
   - Verify games exist on ESPN for today
   - Check if ESPN API is returning data

3. **Check logs for errors:**
   - Look for `✗ Error parsing/storing games`
   - Check for database commit errors

### Odds Still Not Storing?

1. **Check ODDS_API_KEY:**
   - Verify it's set in Railway environment variables
   - Check logs for "ODDS_API_KEY not set" warnings

2. **Check team name matching:**
   - Look for "Unmatched odds" in logs
   - Compare team names between ESPN and The Odds API
   - Team names might be slightly different (e.g., "Lakers" vs "Los Angeles Lakers")

3. **Check date matching:**
   - Verify games exist in database for the date
   - Check if dates match between ESPN and The Odds API

---

## Next Steps

After deploying:
1. ✅ Run daily workflow to fetch games
2. ✅ Check logs to verify games are stored
3. ✅ Check logs to verify odds are matched and stored
4. ✅ Verify in database using health endpoint
5. ✅ Generate predictions (should now work!)

---

## Summary

✅ **Fixed game storage** - Games now properly stored/updated  
✅ **Fixed odds matching** - Multiple matching strategies  
✅ **Better logging** - See exactly what's happening  
✅ **Better error handling** - Catch and report issues  

The system should now properly store both games and odds! 🎉
