# API Fix Summary - Internal Server Error Resolution

## Issue
The `/api/predictions/latest` endpoint was returning a 500 Internal Server Error when querying for NHL (and potentially other sports).

## Root Cause
The endpoint was trying to read predictions from CSV files in the `exports` directory, which:
1. May not exist on Railway (ephemeral filesystem)
2. May be empty or missing files
3. May have missing columns
4. Had no proper error handling

## Solution
Updated `/api/predictions/latest` to:
1. **Query database first** - More reliable on Railway, uses persistent PostgreSQL
2. **Fallback to CSV** - If database has no predictions, try CSV files
3. **Better error handling** - Proper try/except blocks with helpful error messages
4. **Graceful degradation** - Returns empty results instead of crashing

## Changes Made

### `/api/predictions/latest` Endpoint
- Now queries `predictions` table in database for unsettled predictions
- Orders by `predicted_at` (most recent first)
- Falls back to CSV if database has no predictions
- Returns helpful error messages if both fail

### Error Handling
- Added try/except blocks around database queries
- Added validation for CSV file reading
- Returns proper HTTP status codes (404 for not found, 500 for server errors)

## Testing

### Test Database Query (Recommended)
```powershell
# This should now work even if exports directory doesn't exist
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NHL"
```

### Expected Response (if predictions exist)
```json
{
  "sport": "NHL",
  "source": "database",
  "total_predictions": 15,
  "top_picks": [
    {
      "game_id": "401547456",
      "sport": "NHL",
      "market": "moneyline",
      "date": "2024-12-08T17:00:00",
      "home_team": "Team A",
      "away_team": "Team B",
      "side": "home",
      "edge": 0.12,
      "home_win_prob": 0.65,
      "recommended_kelly": 0.05,
      ...
    }
  ]
}
```

### Expected Response (if no predictions)
```json
{
  "detail": "No predictions found for NHL (no CSV files and no database predictions)"
}
```
Status: 404

## Next Steps

1. **Deploy to Railway** - Push these changes to trigger a new deployment
2. **Generate Predictions** - Run the daily workflow to populate the database:
   ```powershell
   Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=true&sports=NHL" -Method POST
   ```
3. **Test Again** - Query the endpoint to verify it works

## Why Database is Better

- ✅ **Persistent** - Railway PostgreSQL persists data across deployments
- ✅ **Reliable** - No filesystem issues
- ✅ **Queryable** - Can filter, sort, and limit easily
- ✅ **Real-time** - Always has latest predictions
- ✅ **Scalable** - Can handle large datasets

## Related Endpoints

All prediction endpoints now use the database:
- `/api/predictions/latest` - Latest unsettled predictions
- `/api/predictions/date-range` - Predictions for date range
- `/api/predictions/next-days` - Predictions for next N days
- `/api/predictions/week` - Predictions for current week
