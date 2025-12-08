# How to Generate Predictions - Step by Step

## Current Status
✅ API is working (200 OK responses)  
✅ CORS is configured  
❌ Database has no predictions yet  

## Solution: Generate Predictions

The API needs to run the daily workflow to:
1. Fetch games from ESPN
2. Fetch odds from The Odds API
3. Train models (optional, can skip if models already exist)
4. Generate predictions and store them in the database

## Quick Start - Generate Predictions

### Option 1: Generate Predictions (No Training) - Fastest

This assumes you already have trained models. If you don't have models yet, use Option 2.

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=true&sports=NFL,NHL,NBA,MLB" -Method POST
```

**What this does:**
- ✅ Settles yesterday's predictions
- ✅ Fetches today's games from ESPN
- ✅ Fetches odds from The Odds API
- ❌ Skips training (uses existing models)
- ✅ Generates predictions for all 4 sports

**Expected response:**
```json
{
  "status": "started",
  "message": "Daily workflow started in background",
  "sports": ["NFL", "NHL", "NBA", "MLB"],
  "train": false,
  "predict": true
}
```

### Option 2: Full Workflow (Train + Predict) - Complete

This will train models AND generate predictions. Takes longer but ensures you have the latest models.

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=true&predict=true&sports=NFL,NHL,NBA,MLB" -Method POST
```

**What this does:**
- ✅ Settles yesterday's predictions
- ✅ Fetches today's games from ESPN
- ✅ Fetches odds from The Odds API
- ✅ Trains models for all 4 sports (takes 5-15 minutes)
- ✅ Generates predictions for all 4 sports

## Monitor Progress

### 1. Check Railway Logs
- Go to Railway dashboard
- Click on your service
- View "Deployments" → Latest deployment → "View Logs"
- You'll see progress like:
  ```
  [1/5] Settling yesterday's predictions...
  [2/5] Fetching today's games...
  [3/5] Fetching odds from The Odds API...
  [4/5] Training models... (if train=true)
  [5/5] Generating predictions...
  ```

### 2. Check Health Endpoint
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/health"
```

This shows:
- Database connection status
- Number of games in database
- Number of trained models

## Verify Predictions Were Generated

### Wait 2-5 Minutes
The workflow runs in the background. Wait a few minutes for it to complete.

### Test Prediction Endpoints

**NHL - Latest:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NHL"
```

**NFL - Latest:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NFL"
```

**NBA - Latest:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NBA"
```

**MLB - Latest:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=MLB"
```

### Expected Response (After Predictions Generated)
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

## Troubleshooting

### Still Getting Empty Predictions?

**1. Check if games were fetched:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/health"
```
Look at `games_in_db` - should be > 0

**2. Check Railway logs for errors:**
- Look for error messages in deployment logs
- Common issues:
  - Missing `ODDS_API_KEY` (predictions still work, just no odds)
  - No games scheduled for today
  - Model training failed (if train=true)

**3. Check if models exist:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/models"
```
Should return a list of model files if training completed.

**4. Try single sport first:**
```powershell
# Test with just NHL
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=true&sports=NHL" -Method POST
```

### No Games Scheduled Today?

If there are no games scheduled for today, predictions will be empty. Try:
- Check if games exist for upcoming dates
- Use `/api/predictions/next-days` endpoint instead:
  ```powershell
  Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/next-days?sport=NHL&days=7"
  ```

### Models Don't Exist?

If you get errors about missing models:
1. Run full workflow with training:
   ```powershell
   Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=true&predict=true&sports=NFL,NHL,NBA,MLB" -Method POST
   ```
2. Wait 10-15 minutes for training to complete
3. Check logs to verify training succeeded

## Quick Reference

### Generate Predictions (Fast)
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=true&sports=NFL,NHL,NBA,MLB" -Method POST
```

### Check Health
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/health"
```

### Get Predictions
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NHL"
```

## Next Steps After Predictions Are Generated

1. ✅ Test from Lovable - Predictions should now appear
2. ✅ Set up daily automation - Use Lovable cron or Railway cron
3. ✅ Monitor predictions - Check daily for new predictions
