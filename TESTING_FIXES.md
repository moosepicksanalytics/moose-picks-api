# Testing Negative Edge Fixes

## ✅ Quick Test Commands

### Step 1: Generate Predictions (Already Started)
The workflow has been triggered. Wait 1-2 minutes, then check the results.

### Step 2: Check Latest Predictions
```powershell
# Get latest NHL predictions and check for negative edges
$predictions = Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NHL&limit=20" -Method GET
$predictions | Where-Object { $_.moneyline.best_edge -lt 0 } | ForEach-Object {
    Write-Host "Game: $($_.away_team) @ $($_.home_team)" -ForegroundColor Yellow
    Write-Host "  Edge: $($_.moneyline.best_edge)" -ForegroundColor Red
    Write-Host "  Best Side: $($_.moneyline.best_side)" -ForegroundColor $(if ($_.moneyline.best_side -eq $null) { "Green" } else { "Red" })
}
```

### Step 3: Verify All Negative Edges Have Null Side
```powershell
$predictions = Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NHL&limit=50" -Method GET
$negativeEdges = $predictions | Where-Object { $_.moneyline.best_edge -lt 0 }
$withSide = $negativeEdges | Where-Object { $_.moneyline.best_side -ne $null }

if ($withSide.Count -eq 0) {
    Write-Host "✅ PASS: All negative edge predictions have best_side = null" -ForegroundColor Green
} else {
    Write-Host "❌ FAIL: Found $($withSide.Count) negative edge predictions with a side!" -ForegroundColor Red
    $withSide | ForEach-Object { Write-Host "  $($_.away_team) @ $($_.home_team): edge=$($_.moneyline.best_edge), side=$($_.moneyline.best_side)" }
}
```

## Quick Test Plan

### 1. Test Locally (Recommended First)

Generate predictions for a specific sport and date to see the warnings in action:

```bash
# Test NHL moneyline predictions for today
python scripts/export_predictions.py --sport NHL --date 2025-01-10 --min-edge 0.0

# Test NFL moneyline predictions
python scripts/export_predictions.py --sport NFL --date 2025-01-10 --min-edge 0.0
```

**What to look for:**
- ✅ Warnings like: `⚠️  Negative edge detected (-X.X%) - not recommending any side`
- ✅ Predictions with `best_side: null` when edge is negative
- ✅ Only positive edge predictions should have a `best_side` value

### 2. Test via Railway API

#### Option A: Generate Predictions Only (No Training)
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=true&sports=NHL&min_edge=0.0" -Method POST
```

#### Option B: Full Workflow (Training + Predictions)
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=true&predict=true&sports=NHL" -Method POST
```

**Check Railway logs** for:
- Warnings about negative edges
- Odds parsing messages
- Validation warnings

### 3. Verify Predictions in Database

Check the latest predictions to ensure negative edges are handled correctly:

```powershell
# Get latest NHL predictions
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NHL&limit=20" -Method GET | ConvertTo-Json -Depth 10
```

**What to verify:**
- ✅ Predictions with negative edges should have `side: null` or no `side` field
- ✅ Only predictions with positive edges should have a `side` value
- ✅ Edge values should be realistic (not -40% or similar extreme values)

### 4. Test Odds Parsing

Check if odds are being parsed correctly:

```powershell
# Check if games have valid odds
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/health" -Method GET | ConvertTo-Json
```

Look for games with valid `home_moneyline` and `away_moneyline` values.

### 5. Manual Verification

1. **Check a specific prediction:**
   - Find a game with a prediction
   - Calculate edge manually: `edge = model_prob - implied_prob`
   - Verify that if edge < 0, `best_side` is null

2. **Check logs for warnings:**
   - Railway logs should show warnings like:
     - `⚠️  Negative edge detected (-X.X%) - not recommending any side`
     - `⚠️  Skipping invalid odds: both positive`
     - `⚠️  Storing prediction with negative edge`

## Expected Behavior

### ✅ Correct Behavior:
- Predictions with **positive edges** → `best_side: "home"` or `"away"`
- Predictions with **negative edges** → `best_side: null`
- Warnings logged for negative edges
- Invalid odds (both positive) are rejected
- Fallback to other bookmakers if Pinnacle odds unavailable

### ❌ Issues to Watch For:
- Predictions with negative edges still having a `best_side` value
- No warnings about negative edges
- Invalid odds being accepted
- Missing odds data (should try multiple bookmakers)

## Quick Test Script

Run this to test locally:

```bash
# Test NHL moneyline
python scripts/export_predictions.py --sport NHL --date $(date +%Y-%m-%d) --min-edge 0.0 2>&1 | grep -E "(Negative edge|best_side|⚠️)"

# Check exports
cat exports/NHL_*.json | jq '.[] | select(.moneyline.best_edge < 0) | {game: .home_team + " vs " + .away_team, edge: .moneyline.best_edge, side: .moneyline.best_side}'
```

This will show all predictions with negative edges and verify they have `best_side: null`.

