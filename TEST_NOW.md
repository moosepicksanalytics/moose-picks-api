# How to Test the Fixes - Step by Step

## 🚀 Quick Test (5 minutes)

### Step 1: Generate Predictions
```powershell
# Trigger prediction generation for NHL (no training, just predictions)
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=true&sports=NHL&min_edge=0.0" -Method POST
```

**Wait 1-2 minutes** for predictions to generate.

### Step 2: Check Predictions
```powershell
# Get latest predictions
$preds = Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NHL&limit=20" -Method GET

# Show all predictions with their edges
$preds | ForEach-Object {
    $edge = [math]::Round($_.moneyline.best_edge * 100, 1)
    $side = if ($_.moneyline.best_side) { $_.moneyline.best_side } else { "null" }
    Write-Host "$($_.away_team) @ $($_.home_team): edge=$edge%, side=$side"
}
```

### Step 3: Verify Negative Edges Have Null Side
```powershell
$preds = Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NHL&limit=50" -Method GET

# Find negative edge predictions
$negEdges = $preds | Where-Object { $_.moneyline.best_edge -lt 0 }

Write-Host "`nFound $($negEdges.Count) predictions with negative edges" -ForegroundColor Yellow

# Check if any have a side (they shouldn't!)
$bad = $negEdges | Where-Object { $_.moneyline.best_side -ne $null }

if ($bad.Count -eq 0) {
    Write-Host "✅ PASS: All negative edge predictions correctly have best_side = null" -ForegroundColor Green
} else {
    Write-Host "❌ FAIL: $($bad.Count) negative edge predictions still have a side!" -ForegroundColor Red
    $bad | ForEach-Object {
        Write-Host "  $($_.away_team) @ $($_.home_team): edge=$([math]::Round($_.moneyline.best_edge*100, 1))%, side=$($_.moneyline.best_side)"
    }
}
```

### Step 4: Check Railway Logs
Go to Railway dashboard → Your service → Logs

**Look for:**
- ✅ `⚠️  Negative edge detected (-X.X%) - not recommending any side`
- ✅ `⚠️  Skipping invalid odds: both positive`
- ✅ `⚠️  Storing prediction with negative edge`

## ✅ What Success Looks Like

1. **Negative edge predictions** → `best_side: null` (or missing)
2. **Positive edge predictions** → `best_side: "home"` or `"away"`
3. **Warnings in logs** about negative edges
4. **No invalid odds** (both positive) being accepted

## 🔍 Manual Verification

Pick one prediction and verify:
1. Get the prediction: `$p = $preds[0]`
2. Check edge: `$p.moneyline.best_edge`
3. Check side: `$p.moneyline.best_side`
4. If edge < 0, side should be `null`
5. If edge > 0, side should be `"home"` or `"away"`

## 📊 Full Test Script

Run this all at once:

```powershell
# 1. Generate predictions
Write-Host "Generating predictions..." -ForegroundColor Cyan
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=true&sports=NHL&min_edge=0.0" -Method POST | Out-Null

# 2. Wait
Write-Host "Waiting 30 seconds for predictions to generate..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# 3. Check results
Write-Host "`nChecking predictions..." -ForegroundColor Cyan
$preds = Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NHL&limit=50" -Method GET

if ($preds.Count -eq 0) {
    Write-Host "No predictions found. Wait longer or check Railway logs." -ForegroundColor Yellow
    exit
}

# 4. Find negative edges
$negEdges = $preds | Where-Object { $_.moneyline.best_edge -lt 0 }
$posEdges = $preds | Where-Object { $_.moneyline.best_edge -gt 0 }

Write-Host "`nResults:" -ForegroundColor Green
Write-Host "  Total predictions: $($preds.Count)"
Write-Host "  Positive edges: $($posEdges.Count)"
Write-Host "  Negative edges: $($negEdges.Count)"

# 5. Verify negative edges have null side
$bad = $negEdges | Where-Object { $_.moneyline.best_side -ne $null }

if ($bad.Count -eq 0) {
    Write-Host "`n✅ TEST PASSED: All negative edge predictions have best_side = null" -ForegroundColor Green
} else {
    Write-Host "`n❌ TEST FAILED: $($bad.Count) negative edge predictions have a side!" -ForegroundColor Red
}
```

