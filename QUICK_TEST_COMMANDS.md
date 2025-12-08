# Quick Test Commands - PowerShell & curl

## Your Railway URL
```
https://moose-picks-api-production.up.railway.app
```

## ⚠️ PowerShell Syntax Fix

**WRONG (causes error):**
```powershell
POST https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=false
```

**CORRECT (wrap URL in quotes):**
```powershell
Invoke-WebRequest -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=false" -Method POST
```

## Quick Test Commands

### 1. Health Check

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/health"
```

**curl:**
```bash
curl https://moose-picks-api-production.up.railway.app/api/health
```

**Browser:**
```
https://moose-picks-api-production.up.railway.app/api/health
```

### 2. Trigger Workflow (Just Fetch Games & Odds - Fast!)

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=false" -Method POST
```

**curl:**
```bash
curl -X POST "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=false"
```

This will:
- ✅ Settle yesterday's predictions
- ✅ Fetch today's games from ESPN
- ✅ Fetch odds from The Odds API
- ❌ Skip training (fast!)
- ❌ Skip predictions (fast!)

### 3. Trigger Workflow (Just Predictions - No Training)

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=true" -Method POST
```

**curl:**
```bash
curl -X POST "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=true"
```

### 4. Full Workflow (All Sports)

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=true&predict=true&sports=NFL,NHL,NBA,MLB" -Method POST
```

**curl:**
```bash
curl -X POST "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=true&predict=true&sports=NFL,NHL,NBA,MLB"
```

### 5. Get Predictions

**NHL - Next 3 Days:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/next-days?sport=NHL&days=3"
```

**NFL - Current Week:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/week?sport=NFL"
```

**NBA - Next 7 Days:**
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/next-days?sport=NBA&days=7"
```

## PowerShell Tips

### Use Invoke-RestMethod (Recommended)
Returns JSON directly, easier to read:
```powershell
$result = Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/health"
$result.status
$result.games_in_db
```

### Use Invoke-WebRequest
Returns full response object:
```powershell
$response = Invoke-WebRequest -Uri "https://moose-picks-api-production.up.railway.app/api/health" -Method GET
$response.Content | ConvertFrom-Json
```

### Always Quote URLs with Query Parameters
```powershell
# ✅ CORRECT
Invoke-RestMethod -Uri "https://api.com/endpoint?param1=value1&param2=value2" -Method POST

# ❌ WRONG (PowerShell will error on &)
Invoke-RestMethod -Uri https://api.com/endpoint?param1=value1&param2=value2 -Method POST
```

## Common Errors

### "The ampersand (&) character is not allowed"
**Fix:** Wrap the entire URL in double quotes:
```powershell
# Wrong
POST https://api.com/endpoint?train=false&predict=false

# Right
Invoke-WebRequest -Uri "https://api.com/endpoint?train=false&predict=false" -Method POST
```

### "Method Not Allowed"
**Fix:** Make sure you're using `-Method POST`:
```powershell
Invoke-RestMethod -Uri "https://api.com/endpoint" -Method POST
```

## Testing Workflow

### Step 1: Check Health
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/health"
```

### Step 2: Fetch Games & Odds (Fast Test)
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=false" -Method POST
```

### Step 3: Check Logs in Railway
- Go to Railway dashboard
- View deployment logs
- See games and odds being fetched

### Step 4: Generate Predictions
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=true" -Method POST
```

### Step 5: Get Predictions
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NFL"
```
