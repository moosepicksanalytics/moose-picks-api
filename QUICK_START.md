# Quick Start Guide - Your Railway API

## Your API URL
```
https://moose-picks-api-production.up.railway.app/api
```

## Quick Test Commands

### PowerShell (Windows) - CORRECT Syntax

**⚠️ Always quote URLs with `&` in PowerShell!**

```powershell
# Test health check
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/health"

# Trigger workflow (fetch games + odds, no training)
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=false" -Method POST

# Trigger workflow (full: train + predict)
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=true&predict=true&sports=NFL,NHL,NBA,MLB" -Method POST

# Get NHL predictions (next 3 days)
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/next-days?sport=NHL&days=3"

# Get NFL predictions (current week)
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/week?sport=NFL"
```

### curl (if available)
```bash
# Health check
curl https://moose-picks-api-production.up.railway.app/api/health

# Trigger workflow
curl -X POST "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=true"
```

### Browser
Just open these URLs:
- Health: `https://moose-picks-api-production.up.railway.app/api/health`
- Latest NFL: `https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NFL`

## Setup Checklist

### 1. Add The Odds API Key
- Railway Dashboard → Your Service → Variables
- Add: `ODDS_API_KEY` = your key
- Save (auto-redeploys)

### 2. Test the API
```powershell
# Quick test
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/health"
```

### 3. Trigger First Workflow
```powershell
# Fetch games and odds (no training yet)
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=false" -Method POST
```

### 4. Check Logs
- Railway Dashboard → Deployments → Latest → View Logs
- Should see games fetched and odds updated

## All 4 Sports Supported

- **NFL** - Weekly schedule
- **NHL** - Daily games  
- **NBA** - Daily games
- **MLB** - Daily games

## API Endpoints

| Endpoint | Use Case |
|----------|----------|
| `/api/health` | Check system status |
| `/api/predictions/latest?sport={SPORT}` | Latest predictions |
| `/api/predictions/next-days?sport=NHL&days=3` | NHL next 3 days |
| `/api/predictions/week?sport=NFL` | NFL current week |
| `/api/trigger-daily-workflow` | Run daily workflow |

See `API_ENDPOINTS_REFERENCE.md` for complete docs.
