# Railway Setup Guide - Cron Jobs & Manual Testing

## Your App is Live! 🎉

Your Railway app URL should look like: `https://your-app-name.railway.app`

### ⚠️ If You Can't Generate Domain (Port Issue)

If Railway says it can't generate a domain without knowing the port:

1. **Check Service Type:**
   - Go to Railway Dashboard → Your Service → Settings
   - Make sure **Service Type** is set to **"Web Service"** (not "Worker")
   
2. **Verify Procfile is Detected:**
   - Settings → Build & Deploy
   - Should show: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - If not, Railway should auto-detect from `Procfile`

3. **Generate Domain:**
   - Settings → Networking → **Generate Domain**
   - Railway will automatically assign a port and create the domain

4. **If Still Not Working:**
   - Make sure `Procfile` is in your project root
   - Redeploy: Railway Dashboard → Deployments → Redeploy
   - The `railway.json` file should also help Railway detect the configuration

## Quick Manual Test (Do This First!)

### 1. Check Health
```bash
curl https://your-app-name.railway.app/api/health
```

Or open in browser: `https://your-app-name.railway.app/api/health`

You should see:
```json
{
  "status": "healthy",
  "database": "connected",
  "games_in_db": 0,
  "models_trained": 0,
  "timestamp": "..."
}
```

### 2. Manually Trigger Daily Workflow

**Option A: Using curl (Terminal/PowerShell)**
```bash
# Full workflow (train + predict)
curl -X POST "https://your-app-name.railway.app/api/trigger-daily-workflow?train=true&predict=true&sports=NFL,NHL"

# Just predictions (no training)
curl -X POST "https://your-app-name.railway.app/api/trigger-daily-workflow?train=false&predict=true"

# Just training (no predictions)
curl -X POST "https://your-app-name.railway.app/api/trigger-daily-workflow?train=true&predict=false"
```

**Option B: Using Browser/Postman**
- Method: `POST`
- URL: `https://your-app-name.railway.app/api/trigger-daily-workflow?train=true&predict=true&sports=NFL,NHL`
- No body needed (all params in URL)

**Option C: Using PowerShell (Windows)**
```powershell
Invoke-WebRequest -Uri "https://your-app-name.railway.app/api/trigger-daily-workflow?train=true&predict=true" -Method POST
```

### 3. Check Logs
- Go to Railway dashboard
- Click on your service
- Click "Deployments" → Latest deployment → "View Logs"
- You'll see the workflow running in real-time

### 4. Check Results
```bash
# Get latest predictions
curl https://your-app-name.railway.app/api/predictions/latest?sport=NFL

# Check health again (should show games and models)
curl https://your-app-name.railway.app/api/health
```

---

## Setting Up Automated Cron Job

### Method 1: Railway Cron Service (Recommended)

1. **In Railway Dashboard:**
   - Go to your project
   - Click "+ New" → "Cron Job"
   - Name it: `daily-ml-workflow`

2. **Configure the Cron:**
   - **Schedule:** `0 6 * * *` (Daily at 6 AM UTC)
     - Or `0 2 * * *` for 2 AM UTC
     - Or `0 10 * * *` for 10 AM UTC
   - **Command:**
     ```bash
     curl -X POST https://your-app-name.railway.app/api/trigger-daily-workflow?train=true&predict=true&sports=NFL,NHL
     ```
   - **Environment:** Same as your main service

3. **Save and Deploy**

### Method 2: Railway Cron via railway.json

Create `railway.json` in your project root:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  },
  "cron": {
    "daily-workflow": {
      "schedule": "0 6 * * *",
      "command": "curl -X POST https://your-app-name.railway.app/api/trigger-daily-workflow?train=true&predict=true"
    }
  }
}
```

Then commit and push:
```bash
git add railway.json
git commit -m "Add Railway cron configuration"
git push
```

### Method 3: External Cron Service (cron-job.org, EasyCron, etc.)

1. Sign up for a free cron service
2. Create new cron job:
   - **URL:** `https://your-app-name.railway.app/api/trigger-daily-workflow?train=true&predict=true`
   - **Method:** POST
   - **Schedule:** Daily at your preferred time
   - **Timezone:** Your local timezone

---

## Understanding the Workflow

When you trigger `/api/trigger-daily-workflow`, it runs:

1. **Settle yesterday's predictions** (compare predictions vs actual results)
2. **Fetch today's games** from ESPN
3. **Train models** (if `train=true`) - This takes 5-15 minutes
4. **Generate predictions** (if `predict=true`) - This takes 1-2 minutes

**Total time:** ~10-20 minutes depending on data volume

---

## Monitoring & Debugging

### Check if Workflow is Running
```bash
# Health check shows current status
curl https://your-app-name.railway.app/api/health
```

### View Logs in Railway
1. Go to Railway dashboard
2. Click your service
3. Click "Deployments" → Latest
4. Click "View Logs"
5. You'll see real-time output from the workflow

### Common Issues

**"No games found"**
- The workflow needs to fetch games first
- Run manually with `predict=true` to fetch today's games
- Or wait for the cron to run

**"No models trained"**
- Models are only trained if `train=true`
- First run will take longer (training all models)
- Subsequent runs are faster (only retrain if needed)

**"Database connection failed"**
- Check your `DATABASE_URL` environment variable in Railway
- Make sure PostgreSQL is provisioned and connected

---

## Quick Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check app status |
| `/api/trigger-daily-workflow` | POST | Run daily workflow |
| `/api/predictions/latest` | GET | Get latest predictions |

### Query Parameters for `/api/trigger-daily-workflow`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train` | bool | `true` | Retrain models |
| `predict` | bool | `true` | Generate predictions |
| `sports` | string | `NFL,NHL` | Comma-separated sports |
| `min_edge` | float | `0.05` | Minimum edge threshold |

### Example URLs

```bash
# Full workflow
https://your-app.railway.app/api/trigger-daily-workflow?train=true&predict=true&sports=NFL,NHL

# Just predictions (skip training)
https://your-app.railway.app/api/trigger-daily-workflow?train=false&predict=true

# Just training (skip predictions)
https://your-app.railway.app/api/trigger-daily-workflow?train=true&predict=false

# Single sport
https://your-app.railway.app/api/trigger-daily-workflow?train=true&predict=true&sports=NFL
```

---

## Next Steps

1. ✅ **Test manually** - Run the workflow once to verify it works
2. ✅ **Set up cron** - Configure automated daily runs
3. ✅ **Monitor first run** - Watch logs to ensure everything works
4. ✅ **Check predictions** - Verify predictions are being generated
5. ✅ **Integrate with Lovable** - Use the API endpoints in your Lovable app

---

## Need Help?

- Check Railway logs for detailed error messages
- Verify environment variables are set correctly
- Make sure your database is connected
- Test endpoints individually to isolate issues
