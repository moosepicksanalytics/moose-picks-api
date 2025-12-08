# Deployment Guide: GitHub + Railway + Lovable

## Overview

Yes! You can absolutely schedule this through Lovable when hosted on Railway. Here are three options:

## Option 1: Railway Cron Jobs (Recommended)

Railway supports cron jobs that can call your API endpoint.

### Setup Steps:

1. **Deploy to Railway:**
   - Connect your GitHub repo to Railway
   - Railway will auto-detect the FastAPI app
   - Deploy!

2. **Add Cron Job in Railway:**
   - Go to your Railway project
   - Add a new service
   - Choose "Cron Job"
   - Set schedule: `0 6 * * *` (daily at 6 AM)
   - Command: 
     ```bash
     curl -X POST https://your-app.railway.app/api/trigger-daily-workflow?train=true&predict=true
     ```

3. **Or use Railway's built-in cron:**
   - Add `railway.json` (already created) to your repo
   - Railway will automatically schedule it

## Option 2: Lovable Scheduled Tasks

Lovable can call your Railway API endpoint on a schedule.

### Setup in Lovable:

1. **Create a Scheduled Task:**
   - In Lovable, go to Scheduled Tasks
   - Create new task
   - Schedule: Daily at 6:00 AM
   - HTTP Request:
     ```
     POST https://your-app.railway.app/api/trigger-daily-workflow
     Body: {
       "train": true,
       "predict": true,
       "sports": "NFL,NHL"
     }
     ```

2. **Or use Lovable's API integration:**
   - Lovable can trigger your endpoint via webhook
   - Set up a webhook trigger in Lovable
   - Point it to your Railway endpoint

## Option 3: GitHub Actions (Alternative)

If you prefer GitHub Actions:

```yaml
# .github/workflows/daily-automation.yml
name: Daily Automation
on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC
  workflow_dispatch:  # Allow manual trigger

jobs:
  automate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Trigger Railway API
        run: |
          curl -X POST ${{ secrets.RAILWAY_API_URL }}/api/trigger-daily-workflow?train=true&predict=true
```

## API Endpoints Available

Once deployed, these endpoints are available:

### 1. Trigger Daily Workflow
```
POST /api/trigger-daily-workflow
Query params:
  - train=true/false (default: true)
  - predict=true/false (default: true)
  - sports=NFL,NHL (optional)
  - min_edge=0.05 (optional)
```

### 2. Health Check
```
GET /api/health
Returns: Database status, game count, model count
```

### 3. Get Latest Predictions
```
GET /api/predictions/latest?sport=NFL&limit=10
Returns: Latest predictions from exports directory
```

### 4. Existing Endpoints
- `GET /predict` - Single game prediction
- `GET /daily-picks` - All picks for a date
- `POST /train` - Manual training trigger
- `GET /models` - List trained models

## Railway Configuration

### Environment Variables

Set these in Railway:
```
DATABASE_URL=postgresql://... (Railway provides this)
MODEL_DIR=models/
ODDS_API_KEY=your_key_here
```

### Database

Railway will provide a PostgreSQL database. Update your connection:
- Railway auto-injects `DATABASE_URL`
- SQLite works locally, PostgreSQL in production

### Persistent Storage

Models and exports need persistent storage:
- Use Railway's volume mounts for `models/` and `exports/`
- Or use S3/cloud storage for models

## Testing Locally

Test the API endpoint locally:
```bash
# Start the server
uvicorn app.main:app --reload

# Trigger workflow
curl -X POST http://localhost:8000/api/trigger-daily-workflow?train=false&predict=true
```

## Monitoring

Check logs in Railway dashboard to see:
- When cron jobs run
- Training progress
- Prediction generation
- Any errors

## Recommended Setup

**Best Practice:**
1. Deploy to Railway
2. Use Railway cron job (Option 1) - most reliable
3. Have Lovable fetch predictions via `/api/predictions/latest`
4. Monitor via `/api/health` endpoint

This gives you:
- ✅ Automated daily training
- ✅ Automated predictions
- ✅ Prediction tracking and settling
- ✅ Lovable integration
- ✅ Monitoring and health checks

## Prediction Tracking System

The system now automatically tracks all predictions and their outcomes:

### How It Works

1. **Prediction Storage**: When predictions are generated, they're automatically stored in the `predictions` table in Railway PostgreSQL
2. **Automatic Settling**: The daily workflow settles yesterday's predictions by comparing them against final game scores
3. **Market Support**: Tracks all markets (moneyline, spread, totals) with proper win/loss/push logic
4. **PnL Tracking**: Calculates profit/loss for each prediction based on odds

### Daily Workflow Steps

The automated daily workflow now includes:

1. **Settle Yesterday's Predictions** - Compares predictions against final scores
2. **Fetch Today's Games** - Gets upcoming games from ESPN
3. **Train Models** (optional) - Retrains on all historical data
4. **Generate Predictions** - Creates predictions and stores them in database

### Database Storage

- **Local Development**: Uses SQLite (`moose.db`) - stored locally
- **Production (Railway)**: Uses PostgreSQL - Railway automatically provides this
- **No GitHub Storage**: Database files are in `.gitignore` - never committed
- **Persistent**: Railway PostgreSQL persists independently - no computer needed

### Accessing Prediction Data

Predictions are stored in the `predictions` table with:
- Game ID, sport, market type
- Model probabilities (home_win_prob, spread_cover_prob, over_prob)
- Model version used
- Settlement status (settled, result, pnl)
- Timestamps (predicted_at, settled_at)

You can query this data via the API or directly from the database to track model performance over time.
