# Model Accuracy Fixes - Implementation Summary

This document summarizes the implementation of model accuracy diagnostics, recalibration, and result tracking features.

## ✅ Completed Tasks

### 1. Diagnostic Script (`scripts/diagnose_model_accuracy.py`)
- Loads predictions from CSV exports or database
- Queries game results from database
- Matches predictions to actual results
- Calculates:
  - Win rate by sport and market
  - Confidence calibration (ECE)
  - Edge statistics
- Outputs:
  - Console report with formatted metrics
  - CSV with detailed results
  - JSON summary

**Usage:**
```bash
python scripts/diagnose_model_accuracy.py --sport NFL
python scripts/diagnose_model_accuracy.py --sport NFL --source db
```

### 2. Recalibration Script (`scripts/recalibrate_model.py`)
- Loads trained models from `/models` directory
- Splits data into training, calibration, and test sets
- Applies Platt scaling (logistic regression calibration)
- Tests calibration on unseen test set
- Saves calibrated model with `_CALIBRATED_` suffix
- Reports ECE improvement

**Usage:**
```bash
python scripts/recalibrate_model.py --sport NFL --model models/NFL_spread_20251207_191802.pkl
```

### 3. Database Model Updates (`app/models/db_models.py`)
- Added `settled_result` column to `Prediction` model
- Kept `result` column for backward compatibility
- Updated settling logic to set both columns

### 4. API Endpoints (`app/api_endpoints.py`)

#### `POST /api/settle-predictions?sport={sport}`
- Settles all unsettled predictions for a sport
- Matches predictions to game results
- Marks predictions as won/lost/push
- Returns settlement statistics

**Example:**
```bash
POST /api/settle-predictions?sport=NFL
```

**Response:**
```json
{
  "sport": "NFL",
  "settled": 45,
  "accurate": 23,
  "wins": 23,
  "losses": 22,
  "pushes": 0,
  "win_rate": 0.511
}
```

#### `GET /api/metrics/accuracy/{sport}?days=30`
- Returns accuracy metrics for settled predictions
- Defaults to last 30 days
- Includes win rate, vs random, sample size adequacy

**Example:**
```bash
GET /api/metrics/accuracy/NFL?days=30
```

**Response:**
```json
{
  "sport": "NFL",
  "total_predictions": 234,
  "wins": 122,
  "losses": 112,
  "pushes": 0,
  "win_rate": 0.521,
  "vs_random": 0.021,
  "sample_size_adequate": true
}
```

#### `POST /api/jobs/settle-daily`
- Scheduled job endpoint for daily settling
- Settles predictions for all sports
- Requires API key authentication
- Should be called daily at 11 PM UTC

**Example:**
```bash
POST /api/jobs/settle-daily
# Requires API key in header
```

### 5. Settling Logic Updates (`app/prediction/settling.py`)
- Updated to set both `result` and `settled_result` columns
- Maintains backward compatibility

## 📋 Next Steps

### Database Migration
The `settled_result` column has been added to the model, but you'll need to run a database migration:

**Option 1: Using Alembic (if configured)**
```bash
alembic revision --autogenerate -m "add_settled_result_column"
alembic upgrade head
```

**Option 2: Manual SQL (PostgreSQL)**
```sql
ALTER TABLE predictions ADD COLUMN settled_result VARCHAR;
```

### Railway Cron Configuration
Add to `railway.json`:
```json
{
  "crons": {
    "settle-predictions": {
      "schedule": "0 23 * * *",
      "command": "curl -X POST https://moose-picks-api-production.up.railway.app/api/jobs/settle-daily -H 'X-API-Key: YOUR_API_KEY'"
    }
  }
}
```

## 🧪 Testing

### Run Scripts Locally (via venv)
```bash
# Activate venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Run diagnostic script
python scripts/diagnose_model_accuracy.py --sport NFL

# Run recalibration script
python scripts/recalibrate_model.py --sport NFL --model models/NFL_spread_20251207_191802.pkl
```

### Run Scripts via Railway API

**Diagnostic Script:**
```bash
# Via Railway API (runs in background)
curl -X POST "https://moose-picks-api-production.up.railway.app/api/diagnose-accuracy?sport=NFL&source=csv" \
  -H "X-API-Key: YOUR_API_KEY"

# Or from database
curl -X POST "https://moose-picks-api-production.up.railway.app/api/diagnose-accuracy?sport=NFL&source=db" \
  -H "X-API-Key: YOUR_API_KEY"
```

**Recalibration Script:**
```bash
# Via Railway API (runs in background)
curl -X POST "https://moose-picks-api-production.up.railway.app/api/recalibrate-model?sport=NFL&model=NFL_spread_20251207_191802.pkl" \
  -H "X-API-Key: YOUR_API_KEY"
```

**Other API Endpoints:**
```bash
# Settle predictions
curl -X POST "https://moose-picks-api-production.up.railway.app/api/settle-predictions?sport=NFL" \
  -H "X-API-Key: YOUR_API_KEY"

# Get accuracy metrics
curl "https://moose-picks-api-production.up.railway.app/api/metrics/accuracy/NFL?days=30"

# Daily settling job (requires API key)
curl -X POST "https://moose-picks-api-production.up.railway.app/api/jobs/settle-daily" \
  -H "X-API-Key: YOUR_API_KEY"
```

**PowerShell Examples:**
```powershell
# Diagnostic
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/diagnose-accuracy?sport=NFL&source=csv" `
  -Method POST -Headers @{"X-API-Key"="YOUR_API_KEY"}

# Recalibration
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/recalibrate-model?sport=NFL&model=NFL_spread_20251207_191802.pkl" `
  -Method POST -Headers @{"X-API-Key"="YOUR_API_KEY"}

# Settle predictions
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/settle-predictions?sport=NFL" `
  -Method POST -Headers @{"X-API-Key"="YOUR_API_KEY"}

# Get metrics
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/metrics/accuracy/NFL?days=30"
```

**Note:** Scripts run via API execute in the background. Check Railway logs to see results:
- Diagnostic: Results saved to `exports/{sport}_accuracy_diagnostic_{timestamp}.csv` and `.json`
- Recalibration: Calibrated model saved to `models/{sport}_{market}_CALIBRATED_{timestamp}.pkl`

## 📊 Expected Results

After running the diagnostic script, you should see:
- Win rates around 50-55% (realistic for sports betting)
- ECE < 0.10 for well-calibrated models
- Average confidence around 50-60% (not 83%)
- Edge statistics showing realistic 2-5% edges (not 32%)

After recalibration:
- ECE should improve (decrease)
- Confidence scores should be more realistic
- Calibrated model saved with `_CALIBRATED_` suffix

## 🔍 Notes

- The diagnostic script can work with either CSV exports or database predictions
- The recalibration script requires at least 4 weeks of historical data (2 for calibration, 2 for test)
- All endpoints are production-ready with error handling
- The settling logic handles moneyline, spread, and totals markets correctly

