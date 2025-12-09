# Moose Picks ML - Sports Betting Prediction API

A production-ready machine learning pipeline for sports betting predictions across **NFL, NHL, NBA, and MLB**. The system predicts win probabilities, spread covers, totals, and calculates betting edges using advanced feature engineering and ensemble models.

## 🚀 Quick Start

### Railway Deployment

**API URL:** `https://moose-picks-api-production.up.railway.app/api`

**Key Endpoints:**
- `GET /api/health` - Health check
- `POST /api/trigger-daily-workflow` - Run daily workflow (fetch games, odds, train, predict)
- `POST /api/backfill-odds` - Backfill historical odds data
- `GET /api/predictions/latest?sport=NFL` - Get latest predictions
- `GET /api/predictions/next-days?sport=NHL&days=3` - Get predictions for next N days

### Environment Variables (Railway)

Required:
- `DATABASE_URL` - PostgreSQL connection string (auto-provided by Railway)
- `ODDS_API_KEY` - The Odds API key (20k calls/month)

## Features

### ✅ Multi-Sport Support
- **NFL** - Moneyline, Spread, Totals
- **NHL** - Moneyline, Spread, Totals  
- **NBA** - Moneyline, Spread, Totals
- **MLB** - Moneyline, Run Line

### ✅ Advanced ML Pipeline
- **150+ features** per game (rolling stats, rest days, H2H, ATS records, etc.)
- **Sport-specific feature engineering** (NFL division matchups, NHL goalie stats, NBA pace, MLB pitcher stats)
- **Multiple algorithms**: XGBoost, LightGBM, CatBoost, Gradient Boosting
- **Ensemble models** with stacking meta-learner
- **Hyperparameter optimization** with Optuna
- **Temporal train-test splits** (no data leakage)
- **Calibration metrics** (ECE, Brier Score, Log Loss)

### ✅ Betting Strategy
- **Edge calculation** (Model Probability - Implied Probability)
- **Kelly Criterion** bet sizing (fractional Kelly: 1/4)
- **ROI simulation** on test sets
- **Value bet filtering** (min edge threshold)

### ✅ Data Integration
- **ESPN API** - Game schedules and results
- **The Odds API** - Real-time and historical betting odds
- **PostgreSQL** - Production database on Railway
- **Historical data backfill** - Past 5 seasons

## Project Structure

```
moose-picks-api/
├── app/
│   ├── api_endpoints.py          # FastAPI endpoints
│   ├── config.py                 # Settings and sport configs
│   ├── database.py               # SQLAlchemy database setup
│   ├── data_loader.py             # Historical data loading
│   ├── espn_client/               # ESPN API client
│   ├── odds_api/                  # The Odds API client
│   ├── training/
│   │   ├── pipeline.py            # Main training pipeline
│   │   ├── advanced_pipeline.py    # Advanced ML (Optuna, ensembles)
│   │   ├── features.py             # Feature engineering
│   │   ├── sport_feature_engineers.py  # Sport-specific features
│   │   ├── base_feature_engineer.py    # Base feature utilities
│   │   └── evaluate.py             # Model evaluation metrics
│   ├── prediction/                # Prediction engine and storage
│   └── utils/                     # Betting utilities, odds conversion
├── scripts/
│   ├── train_all.py               # Train all models
│   ├── train_advanced.py           # Advanced ML training
│   ├── backfill_historical_data.py # Backfill game data
│   ├── backfill_odds.py            # Backfill historical odds
│   ├── daily_automation.py         # Daily workflow
│   └── export_predictions.py      # Export predictions
├── config.yaml                     # Training configuration
└── requirements.txt                # Python dependencies
```

## Setup

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

**Note:** CatBoost is optional. If installation fails on Windows, the code will work without it (uses XGBoost/LightGBM instead).

### 2. Configure Environment

**For Railway (Production):**
- Set `DATABASE_URL` (auto-provided)
- Set `ODDS_API_KEY` in Railway Variables

**For Local Development:**
```powershell
# Connect to Railway database
$env:DATABASE_URL = "postgresql://postgres:PASSWORD@HOST:PORT/railway"
```

### 3. Configure Training

Edit `config.yaml`:
- Training seasons for each sport
- Feature engineering options
- Model hyperparameters
- Edge thresholds

## Usage

### Daily Workflow (Railway)

```powershell
# Trigger complete daily workflow
POST https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow

# Just fetch games and odds (no training)
POST https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=false

# Train only (no predictions)
POST https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?predict=false
```

### Backfill Historical Odds

**⚠️ Cost Warning:** Historical odds cost 30 credits per date (10x more than current odds).

**Via Railway API:**
```powershell
# Backfill NFL odds for October 2024 (costs ~930 credits)
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/backfill-odds?sport=NFL&start_date=2024-10-01&end_date=2024-10-31" -Method POST

# Backfill all sports for a date range
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/backfill-odds?start_date=2024-10-01&end_date=2024-10-31" -Method POST
```

**Locally:**
```powershell
# Dry run first (no cost)
python scripts/backfill_odds.py --sport NFL --start-date 2024-10-01 --end-date 2024-10-31 --dry-run

# Actually backfill
python scripts/backfill_odds.py --sport NFL --start-date 2024-10-01 --end-date 2024-10-31
```

**Cost:** ~30 credits per date × number of dates. With 20k credits/month, you can backfill ~666 dates (single sport) or ~166 dates (all 4 sports).

### Training Models

**Train all models:**
```powershell
python scripts/train_all.py
```

**Train specific sport/market:**
```powershell
python -c "from app.training.pipeline import train_model_for_market; train_model_for_market('NFL', 'moneyline')"
```

**Advanced training (Optuna + Ensembles):**
```powershell
python scripts/train_advanced.py --sport NFL --market moneyline --use-optuna --use-ensemble
```

### Get Predictions

**Latest predictions:**
```powershell
GET https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NFL
```

**Next 3 days (NHL):**
```powershell
GET https://moose-picks-api-production.up.railway.app/api/predictions/next-days?sport=NHL&days=3
```

**Date range:**
```powershell
GET https://moose-picks-api-production.up.railway.app/api/predictions/date-range?sport=NFL&start_date=2024-12-15&end_date=2024-12-21
```

## Model Performance

### Expected Accuracies (After Data Leakage Fixes)

| Market | Expected Accuracy | Notes |
|--------|-------------------|-------|
| Moneyline | 55-65% | Realistic for betting markets |
| Spread | 50-55% | Close to random (spread betting is hard) |
| Totals | 50-55% | Close to random |

**⚠️ If you see 100% accuracy, there's still data leakage!**

### Evaluation Metrics

Priority order:
1. **ROI** - Simulated profitability using Kelly sizing
2. **Calibration (ECE)** - Expected Calibration Error
3. **Value bet accuracy** - Accuracy on bets with edge >= 5%
4. **Overall accuracy** - General model performance
5. **Precision/Recall** - Classification metrics

## Data Leakage Prevention

The system has been extensively audited to prevent data leakage. Removed features include:
- `spread_value`, `totals_value` - Directly encode target
- `spread_line`, `over_under_line` - Allow edge reconstruction
- Point differential features for spread market
- Strength metrics derived from point differentials
- Sport-specific leakage (Corsi, time of possession, turnover differential)

**Validation:** Automated leakage detection checks for high correlations and known leakage patterns.

## Feature Engineering

### Base Features (All Sports)
- Rolling statistics (3, 5, 10, 15 game windows)
- Win rates, points/goals for/against
- Rest days and rest advantage
- Home/away splits
- Head-to-head records
- ATS (Against The Spread) records
- Over/Under records
- Opponent-adjusted statistics

### Sport-Specific Features

**NFL:**
- Division matchup indicators
- Estimated third-down conversion rates
- Estimated redzone efficiency

**NHL:**
- Estimated goalie save percentage
- Estimated power play efficiency

**NBA:**
- Estimated pace (possessions per game)
- Estimated effective field goal percentage

**MLB:**
- Estimated pitcher ERA/WHIP/K9
- Estimated batter OPS

## API Endpoints

### Health Check
```
GET /api/health
```

### Daily Workflow
```
POST /api/trigger-daily-workflow
  ?train=true
  &predict=true
  &sports=NFL,NHL
  &min_edge=0.05
```

### Backfill Odds (Historical)
```
POST /api/backfill-odds
  ?sport=NFL                    # Optional: omit for all sports
  &start_date=2024-10-01        # Required: YYYY-MM-DD
  &end_date=2024-10-31          # Required: YYYY-MM-DD
  &dry_run=false                # Optional: true to preview only
```
**Note:** Historical odds cost 30 credits per date. Check Railway logs for progress.

### Predictions
```
GET /api/predictions/latest?sport=NFL&limit=10
GET /api/predictions/next-days?sport=NHL&days=3
GET /api/predictions/week?sport=NFL
GET /api/predictions/date-range?sport=NFL&start_date=2024-12-15&end_date=2024-12-21
```

## Troubleshooting

### "No games found for today"
- Check if games exist in database: `GET /api/health`
- Verify ESPN API is working
- Check game dates (ESPN uses UTC)

### "All games have null spreads"
- Expected if odds data wasn't collected historically
- Backfill historical odds: `POST /api/backfill-odds`
- Or wait for daily workflow to collect future odds

### "100% accuracy" (Data Leakage)
- Check for perfect correlations in logs
- Review `DATA_LEAKAGE_FIXES.md` (if still exists)
- Verify no features encode the target directly

### Database Connection Issues
- Verify `DATABASE_URL` is set correctly
- Use **PUBLIC** connection string (not internal `postgres.railway.internal`)
- Check Railway database is not paused

## Local Testing

Test locally without backfilling data:

```powershell
# 1. Connect to Railway database
$env:DATABASE_URL = "postgresql://postgres:PASSWORD@HOST:PORT/railway"

# 2. Test training
python -c "from app.training.pipeline import train_model_for_market; train_model_for_market('NFL', 'moneyline')"

# 3. Use test scripts
.\test_local.ps1 -Sport NFL -Market moneyline
python test_local.py NFL moneyline
```

## Cost Management

**The Odds API:**
- Current odds: 1 credit per region per market
- Historical odds: **10 credits per region per market**
- 3 markets (h2h, spreads, totals) × 1 region = 3 credits (current) or 30 credits (historical)
- Daily workflow: ~4 calls/day = ~120 credits/month
- Historical backfill: 30 credits per date

**With 20k credits/month:**
- Daily operations: ~120 credits/month
- Can backfill: ~666 dates (single sport) or ~166 dates (all 4 sports)

## Deployment

**Railway:**
- Auto-deploys on git push
- Uses `Procfile` for process definition
- PostgreSQL database auto-provisioned
- Environment variables set in Railway dashboard

**Lovable Integration:**
- API endpoints ready for frontend
- Predictions available via REST API
- Real-time updates via daily workflow

## License

MIT
