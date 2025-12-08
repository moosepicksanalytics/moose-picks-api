# Multi-Sport Support - NFL, NHL, NBA, MLB

## ✅ All 4 Sports Fully Supported!

Your ML pipeline now supports **all 4 major sports** with advanced sports betting analytics.

## Sports Configuration

### NFL (National Football League)
- **Markets:** Moneyline, Spread, Totals
- **Schedule:** Weekly (Sunday/Monday)
- **Recommended View:** Current week
- **Min Training Games:** 100

### NHL (National Hockey League)
- **Markets:** Moneyline, Spread, Totals
- **Schedule:** Daily games
- **Recommended View:** Next 3 days
- **Min Training Games:** 100

### NBA (National Basketball Association)
- **Markets:** Moneyline, Spread, Totals
- **Schedule:** Daily games
- **Recommended View:** Next 7 days or current week
- **Min Training Games:** 200

### MLB (Major League Baseball)
- **Markets:** Moneyline, Run Line
- **Schedule:** Daily games
- **Recommended View:** Next 7 days
- **Min Training Games:** 150

## Feature Engineering Support

All sports use the same comprehensive feature set (~150+ features):

### ✅ Rolling Statistics
- Multi-window averages (3, 5, 10, 15 games)
- Win rates, points/goals for/against
- Point differentials
- Momentum metrics

### ✅ Rest Days & Scheduling
- Rest days between games
- Rest advantage calculations
- Back-to-back indicators

### ✅ Home/Away Splits
- Home win rate at home vs away
- Away win rate at home vs away
- Home field advantage metrics

### ✅ Head-to-Head Records
- Historical H2H win rates
- Average margins in matchups

### ✅ ATS Records
- Against The Spread win rates
- Cover probabilities

### ✅ Over/Under Records
- Over/under performance history

### ✅ Betting Market Features
- Implied probabilities from odds
- Spread value calculations
- Totals value calculations

### ✅ Sport-Specific Optimizations
- **NFL/NBA:** Blowout threshold = 14 points
- **MLB:** Blowout threshold = 5 runs
- **NHL:** Blowout threshold = 3 goals
- **NFL/NBA:** Close game threshold = 7 points
- **MLB:** Close game threshold = 2 runs
- **NHL:** Close game threshold = 1 goal

## API Endpoints

### Your Railway URL
```
https://moose-picks-api-production.up.railway.app/api
```

### Endpoints for Each Sport

**NHL - Next 3 Days:**
```
GET /api/predictions/next-days?sport=NHL&days=3&limit=50
```

**NFL - Current Week:**
```
GET /api/predictions/week?sport=NFL&limit=50
```

**NBA - Next 7 Days:**
```
GET /api/predictions/next-days?sport=NBA&days=7&limit=50
```

**MLB - Next 7 Days:**
```
GET /api/predictions/next-days?sport=MLB&days=7&limit=50
```

**Any Sport - Latest:**
```
GET /api/predictions/latest?sport={SPORT}&limit=20
```

**Any Sport - Custom Range:**
```
GET /api/predictions/date-range?sport={SPORT}&start_date=2024-12-08&end_date=2024-12-15
```

## The Odds API Integration

### Setup
1. Add `ODDS_API_KEY` to Railway environment variables
2. Daily workflow automatically fetches odds
3. Real-time odds used for edge calculations

### Usage
- **20,000 calls/month** available
- **~120 calls/month** used (4 sports × 30 days)
- **Plenty of headroom** for more features

### What Gets Fetched
- Moneyline odds (home/away)
- Spread odds and lines
- Totals (over/under) odds and lines

## Training Configuration

All sports are configured in `config.yaml`:

```yaml
training_seasons:
  NFL: [2020, 2021, 2022, 2023, 2024]
  NHL: [2020, 2021, 2022, 2023, 2024]
  NBA: [2020, 2021, 2022, 2023, 2024]
  MLB: [2020, 2021, 2022, 2023, 2024]

models:
  moneyline:
    algorithm: "xGBoost"
    n_estimators: 100
    max_depth: 3
    learning_rate: 0.01
  spread:
    algorithm: "xGBoost"
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1
  totals:
    algorithm: "xGBoost"
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1
```

## Daily Workflow

The automated workflow now includes 5 steps:

1. **Settle yesterday's predictions** - Compare predictions vs results
2. **Fetch today's games** - Get games from ESPN
3. **Fetch odds from The Odds API** - Get real-time betting odds
4. **Train models** - Retrain on all historical data
5. **Generate predictions** - Create predictions with edges

## Files Updated for Multi-Sport Support

### ✅ Core Files
- `app/training/features.py` - Supports all 4 sports
- `app/training/pipeline.py` - Updated docstrings
- `app/espn_client/fetcher.py` - Fixed URL bug, supports all sports
- `app/config.py` - All 4 sports configured
- `app/models/db_models.py` - Supports all sports

### ✅ New Files
- `app/odds_api/client.py` - The Odds API integration
- `app/api_endpoints.py` - Multi-sport date range endpoints
- `API_ENDPOINTS_REFERENCE.md` - Complete API docs
- `THE_ODDS_API_SETUP.md` - Odds API setup guide
- `MULTI_SPORT_SUPPORT.md` - This file

### ✅ Updated Files
- `scripts/daily_automation.py` - Added odds fetching step
- `LOVABLE_MIGRATION_PLAN.md` - Updated with all sports
- `RAILWAY_SETUP.md` - Updated with actual URL

## Testing All Sports

### Test Each Sport
```bash
# NFL
curl https://moose-picks-api-production.up.railway.app/api/predictions/week?sport=NFL

# NHL
curl https://moose-picks-api-production.up.railway.app/api/predictions/next-days?sport=NHL&days=3

# NBA
curl https://moose-picks-api-production.up.railway.app/api/predictions/next-days?sport=NBA&days=7

# MLB
curl https://moose-picks-api-production.up.railway.app/api/predictions/next-days?sport=MLB&days=7
```

### Train All Sports
```bash
POST https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=true&predict=true&sports=NFL,NHL,NBA,MLB
```

## Summary

✅ **All 4 sports supported** (NFL, NHL, NBA, MLB)
✅ **150+ features** for each sport
✅ **The Odds API integrated** for real-time odds
✅ **Multi-sport API endpoints** with date ranges
✅ **Sport-specific optimizations** (blowout thresholds, etc.)
✅ **Advanced analytics** (ATS, totals, H2H, rest days, etc.)

Your system is now a **complete multi-sport betting analytics platform**!
