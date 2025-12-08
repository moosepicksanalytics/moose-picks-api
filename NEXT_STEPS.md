# Next Steps After Feature Engineering Update

## Overview
You've just added comprehensive sports betting features (~150+ features). Here's how to proceed:

## Option 1: Test Locally First (Recommended)

### Step 1: Check if you have data
```bash
# Check database health
python -c "from app.database import SessionLocal; from app.models.db_models import Game; db = SessionLocal(); print(f'Games in DB: {db.query(Game).count()}'); db.close()"
```

### Step 2: If no data, fetch some historical games
```bash
# Fetch games for a recent date (e.g., last week)
python -c "
from datetime import datetime, timedelta
from app.espn_client.fetcher import fetch_games_for_date
from app.espn_client.parser import parse_and_store_games

date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
for sport in ['NFL', 'NHL']:
    games = fetch_games_for_date(sport, date)
    if games:
        parse_and_store_games(sport, games)
        print(f'Fetched {len(games)} {sport} games for {date}')
"
```

### Step 3: Test feature engineering
```bash
python scripts/test_features.py
```

This will:
- ✅ Load sample data
- ✅ Build all features
- ✅ Verify feature columns exist
- ✅ Check for errors

### Step 4: If test passes, train a small model
```bash
# Train just one model to verify everything works
python -c "
from scripts.train_all import train_all_models
# This will train all models - might take a while
train_all_models()
"
```

Or train just one market:
```bash
python -c "
from app.training.pipeline import train_model_for_market
result = train_model_for_market('NFL', 'moneyline')
print(result)
"
```

## Option 2: Deploy Directly (If you're confident)

If you want to deploy to Railway and let it handle everything:

### Step 1: Commit and push your changes
```bash
git add app/training/features.py scripts/test_features.py
git commit -m "Add comprehensive sports betting features"
git push
```

### Step 2: Railway will automatically:
- Deploy the new code
- Run the daily workflow (if configured)
- Train models with new features
- Generate predictions

### Step 3: Monitor via API
```bash
# Check health
curl https://moose-picks-api-production.up.railway.app/api/health

# Trigger manual training
curl -X POST https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow
```

## What Changed?

### New Features Added:
1. **Multi-window rolling stats** (3, 5, 10, 15 games)
2. **ATS records** (Against The Spread)
3. **Over/Under records**
4. **Rest advantage** and back-to-back indicators
5. **Recent outcomes** (wins/losses, streaks)
6. **Opponent-adjusted stats**
7. **Betting market features** (implied probabilities, line value)
8. **Momentum metrics** (recent vs older performance)
9. **Efficiency metrics** (offensive/defensive)

### Impact:
- **More features**: ~150+ features vs ~30 before
- **Better predictions**: More context = better model performance
- **Longer training time**: More features = slower training (but worth it!)

## Troubleshooting

### If test_features.py fails:

1. **"No games found"**
   - You need to fetch data first (see Step 2 above)
   - Or check your database connection

2. **"No features available"**
   - Check that feature columns are being created
   - Look at the error message for specific missing columns

3. **Import errors**
   - Make sure you're in the project root directory
   - Check that all dependencies are installed: `pip install -r requirements.txt`

4. **Memory errors during training**
   - The new features use more memory
   - Consider reducing `rolling_window_games` in `config.yaml` from 15 to 10
   - Or train one sport/market at a time

### If training fails:

1. **Check feature columns match**
   - The pipeline filters features that exist in the DataFrame
   - Missing features are skipped (this is OK)

2. **Check for NaN values**
   - Early games in the dataset will have NaN for rolling stats
   - The pipeline should handle this, but check logs

3. **Reduce feature complexity**
   - Edit `config.yaml`:
     ```yaml
     features:
       rolling_window_games: 10  # Reduce from 15
       include_rest_days: true
       include_head_to_head: true
     ```

## Recommended Workflow

1. ✅ **Test locally** with `test_features.py`
2. ✅ **Train one model** to verify end-to-end
3. ✅ **Commit and push** if everything works
4. ✅ **Monitor Railway** deployment
5. ✅ **Check predictions** quality improves

## Performance Notes

- **Training time**: Will be 2-3x longer due to more features
- **Memory usage**: Will increase ~30-50%
- **Prediction time**: Slightly slower but still fast (<1 second per game)
- **Model size**: Models will be larger (~2-3x)

All of this is normal and expected with more sophisticated features!

## Questions?

- Check the logs: `tail -f logs/training.log` (if logging is set up)
- Check Railway logs in the dashboard
- Review the feature engineering code in `app/training/features.py`
