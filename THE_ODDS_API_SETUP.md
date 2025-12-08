# The Odds API Integration Guide

## Overview

Your Railway API now integrates with **The Odds API** to fetch real-time betting odds for NFL, NHL, NBA, and MLB games. This provides accurate, up-to-date odds for edge calculations.

## Your API Key

You have **20,000 calls per month** - more than enough for daily operations!

## Setup

### 1. Add API Key to Railway

In Railway Dashboard:
1. Go to your service → **Variables**
2. Add new variable:
   - **Name:** `ODDS_API_KEY`
   - **Value:** Your The Odds API key
3. **Save** - Railway will automatically redeploy

### 2. Verify It's Working

After adding the key, trigger the daily workflow:
```bash
POST https://moose-picks-api-production.up.railway.app/api/trigger-daily-workflow?train=false&predict=false
```

Check the logs - you should see:
```
[3/5] Fetching odds from The Odds API...
Fetching NFL odds...
  ✓ Updated odds for X NFL games
The Odds API: 150 used, 19850 remaining this month
```

## How It Works

### Automatic Integration

The daily workflow now includes:
1. Settle yesterday's predictions
2. Fetch today's games (ESPN)
3. **Fetch odds from The Odds API** ← NEW!
4. Train models
5. Generate predictions

### What Gets Updated

For each game, The Odds API provides:
- **Moneyline odds** (home/away)
- **Spread odds** and lines
- **Totals (over/under)** odds and lines

These are automatically stored in your database and used for:
- Edge calculations
- Prediction recommendations
- Kelly criterion bet sizing

## API Usage

### Monthly Limit: 20,000 calls

**Daily usage estimate:**
- 4 sports × 1 call per day = 4 calls/day
- 4 calls/day × 30 days = **120 calls/month**
- **Well within your 20k limit!**

### Rate Limiting

The Odds API allows:
- **500 requests per day** (free tier)
- You're using ~4 per day - plenty of headroom

## Manual Odds Fetching

You can also fetch odds manually via API endpoint (if you add one):

```python
from app.odds_api.client import fetch_and_update_game_odds

# Fetch NFL odds for today
updated = fetch_and_update_game_odds("NFL", "2024-12-08")

# Fetch NHL odds for specific date
updated = fetch_and_update_game_odds("NHL", "2024-12-10")
```

## Supported Sports

All 4 sports are supported:
- ✅ **NFL** - `americanfootball_nfl`
- ✅ **NHL** - `icehockey_nhl`
- ✅ **NBA** - `basketball_nba`
- ✅ **MLB** - `baseball_mlb`

## Markets Fetched

For each sport, we fetch:
- **Moneyline** (h2h) - Win/loss odds
- **Spreads** - Point spreads with odds
- **Totals** - Over/under with odds

## Data Quality

The Odds API provides:
- **Multiple bookmakers** - We use the first available (you can enhance to use best odds)
- **Real-time updates** - Odds are current
- **Accurate lines** - Professional sportsbook data

## Monitoring Usage

Check your usage in Railway logs:
```
The Odds API: 150 used, 19850 remaining this month
```

Or check The Odds API dashboard:
- https://the-odds-api.com/liveapi/guides/v4/

## Troubleshooting

### "ODDS_API_KEY not set"
- Make sure you added `ODDS_API_KEY` to Railway environment variables
- Redeploy after adding the variable

### "No games found to update"
- Games need to be fetched from ESPN first
- Make sure Step 2 (fetch games) runs before Step 3 (fetch odds)

### "Error fetching odds"
- Check your API key is valid
- Verify you haven't exceeded rate limits
- Check The Odds API status page

### Low Update Count
- Team name matching might be failing
- The Odds API uses different team names than ESPN
- The system does fuzzy matching, but some games might not match

## Enhancing the Integration

### Use Best Odds (Future Enhancement)

Currently uses first bookmaker. You could enhance to:
- Compare odds across all bookmakers
- Use the best odds for each market
- Track which bookmaker has best odds

### Add More Markets

The Odds API supports:
- `h2h` - Moneyline
- `spreads` - Point spreads
- `totals` - Over/under
- `outrights` - Championship/futures
- `player_props` - Player-specific bets

## Cost Efficiency

With 20k calls/month:
- **Current usage:** ~120 calls/month
- **Available:** 19,880 calls/month
- **Can add:** More frequent updates, multiple bookmakers, historical odds

## Next Steps

1. ✅ Add `ODDS_API_KEY` to Railway
2. ✅ Test the daily workflow
3. ✅ Verify odds are being fetched
4. ✅ Check edge calculations use real odds
5. ✅ Monitor usage monthly

Your predictions will now use **real-time odds** from professional sportsbooks!
