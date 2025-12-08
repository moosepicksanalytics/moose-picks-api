# Lovable Migration Plan - Remove Old Bet Recommendations System

## Overview

We're migrating from the old bet recommendations system to a new ML-powered API backend. This document outlines what to keep, what to remove, and how to integrate the new system.

## ✅ KEEP (Do Not Remove)

### 1. User Management
- User authentication/login system
- User profiles
- User preferences
- User settings
- All user-related database tables/collections

### 2. Settings & Configuration
- App settings
- User preferences
- Theme/UI settings
- Any configuration files

### 3. Whop Integration
- Whop payment/subscription integration
- Whop webhooks
- Whop API connections
- Subscription management
- All Whop-related code

### 4. Core App Structure
- Navigation/routing
- Layout components
- Authentication flow
- Basic UI components (buttons, forms, etc.)
- Database connections (keep structure, update endpoints)

## ❌ REMOVE (Delete Completely)

### 1. Old Bet Recommendation Components
- Any components that display bet recommendations
- Old prediction cards/displays
- Old recommendation lists
- Old betting UI components
- Any hardcoded bet data

### 2. Old Data Sources
- Old API endpoints for recommendations (if they exist)
- Old prediction generation logic (frontend)
- Old data fetching for bets
- Any mock/fake bet data

### 3. Old Betting Logic
- Client-side bet calculation logic
- Old edge calculation (if in frontend)
- Old probability calculations (if in frontend)

## 🔄 REPLACE WITH NEW SYSTEM

### New API Endpoints

**Base URL:** `https://moose-picks-api-production.up.railway.app/api`

**Supported Sports:** NFL, NHL, NBA, MLB

#### 1. Get Latest Predictions (Latest Export File)
```
GET /api/predictions/latest?sport=NFL&limit=10
GET /api/predictions/latest?sport=NHL&limit=20
GET /api/predictions/latest?sport=NBA&limit=15
GET /api/predictions/latest?sport=MLB&limit=20
```

**Response:**
```json
{
  "sport": "NFL",
  "file": "NFL_2024-12-08_predictions.csv",
  "total_predictions": 15,
  "top_picks": [
    {
      "game_id": "...",
      "home_team": "Kansas City Chiefs",
      "away_team": "Buffalo Bills",
      "market": "moneyline",
      "side": "home",
      "edge": 0.12,
      "home_win_prob": 0.65,
      "recommended_kelly": 0.05,
      ...
    }
  ]
}
```

#### 2. Get Predictions for Next N Days (Perfect for NHL)
```
GET /api/predictions/next-days?sport=NHL&days=3&limit=50
```

**Use Case:** Get NHL predictions for the next 3 days
- `sport`: NHL (or NFL, NBA, MLB)
- `days`: Number of days ahead (default: 3)
- `limit`: Max predictions to return

**Response:**
```json
{
  "sport": "NHL",
  "start_date": "2024-12-08",
  "end_date": "2024-12-10",
  "total_predictions": 25,
  "top_picks": [...]
}
```

#### 3. Get Predictions for Current Week (Perfect for NFL)
```
GET /api/predictions/week?sport=NFL&limit=50
```

**Use Case:** Get NFL predictions for the current week (Monday-Sunday)
- `sport`: NFL (or NHL, NBA, MLB)
- `limit`: Max predictions to return

**Response:**
```json
{
  "sport": "NFL",
  "start_date": "2024-12-02",
  "end_date": "2024-12-08",
  "total_predictions": 30,
  "top_picks": [...]
}
```

#### 4. Get Predictions for Custom Date Range
```
GET /api/predictions/date-range?sport=NBA&start_date=2024-12-08&end_date=2024-12-15&limit=50
```

**Use Case:** Get predictions for any custom date range
- `sport`: NFL, NHL, NBA, or MLB
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD, optional - defaults to start_date)
- `limit`: Max predictions to return

#### 2. Health Check
```
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "games_in_db": 1250,
  "models_trained": 12,
  "timestamp": "2024-12-08T12:00:00"
}
```

#### 3. Trigger Workflow (Admin Only)
```
POST /api/trigger-daily-workflow?train=false&predict=true
```

## 📋 Migration Steps for Lovable

### Step 1: Identify Old Components
1. Search for components with names like:
   - `BetRecommendation`
   - `BetCard`
   - `PredictionCard`
   - `RecommendationList`
   - `BettingDisplay`
   - Any component that shows bet suggestions

2. Search for API calls to old endpoints
3. Search for any hardcoded bet data

### Step 2: Remove Old Components
1. Delete all bet recommendation components
2. Remove routes to bet recommendation pages
3. Remove navigation links to betting features
4. Clean up unused imports

### Step 3: Create New Components

**New Component: `PredictionsDisplay.jsx` (or similar)**
```javascript
import { useState, useEffect } from 'react';

const API_BASE = 'https://moose-picks-api-production.up.railway.app/api';

export function PredictionsDisplay({ 
  sport = 'NFL', 
  viewMode = 'latest' // 'latest', 'next-days', 'week', 'date-range'
}) {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let url;
    
    // Different endpoints based on view mode
    if (viewMode === 'next-days') {
      // NHL: Next 3 days
      const days = sport === 'NHL' ? 3 : 7;
      url = `${API_BASE}/predictions/next-days?sport=${sport}&days=${days}&limit=50`;
    } else if (viewMode === 'week') {
      // NFL: Current week
      url = `${API_BASE}/predictions/week?sport=${sport}&limit=50`;
    } else {
      // Latest predictions
      url = `${API_BASE}/predictions/latest?sport=${sport}&limit=20`;
    }

    fetch(url)
      .then(res => res.json())
      .then(data => {
        setPredictions(data.top_picks || []);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [sport, viewMode]);

  if (loading) return <div>Loading predictions...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!predictions.length) return <div>No predictions available</div>;

  return (
    <div className="predictions-container">
      <h2>
        {viewMode === 'week' && `${sport} - This Week's Picks`}
        {viewMode === 'next-days' && `${sport} - Next ${sport === 'NHL' ? '3' : '7'} Days`}
        {viewMode === 'latest' && `${sport} - Latest Predictions`}
      </h2>
      {predictions.map((pick, idx) => (
        <div key={idx} className="prediction-card">
          <h3>{pick.home_team} vs {pick.away_team}</h3>
          {pick.date && <p>Date: {new Date(pick.date).toLocaleDateString()}</p>}
          <p>Market: {pick.market}</p>
          <p>Side: {pick.side}</p>
          <p>Edge: {(pick.edge * 100).toFixed(1)}%</p>
          {pick.home_win_prob && (
            <p>Win Probability: {(pick.home_win_prob * 100).toFixed(1)}%</p>
          )}
          {pick.spread_cover_prob && (
            <p>Cover Probability: {(pick.spread_cover_prob * 100).toFixed(1)}%</p>
          )}
          {pick.over_prob && (
            <p>Over Probability: {(pick.over_prob * 100).toFixed(1)}%</p>
          )}
          {pick.recommended_kelly && (
            <p>Recommended Kelly: {(pick.recommended_kelly * 100).toFixed(1)}%</p>
          )}
        </div>
      ))}
    </div>
  );
}
```

**Usage Examples:**
```javascript
// NHL: Next 3 days
<PredictionsDisplay sport="NHL" viewMode="next-days" />

// NFL: Current week
<PredictionsDisplay sport="NFL" viewMode="week" />

// NBA: Latest
<PredictionsDisplay sport="NBA" viewMode="latest" />

// MLB: Next 7 days
<PredictionsDisplay sport="MLB" viewMode="next-days" />
```

### Step 4: Update Navigation
- Remove old "Bets" or "Recommendations" links
- Add new "Predictions" or "ML Picks" link
- Update routing to use new components

### Step 5: Update Settings/Admin Panel
- Add option to trigger workflow manually (if admin)
- Add health check display
- Show model status

### Step 6: Environment Variables
Add to Lovable environment:
```
VITE_RAILWAY_API_URL=https://moose-picks-api-production.up.railway.app/api
```

## 🎯 What the New System Provides

### Data Structure
Each prediction includes:
- `game_id` - Unique game identifier
- `home_team` / `away_team` - Team names
- `market` - moneyline, spread, or totals
- `side` - Which side to bet (home/away, over/under)
- `edge` - Calculated edge (0.05 = 5% edge)
- `home_win_prob` / `spread_cover_prob` / `over_prob` - Model probabilities
- `recommended_kelly` - Kelly criterion bet size
- `recommended_unit_size` - Unit size recommendation
- `date` - Game date
- `spread` / `over_under` - Betting lines

### Features
- ✅ Real-time predictions from ML models
- ✅ Edge calculation against market odds
- ✅ Kelly criterion bankroll management
- ✅ Multiple markets (moneyline, spread, totals)
- ✅ Multiple sports: **NFL, NHL, NBA, MLB**
- ✅ Filtered by minimum edge threshold
- ✅ Date range queries (next N days, current week, custom range)
- ✅ Sport-specific view modes:
  - **NHL**: Next 3 days (daily games)
  - **NFL**: Current week (weekly schedule)
  - **NBA**: Next 7 days or current week
  - **MLB**: Next 7 days (daily games)

## 🔍 Files to Check in Lovable

Look for and remove:
- `components/BetRecommendation.jsx` (or similar)
- `components/BetCard.jsx`
- `pages/Bets.jsx` or `pages/Recommendations.jsx`
- `services/betService.js` or `api/bets.js`
- Any mock data files with bet recommendations
- Old API endpoint configurations

## 📝 Integration Checklist

- [ ] Remove all old bet recommendation components
- [ ] Remove old API endpoints/calls
- [ ] Create new `PredictionsDisplay` component
- [ ] Update routing to use new component
- [ ] Add Railway API URL to environment variables
- [ ] Test fetching predictions from new API
- [ ] Update navigation/menu
- [ ] Add error handling for API failures
- [ ] Add loading states
- [ ] Test with real Railway API
- [ ] Update any documentation

## 🚨 Important Notes

1. **Keep User System Intact** - Don't touch authentication or user management
2. **Keep Whop Integration** - All payment/subscription code stays
3. **Keep Settings** - All user preferences and app settings remain
4. **Only Remove Bet Display Logic** - We're just changing where predictions come from
5. **New System is Read-Only** - Users view predictions, they don't generate them
6. **Admin Can Trigger Workflow** - But predictions come from Railway API

## 🎨 UI/UX Considerations

The new predictions should:
- Display clearly with edge percentage
- Show recommended bet size
- Filter by sport
- Show game date/time
- Display market type (moneyline/spread/totals)
- Show model confidence/probability

## 📞 Questions for Lovable

If you need clarification:
1. "Remove all components that display bet recommendations"
2. "Keep all user authentication and settings"
3. "Keep all Whop integration code"
4. "Replace bet data source with Railway API endpoint"
5. "New predictions come from: https://moose-picks-api-production.up.railway.app/api/predictions/latest"

## 🎯 Summary

**Remove:** Old bet recommendation display components and data fetching
**Keep:** Users, settings, Whop, core app structure
**Add:** New component that fetches from Railway API
**Result:** Same app, better predictions from ML backend

## 🏀 Multi-Sport Support

The API now supports **4 sports** with different view modes:

### Recommended View Modes by Sport:

1. **NHL** - Use `next-days` with `days=3`
   ```
   GET /api/predictions/next-days?sport=NHL&days=3
   ```
   - NHL games happen daily
   - 3 days gives good coverage

2. **NFL** - Use `week` for current week
   ```
   GET /api/predictions/week?sport=NFL
   ```
   - NFL games are weekly (Sunday/Monday)
   - Week view shows all games for the week

3. **NBA** - Use `next-days` with `days=7` or `week`
   ```
   GET /api/predictions/next-days?sport=NBA&days=7
   ```
   - NBA games happen daily
   - 7 days covers a full week

4. **MLB** - Use `next-days` with `days=7`
   ```
   GET /api/predictions/next-days?sport=MLB&days=7
   ```
   - MLB games happen daily
   - 7 days covers a full week

### All Sports Supported:
- ✅ **NFL** - Football (weekly schedule)
- ✅ **NHL** - Hockey (daily games)
- ✅ **NBA** - Basketball (daily games)
- ✅ **MLB** - Baseball (daily games)
