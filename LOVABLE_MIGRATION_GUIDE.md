# Lovable Migration Guide - Complete App Redesign

## 🎯 Migration Goal

**Complete redesign** of the betting/predictions system while preserving core infrastructure:
- ✅ Keep: Users, Settings, Whop Integration, Core App Structure
- ❌ Remove: All old bet recommendations, track record displays, history views
- 🔄 Replace: With new ML-powered predictions from Railway API

## ✅ PRESERVE (Do NOT Touch)

### 1. User Management System
- ✅ User authentication/login
- ✅ User registration
- ✅ User profiles
- ✅ User preferences
- ✅ User settings pages
- ✅ All user database tables/collections
- ✅ Session management
- ✅ Password reset functionality

### 2. Settings & Configuration
- ✅ App settings page
- ✅ User preferences
- ✅ Theme/UI settings (dark mode, etc.)
- ✅ Notification settings
- ✅ All configuration UI

### 3. Whop Integration (CRITICAL - DO NOT REMOVE)
- ✅ Whop payment processing
- ✅ Whop subscription management
- ✅ Whop webhooks
- ✅ Whop API connections
- ✅ Subscription status checks
- ✅ Payment history (if exists)
- ✅ All Whop-related components
- ✅ All Whop-related API calls

### 4. Core App Infrastructure
- ✅ Navigation/routing system
- ✅ Layout components (header, footer, sidebar)
- ✅ Authentication flow
- ✅ Protected routes
- ✅ Basic UI components (buttons, forms, inputs, cards)
- ✅ Database connection setup
- ✅ Error handling
- ✅ Loading states

## ❌ REMOVE COMPLETELY

### 1. Bet Recommendation System
- ❌ All bet recommendation components
- ❌ Old prediction cards/displays
- ❌ Recommendation lists
- ❌ Betting suggestion UI
- ❌ Any hardcoded bet data
- ❌ Mock/fake prediction data

### 2. Track Record & History Displays
- ❌ **Track record pages** (win/loss history displays)
- ❌ **History views** (past predictions display)
- ❌ **Performance charts** (if showing user betting history)
- ❌ **Statistics pages** (if showing user track record)
- ❌ **Past results displays**
- ❌ **Win/loss counters** (user-facing)
- ❌ **ROI displays** (user-facing)
- ❌ Any components showing "Your betting history" or "Your track record"

**IMPORTANT:** Remove the **DISPLAY** of track records, but the **backend functionality** for tracking and settling predictions should remain (it runs automatically on Railway).

### 3. Old Data Sources
- ❌ Old API endpoints for recommendations
- ❌ Old prediction generation logic (frontend)
- ❌ Old data fetching for bets
- ❌ Old betting calculation logic

### 4. Old Betting UI
- ❌ Old bet cards
- ❌ Old recommendation lists
- ❌ Old prediction displays
- ❌ Old edge calculation displays (if in frontend)

## 🔄 NEW SYSTEM - Railway ML API

### API Base URL
```
https://moose-picks-api-production.up.railway.app/api
```

### Supported Sports
- **NFL** - National Football League
- **NHL** - National Hockey League
- **NBA** - National Basketball Association
- **MLB** - Major League Baseball

### Endpoints

#### 1. Get Predictions by Sport & View Mode

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

**Latest Predictions (Any Sport):**
```
GET /api/predictions/latest?sport={SPORT}&limit=20
```

**Custom Date Range:**
```
GET /api/predictions/date-range?sport={SPORT}&start_date=2024-12-08&end_date=2024-12-15&limit=50
```

#### 2. Health Check
```
GET /api/health
```

Returns:
```json
{
  "status": "healthy",
  "database": "connected",
  "games_in_db": 1250,
  "models_trained": 12,
  "timestamp": "2024-12-08T12:00:00"
}
```

## 📋 Migration Steps

### Phase 1: Remove Old System

1. **Delete Components:**
   - Search for: `BetRecommendation`, `BetCard`, `PredictionCard`, `RecommendationList`
   - Search for: `TrackRecord`, `History`, `Performance`, `Stats`, `WinLoss`
   - Delete all found components

2. **Remove Routes:**
   - Remove routes to `/bets`, `/recommendations`, `/predictions` (old)
   - Remove routes to `/history`, `/track-record`, `/stats`, `/performance`
   - Remove navigation links to these pages

3. **Remove API Calls:**
   - Find all API calls to old endpoints
   - Remove old data fetching logic
   - Clean up unused imports

4. **Remove Mock Data:**
   - Delete any mock bet data files
   - Remove hardcoded prediction arrays
   - Remove fake history data

### Phase 2: Create New Predictions System

#### New Component Structure

**1. Main Predictions Page: `PredictionsPage.jsx`**
```javascript
import { useState } from 'react';
import { PredictionsDisplay } from '../components/PredictionsDisplay';
import { SportSelector } from '../components/SportSelector';

export function PredictionsPage() {
  const [selectedSport, setSelectedSport] = useState('NFL');
  
  return (
    <div className="predictions-page">
      <h1>ML-Powered Predictions</h1>
      
      <SportSelector 
        selectedSport={selectedSport}
        onSportChange={setSelectedSport}
      />
      
      <PredictionsDisplay 
        sport={selectedSport}
        viewMode={selectedSport === 'NFL' ? 'week' : selectedSport === 'NHL' ? 'next-days' : 'next-days'}
      />
    </div>
  );
}
```

**2. Predictions Display Component: `PredictionsDisplay.jsx`**
```javascript
import { useState, useEffect } from 'react';

const API_BASE = 'https://moose-picks-api-production.up.railway.app/api';

export function PredictionsDisplay({ 
  sport = 'NFL', 
  viewMode = 'latest'
}) {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let url;
    
    // Sport-specific view modes
    if (sport === 'NHL' && viewMode === 'next-days') {
      url = `${API_BASE}/predictions/next-days?sport=NHL&days=3&limit=50`;
    } else if (sport === 'NFL' && viewMode === 'week') {
      url = `${API_BASE}/predictions/week?sport=NFL&limit=50`;
    } else if (sport === 'NBA' && viewMode === 'next-days') {
      url = `${API_BASE}/predictions/next-days?sport=NBA&days=7&limit=50`;
    } else if (sport === 'MLB' && viewMode === 'next-days') {
      url = `${API_BASE}/predictions/next-days?sport=MLB&days=7&limit=50`;
    } else {
      url = `${API_BASE}/predictions/latest?sport=${sport}&limit=20`;
    }

    setLoading(true);
    fetch(url)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        setPredictions(data.top_picks || []);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [sport, viewMode]);

  if (loading) {
    return (
      <div className="loading">
        <p>Loading {sport} predictions...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error">
        <p>Error loading predictions: {error}</p>
        <p>Make sure the Railway API is running.</p>
      </div>
    );
  }

  if (!predictions.length) {
    return (
      <div className="no-predictions">
        <p>No predictions available for {sport} at this time.</p>
        <p>Predictions are generated daily. Check back later!</p>
      </div>
    );
  }

  return (
    <div className="predictions-container">
      <div className="predictions-header">
        <h2>
          {sport} Predictions
          {viewMode === 'week' && ' - This Week'}
          {viewMode === 'next-days' && ` - Next ${sport === 'NHL' ? '3' : '7'} Days`}
        </h2>
        <p className="prediction-count">
          {predictions.length} predictions found
        </p>
      </div>

      <div className="predictions-grid">
        {predictions.map((pick, idx) => (
          <PredictionCard key={pick.game_id || idx} prediction={pick} />
        ))}
      </div>
    </div>
  );
}
```

**3. Prediction Card Component: `PredictionCard.jsx`**
```javascript
export function PredictionCard({ prediction }) {
  const {
    home_team,
    away_team,
    date,
    market,
    side,
    edge,
    home_win_prob,
    spread_cover_prob,
    over_prob,
    recommended_kelly,
    spread,
    over_under
  } = prediction;

  // Format edge as percentage
  const edgePercent = (edge * 100).toFixed(1);
  
  // Get probability based on market
  let probability = 0;
  let marketLabel = '';
  
  if (market === 'moneyline') {
    probability = home_win_prob ? (home_win_prob * 100).toFixed(1) : 0;
    marketLabel = `Moneyline - ${side === 'home' ? home_team : away_team}`;
  } else if (market === 'spread') {
    probability = spread_cover_prob ? (spread_cover_prob * 100).toFixed(1) : 0;
    marketLabel = `Spread - ${side === 'home' ? home_team : away_team} ${spread ? (spread > 0 ? '+' : '') + spread : ''}`;
  } else if (market === 'totals' || market === 'over_under') {
    probability = over_prob ? (over_prob * 100).toFixed(1) : 0;
    marketLabel = `Totals - ${side.toUpperCase()} ${over_under || ''}`;
  }

  return (
    <div className="prediction-card">
      <div className="game-header">
        <h3>{home_team} vs {away_team}</h3>
        {date && (
          <p className="game-date">
            {new Date(date).toLocaleDateString('en-US', {
              weekday: 'short',
              month: 'short',
              day: 'numeric',
              hour: 'numeric',
              minute: '2-digit'
            })}
          </p>
        )}
      </div>

      <div className="prediction-details">
        <div className="market-info">
          <span className="market-type">{marketLabel}</span>
        </div>

        <div className="edge-badge">
          <span className="edge-label">Edge</span>
          <span className="edge-value">{edgePercent}%</span>
        </div>

        <div className="probability">
          <span>Win Probability: {probability}%</span>
        </div>

        {recommended_kelly && (
          <div className="kelly">
            <span>Recommended Bet: {(recommended_kelly * 100).toFixed(1)}% of bankroll</span>
          </div>
        )}
      </div>
    </div>
  );
}
```

**4. Sport Selector Component: `SportSelector.jsx`**
```javascript
export function SportSelector({ selectedSport, onSportChange }) {
  const sports = [
    { code: 'NFL', name: 'NFL', viewMode: 'week' },
    { code: 'NHL', name: 'NHL', viewMode: 'next-days' },
    { code: 'NBA', name: 'NBA', viewMode: 'next-days' },
    { code: 'MLB', name: 'MLB', viewMode: 'next-days' }
  ];

  return (
    <div className="sport-selector">
      {sports.map(sport => (
        <button
          key={sport.code}
          className={selectedSport === sport.code ? 'active' : ''}
          onClick={() => onSportChange(sport.code)}
        >
          {sport.name}
        </button>
      ))}
    </div>
  );
}
```

### Phase 3: Update Navigation

1. **Remove Old Links:**
   - Remove "Bets", "Recommendations", "History", "Track Record", "Stats" from navigation

2. **Add New Links:**
   - Add "Predictions" or "ML Picks" to main navigation
   - Add sport filter in predictions page

3. **Update Routes:**
   ```javascript
   // Remove old routes
   // <Route path="/bets" component={OldBetsPage} />
   // <Route path="/history" component={HistoryPage} />
   
   // Add new route
   <Route path="/predictions" component={PredictionsPage} />
   ```

### Phase 4: Environment Setup

Add to Lovable environment variables:
```
VITE_RAILWAY_API_URL=https://moose-picks-api-production.up.railway.app/api
```

Then use in code:
```javascript
const API_BASE = import.meta.env.VITE_RAILWAY_API_URL || 'https://moose-picks-api-production.up.railway.app/api';
```

## 🎨 UI/UX Guidelines

### New Predictions Display Should:

1. **Show Clear Information:**
   - Team names prominently
   - Game date/time
   - Market type (moneyline/spread/totals)
   - Side to bet (home/away/over/under)
   - Edge percentage (highlighted)
   - Win probability
   - Recommended bet size (Kelly criterion)

2. **Visual Hierarchy:**
   - Edge percentage should be most prominent
   - Use color coding: Green for high edge, yellow for medium, red for low
   - Sort by edge (highest first)

3. **Filtering:**
   - Filter by sport (NFL, NHL, NBA, MLB)
   - Filter by market type (optional)
   - Filter by minimum edge (optional)

4. **No History/Track Record:**
   - **DO NOT** show past performance
   - **DO NOT** show win/loss records
   - **DO NOT** show ROI or PnL
   - **DO NOT** show "Your betting history"
   - Focus only on **current/future predictions**

## 📊 Data Structure

Each prediction from the API includes:

```json
{
  "game_id": "401547456",
  "sport": "NFL",
  "market": "moneyline",
  "date": "2024-12-08T17:00:00",
  "home_team": "Kansas City Chiefs",
  "away_team": "Buffalo Bills",
  "side": "home",
  "edge": 0.12,
  "home_win_prob": 0.65,
  "spread_cover_prob": 0.58,
  "over_prob": 0.62,
  "spread": -3.5,
  "over_under": 48.5,
  "recommended_kelly": 0.05,
  "recommended_unit_size": 1.0,
  "model_version": "20241208_120000"
}
```

## 🚨 Critical Notes

### What to Remove:
- ❌ **ALL** track record displays
- ❌ **ALL** history views
- ❌ **ALL** past performance charts
- ❌ **ALL** win/loss counters (user-facing)
- ❌ **ALL** ROI displays (user-facing)
- ❌ **ALL** "Your betting history" pages
- ❌ **ALL** statistics pages showing user performance

### What to Keep (Backend Only):
- ✅ Backend prediction tracking (runs on Railway automatically)
- ✅ Prediction settling (runs on Railway automatically)
- ✅ Database storage (for future analytics, not user display)

**Key Point:** The backend still tracks everything, but users should **NOT see** their track record or history. This is a fresh start for the UI while keeping the backend functionality.

## 📝 Complete Checklist

### Remove:
- [ ] All bet recommendation components
- [ ] All track record/history display components
- [ ] All statistics/performance display components
- [ ] Old API endpoints/calls
- [ ] Old routes (`/bets`, `/history`, `/track-record`, `/stats`)
- [ ] Navigation links to removed pages
- [ ] Mock/fake data files

### Create:
- [ ] New `PredictionsPage` component
- [ ] New `PredictionsDisplay` component
- [ ] New `PredictionCard` component
- [ ] New `SportSelector` component
- [ ] New route `/predictions`
- [ ] Environment variable for API URL

### Update:
- [ ] Navigation menu
- [ ] Routing configuration
- [ ] Environment variables
- [ ] Error handling for API failures
- [ ] Loading states

### Keep (DO NOT TOUCH):
- [ ] User authentication system
- [ ] User management
- [ ] Settings pages
- [ ] Whop integration (all of it)
- [ ] Core app structure
- [ ] Layout components

## 🎯 Summary

**Remove:** All old betting UI, track records, history displays
**Keep:** Users, settings, Whop, core structure
**Add:** New ML-powered predictions from Railway API
**Result:** Fresh, modern predictions interface with no history/track record displays

The backend still tracks everything automatically on Railway, but users get a clean slate with only current/future predictions visible.

## 📞 For Lovable Team

**Key Instructions:**
1. Remove ALL track record/history displays (but backend tracking continues automatically)
2. Keep ALL user, settings, and Whop functionality
3. Replace bet recommendations with new Railway API predictions
4. Support all 4 sports (NFL, NHL, NBA, MLB) with sport-specific view modes
5. Focus on current/future predictions only - no past performance shown to users

**API Endpoint:**
```
https://moose-picks-api-production.up.railway.app/api
```

**Test Endpoints:**
- Health: `GET /api/health`
- NHL (3 days): `GET /api/predictions/next-days?sport=NHL&days=3`
- NFL (week): `GET /api/predictions/week?sport=NFL`
- NBA/MLB (7 days): `GET /api/predictions/next-days?sport=NBA&days=7`
