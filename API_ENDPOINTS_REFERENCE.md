# API Endpoints Reference - Multi-Sport Predictions

## Base URL
```
https://moose-picks-api-production.up.railway.app/api
```

## Supported Sports
- **NFL** - National Football League
- **NHL** - National Hockey League  
- **NBA** - National Basketball Association
- **MLB** - Major League Baseball

---

## Endpoint 1: Latest Predictions

Get the most recent predictions from the latest export file.

```
GET /api/predictions/latest?sport={SPORT}&limit={LIMIT}
```

**Parameters:**
- `sport` (required): NFL, NHL, NBA, or MLB
- `limit` (optional): Max predictions to return (default: 10)

**Examples:**
```bash
# Latest NFL predictions
GET /api/predictions/latest?sport=NFL&limit=20

# Latest NHL predictions
GET /api/predictions/latest?sport=NHL&limit=30

# Latest NBA predictions
GET /api/predictions/latest?sport=NBA&limit=25

# Latest MLB predictions
GET /api/predictions/latest?sport=MLB&limit=30
```

---

## Endpoint 2: Next N Days (Perfect for NHL, NBA, MLB)

Get predictions for the next N days starting from today.

```
GET /api/predictions/next-days?sport={SPORT}&days={DAYS}&limit={LIMIT}
```

**Parameters:**
- `sport` (required): NFL, NHL, NBA, or MLB
- `days` (optional): Number of days ahead (default: 3)
- `limit` (optional): Max predictions to return (default: 50)

**Recommended Usage:**

### NHL - Next 3 Days
```bash
GET /api/predictions/next-days?sport=NHL&days=3&limit=50
```
- NHL games happen daily
- 3 days covers upcoming games well

### NBA - Next 7 Days
```bash
GET /api/predictions/next-days?sport=NBA&days=7&limit=50
```
- NBA games happen daily
- 7 days covers a full week

### MLB - Next 7 Days
```bash
GET /api/predictions/next-days?sport=MLB&days=7&limit=50
```
- MLB games happen daily
- 7 days covers a full week

### NFL - Next 7 Days (Alternative)
```bash
GET /api/predictions/next-days?sport=NFL&days=7&limit=50
```
- Can use this, but `week` endpoint is better for NFL

---

## Endpoint 3: Current Week (Perfect for NFL)

Get predictions for the current week (Monday to Sunday).

```
GET /api/predictions/week?sport={SPORT}&limit={LIMIT}
```

**Parameters:**
- `sport` (required): NFL, NHL, NBA, or MLB
- `limit` (optional): Max predictions to return (default: 50)

**Recommended Usage:**

### NFL - Current Week
```bash
GET /api/predictions/week?sport=NFL&limit=50
```
- NFL games are weekly (primarily Sunday/Monday)
- Week view shows all games for the current week

### Other Sports
Can also use for NHL, NBA, MLB if you want week-based view:
```bash
GET /api/predictions/week?sport=NHL&limit=50
GET /api/predictions/week?sport=NBA&limit=50
GET /api/predictions/week?sport=MLB&limit=50
```

---

## Endpoint 4: Custom Date Range

Get predictions for any custom date range.

```
GET /api/predictions/date-range?sport={SPORT}&start_date={START}&end_date={END}&limit={LIMIT}
```

**Parameters:**
- `sport` (required): NFL, NHL, NBA, or MLB
- `start_date` (required): Start date in YYYY-MM-DD format
- `end_date` (optional): End date in YYYY-MM-DD format (defaults to start_date)
- `limit` (optional): Max predictions to return (default: 50)

**Examples:**
```bash
# Specific date range for NFL
GET /api/predictions/date-range?sport=NFL&start_date=2024-12-08&end_date=2024-12-15&limit=50

# Single day for NHL
GET /api/predictions/date-range?sport=NHL&start_date=2024-12-08&limit=30

# Next 2 weeks for NBA
GET /api/predictions/date-range?sport=NBA&start_date=2024-12-08&end_date=2024-12-22&limit=100
```

---

## Response Format

All endpoints return predictions in this format:

```json
{
  "sport": "NFL",
  "start_date": "2024-12-08",  // Only for date-range/next-days/week
  "end_date": "2024-12-14",    // Only for date-range/next-days/week
  "total_predictions": 25,
  "top_picks": [
    {
      "game_id": "401547456",
      "sport": "NFL",
      "market": "moneyline",
      "date": "2024-12-08T17:00:00",
      "home_team": "Kansas City Chiefs",
      "away_team": "Buffalo Bills",
      "model_version": "20241208_120000",
      
      // Market-specific fields
      "side": "home",  // or "away", "over", "under"
      "edge": 0.12,    // 12% edge
      
      // Probabilities (varies by market)
      "home_win_prob": 0.65,      // For moneyline
      "spread_cover_prob": 0.58,   // For spread
      "over_prob": 0.62,           // For totals
      
      // Betting lines
      "spread": -3.5,              // For spread market
      "over_under": 48.5,          // For totals market
      
      // Bankroll management
      "recommended_kelly": 0.05,   // Kelly criterion (5% of bankroll)
      "recommended_unit_size": 1.0  // Unit size recommendation
    }
  ]
}
```

---

## Quick Reference by Use Case

### "Show me NHL games for the next 3 days"
```bash
GET /api/predictions/next-days?sport=NHL&days=3&limit=50
```

### "Show me NFL games for this week"
```bash
GET /api/predictions/week?sport=NFL&limit=50
```

### "Show me NBA games for the next 7 days"
```bash
GET /api/predictions/next-days?sport=NBA&days=7&limit=50
```

### "Show me MLB games for the next 7 days"
```bash
GET /api/predictions/next-days?sport=MLB&days=7&limit=50
```

### "Show me latest predictions for any sport"
```bash
GET /api/predictions/latest?sport={SPORT}&limit=20
```

---

## JavaScript/React Examples

### NHL - Next 3 Days
```javascript
fetch('https://moose-picks-api-production.up.railway.app/api/predictions/next-days?sport=NHL&days=3&limit=50')
  .then(res => res.json())
  .then(data => {
    console.log(`Found ${data.total_predictions} NHL predictions`);
    data.top_picks.forEach(pick => {
      console.log(`${pick.home_team} vs ${pick.away_team} - ${pick.market} ${pick.side}`);
    });
  });
```

### NFL - Current Week
```javascript
fetch('https://moose-picks-api-production.up.railway.app/api/predictions/week?sport=NFL&limit=50')
  .then(res => res.json())
  .then(data => {
    console.log(`NFL Week: ${data.start_date} to ${data.end_date}`);
    console.log(`Found ${data.total_predictions} predictions`);
  });
```

### NBA - Next 7 Days
```javascript
fetch('https://moose-picks-api-production.up.railway.app/api/predictions/next-days?sport=NBA&days=7&limit=50')
  .then(res => res.json())
  .then(data => {
    console.log(`NBA Next 7 Days: ${data.start_date} to ${data.end_date}`);
  });
```

---

## Error Responses

### No Predictions Found
```json
{
  "detail": "No predictions found for NFL"
}
```
Status: 404

### Invalid Date Format
```json
{
  "detail": "Invalid date format. Use YYYY-MM-DD"
}
```
Status: 400

### Invalid Date Range
```json
{
  "detail": "end_date must be >= start_date"
}
```
Status: 400

---

## Notes

1. **Predictions are sorted by edge** - Highest edge first
2. **Only unsettled predictions** - Settled predictions are excluded
3. **All sports supported** - NFL, NHL, NBA, MLB
4. **Date ranges are inclusive** - Start and end dates are included
5. **Week starts Monday** - Week endpoint uses Monday-Sunday
