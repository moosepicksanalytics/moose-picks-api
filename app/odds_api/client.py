"""
The Odds API client for fetching real-time betting odds.
Supports NFL, NHL, NBA, MLB with 20k calls/month limit.
"""
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from app.config import settings


# The Odds API sport keys
ODDS_API_SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "NHL": "icehockey_nhl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
}

# The Odds API region (us, uk, au, etc.)
ODDS_API_REGION = "us"

# The Odds API markets
ODDS_API_MARKETS = {
    "moneyline": "h2h",  # Head-to-head (moneyline)
    "spread": "spreads",
    "totals": "totals",
    "over_under": "totals",
}


def fetch_odds_for_sport(
    sport: str,
    date: Optional[str] = None,
    markets: Optional[List[str]] = None
) -> List[Dict]:
    """
    Fetch odds from The Odds API for a sport and date.
    
    Args:
        sport: Sport code (NFL, NHL, NBA, MLB)
        date: Date in YYYY-MM-DD format (optional, defaults to today)
        markets: List of markets to fetch (default: ["h2h", "spreads", "totals"])
    
    Returns:
        List of game odds data
    """
    if not settings.ODDS_API_KEY:
        print("Warning: ODDS_API_KEY not set. Cannot fetch odds from The Odds API.")
        return []
    
    sport_key = ODDS_API_SPORT_KEYS.get(sport)
    if not sport_key:
        print(f"Warning: Sport {sport} not supported by The Odds API")
        return []
    
    # Default markets
    if markets is None:
        markets = ["h2h", "spreads", "totals"]
    
    # Format date (ISO format: YYYYMMDD or ISO8601)
    if date:
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            date_param = date_obj.strftime("%Y-%m-%dT12:00:00Z")  # ISO8601 format
        except ValueError:
            print(f"Invalid date format: {date}. Use YYYY-MM-DD")
            return []
    else:
        # Default to today
        date_param = datetime.now().strftime("%Y-%m-%dT12:00:00Z")
    
    # The Odds API endpoint
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    
    params = {
        "apiKey": settings.ODDS_API_KEY,
        "regions": ODDS_API_REGION,
        "markets": ",".join(markets),
        "dateFormat": "iso",
    }
    
    # Add date if specified
    if date:
        params["commenceTimeFrom"] = date_param
        # Add 24 hours for end time
        end_date = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
        params["commenceTimeTo"] = end_date.strftime("%Y-%m-%dT12:00:00Z")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Log API usage
        remaining = response.headers.get("x-requests-remaining", "unknown")
        used = response.headers.get("x-requests-used", "unknown")
        print(f"The Odds API: {used} used, {remaining} remaining this month")
        
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching odds from The Odds API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return []


def parse_odds_data(odds_data: List[Dict], sport: str) -> List[Dict]:
    """
    Parse The Odds API response into our format.
    
    Args:
        odds_data: Raw response from The Odds API
        sport: Sport code
    
    Returns:
        List of parsed odds dicts
    """
    parsed = []
    
    for game in odds_data:
        game_id = game.get("id", "")
        commence_time = game.get("commence_time", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        
        # Extract odds from bookmakers
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue
        
        # Use first bookmaker (or best odds - you can enhance this)
        bookmaker = bookmakers[0]
        markets_data = bookmaker.get("markets", [])
        
        odds_dict = {
            "game_id": game_id,
            "sport": sport,
            "commence_time": commence_time,
            "home_team": home_team,
            "away_team": away_team,
            "bookmaker": bookmaker.get("title", "unknown"),
        }
        
        # Parse moneyline (h2h)
        h2h_market = next((m for m in markets_data if m.get("key") == "h2h"), None)
        if h2h_market:
            outcomes = h2h_market.get("outcomes", [])
            for outcome in outcomes:
                if outcome.get("name") == home_team:
                    odds_dict["home_moneyline"] = outcome.get("price")
                elif outcome.get("name") == away_team:
                    odds_dict["away_moneyline"] = outcome.get("price")
        
        # Parse spreads
        spreads_market = next((m for m in markets_data if m.get("key") == "spreads"), None)
        if spreads_market:
            outcomes = spreads_market.get("outcomes", [])
            for outcome in outcomes:
                if outcome.get("name") == home_team:
                    odds_dict["home_spread"] = outcome.get("point")
                    odds_dict["home_spread_odds"] = outcome.get("price")
                elif outcome.get("name") == away_team:
                    odds_dict["away_spread"] = outcome.get("point")
                    odds_dict["away_spread_odds"] = outcome.get("price")
            
            # Set the spread line (home team's spread)
            if "home_spread" in odds_dict:
                odds_dict["spread"] = odds_dict["home_spread"]
        
        # Parse totals (over/under)
        totals_market = next((m for m in markets_data if m.get("key") == "totals"), None)
        if totals_market:
            outcomes = totals_market.get("outcomes", [])
            for outcome in outcomes:
                if outcome.get("name") == "Over":
                    odds_dict["over_under"] = outcome.get("point")
                    odds_dict["over_odds"] = outcome.get("price")
                elif outcome.get("name") == "Under":
                    odds_dict["under_odds"] = outcome.get("price")
        
        parsed.append(odds_dict)
    
    return parsed


def fetch_and_update_game_odds(
    sport: str,
    date: Optional[str] = None
) -> int:
    """
    Fetch odds from The Odds API and update games in database.
    
    Args:
        sport: Sport code
        date: Date in YYYY-MM-DD format (optional)
    
    Returns:
        Number of games updated
    """
    from app.database import SessionLocal
    from app.models.db_models import Game
    
    # Fetch odds
    odds_data = fetch_odds_for_sport(sport, date)
    if not odds_data:
        return 0
    
    # Parse odds
    parsed_odds = parse_odds_data(odds_data, sport)
    
    # Update database
    db = SessionLocal()
    updated_count = 0
    matched_games = []
    unmatched_odds = []
    
    try:
        print(f"  Attempting to match {len(parsed_odds)} odds entries to games...")
        
        for odds in parsed_odds:
            # Try to match by team names and date
            game_date = None
            if odds.get("commence_time"):
                try:
                    game_date = datetime.fromisoformat(odds["commence_time"].replace("Z", "+00:00"))
                except:
                    pass
            
            # Find matching game
            query = db.query(Game).filter(Game.sport == sport)
            
            if game_date:
                # Match by date (same day) - use date-only comparison
                from sqlalchemy import func
                date_only = game_date.date()
                query = query.filter(func.date(Game.date) == date_only)
            
            # Match by team names (fuzzy match - try multiple strategies)
            home_team = odds.get("home_team", "").strip()
            away_team = odds.get("away_team", "").strip()
            
            # Strategy 1: Exact match (case insensitive)
            game = query.filter(
                (Game.home_team.ilike(home_team)) &
                (Game.away_team.ilike(away_team))
            ).first()
            
            # Strategy 2: Contains match
            if not game:
                game = query.filter(
                    (Game.home_team.ilike(f"%{home_team}%")) &
                    (Game.away_team.ilike(f"%{away_team}%"))
                ).first()
            
            # Strategy 3: Reverse match (in case teams are swapped)
            if not game:
                game = query.filter(
                    (Game.home_team.ilike(f"%{away_team}%")) &
                    (Game.away_team.ilike(f"%{home_team}%"))
                ).first()
            
            if game:
                # Update odds
                updated = False
                if "home_moneyline" in odds and odds["home_moneyline"]:
                    game.home_moneyline = odds["home_moneyline"]
                    updated = True
                if "away_moneyline" in odds and odds["away_moneyline"]:
                    game.away_moneyline = odds["away_moneyline"]
                    updated = True
                if "spread" in odds and odds["spread"]:
                    game.spread = odds["spread"]
                    updated = True
                if "over_under" in odds and odds["over_under"]:
                    game.over_under = odds["over_under"]
                    updated = True
                
                if updated:
                    updated_count += 1
                    matched_games.append(f"{away_team} @ {home_team}")
            else:
                unmatched_odds.append(f"{away_team} @ {home_team}")
        
        db.commit()
        
        if updated_count > 0:
            print(f"  ✓ Updated odds for {updated_count} {sport} games")
            if matched_games:
                print(f"    Matched games: {', '.join(matched_games[:3])}")
        else:
            print(f"  ⚠️  No games matched for odds update")
            if unmatched_odds:
                print(f"    Unmatched odds: {', '.join(unmatched_odds[:3])}")
                # Debug: Show what games exist in DB
                existing_games = db.query(Game).filter(
                    Game.sport == sport,
                    func.date(Game.date) == datetime.now().date()
                ).limit(5).all()
                if existing_games:
                    print(f"    Existing games in DB: {[f'{g.away_team} @ {g.home_team}' for g in existing_games]}")
        
    except Exception as e:
        print(f"Error updating odds: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()
    
    return updated_count
