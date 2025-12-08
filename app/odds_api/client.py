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
    
    try:
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
                # Match by date (same day)
                start_of_day = game_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day = start_of_day + timedelta(days=1)
                query = query.filter(
                    Game.date >= start_of_day,
                    Game.date < end_of_day
                )
            
            # Match by team names (fuzzy match)
            home_team = odds.get("home_team", "")
            away_team = odds.get("away_team", "")
            
            game = query.filter(
                Game.home_team.ilike(f"%{home_team}%"),
                Game.away_team.ilike(f"%{away_team}%")
            ).first()
            
            if game:
                # Update odds
                if "home_moneyline" in odds:
                    game.home_moneyline = odds["home_moneyline"]
                if "away_moneyline" in odds:
                    game.away_moneyline = odds["away_moneyline"]
                if "spread" in odds:
                    game.spread = odds["spread"]
                if "over_under" in odds:
                    game.over_under = odds["over_under"]
                
                updated_count += 1
        
        db.commit()
        print(f"✓ Updated odds for {updated_count} {sport} games")
        
    except Exception as e:
        print(f"Error updating odds: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()
    
    return updated_count
